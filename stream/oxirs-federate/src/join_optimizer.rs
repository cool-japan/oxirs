//! Advanced Join Optimization Algorithms
//!
//! This module implements sophisticated join optimization algorithms for federated queries,
//! including distributed join planning, adaptive execution, and cost-based optimization.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    planner::planning::{ExecutionStep, FilterExpression as PlanningFilterExpression, TriplePattern},
    service_optimizer::{
        QueryInfo, StarJoinPattern, StarOptimizationType, ChainJoinPattern, ChainOptimizationType, 
        CyclePattern, BushyTreeNode, BushyNodeType, JoinExecutionStrategy, JoinEdge,
        JoinPlan, SpecialJoinPatterns, JoinExecutionResult, StrategyPerformance,
        JoinOperation, JoinOperationType, JoinAlgorithm, ParallelizationOpportunity
    },
    ServiceRegistry,
};

/// Advanced join optimizer for distributed federated queries
#[derive(Debug)]
pub struct DistributedJoinOptimizer {
    config: JoinOptimizerConfig,
    statistics: Arc<RwLock<JoinStatistics>>,
    cost_model: JoinCostModel,
    adaptive_controller: AdaptiveExecutionController,
}

/// Configuration for the join optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinOptimizerConfig {
    pub enable_star_join_detection: bool,
    pub enable_chain_optimization: bool,
    pub enable_bushy_trees: bool,
    pub max_join_order_enumeration: usize,
    pub cost_threshold_for_reoptimization: f64,
    pub adaptive_execution_enabled: bool,
    pub memory_budget_mb: usize,
    pub parallelism_factor: f64,
}

impl Default for JoinOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_star_join_detection: true,
            enable_chain_optimization: true,
            enable_bushy_trees: true,
            max_join_order_enumeration: 12,
            cost_threshold_for_reoptimization: 2.0,
            adaptive_execution_enabled: true,
            memory_budget_mb: 1024,
            parallelism_factor: 0.8,
        }
    }
}

/// Distributed join planning algorithms
impl DistributedJoinOptimizer {
    /// Create a new distributed join optimizer
    pub fn new(config: JoinOptimizerConfig) -> Self {
        Self {
            config,
            statistics: Arc::new(RwLock::new(JoinStatistics::default())),
            cost_model: JoinCostModel::new(),
            adaptive_controller: AdaptiveExecutionController::new(),
        }
    }

    /// Optimize join order for distributed execution
    pub async fn optimize_join_order(
        &self,
        patterns: &[TriplePattern],
        filters: &[PlanningFilterExpression],
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        info!("Optimizing join order for {} patterns", patterns.len());

        // Step 1: Analyze join graph structure
        let join_graph = self.build_join_graph(patterns, filters).await?;

        // Step 2: Detect special join patterns
        let special_patterns = self.detect_special_patterns(&join_graph).await?;

        // Step 3: Generate optimization strategy based on detected patterns
        let optimization_strategy = self.select_optimization_strategy(&special_patterns).await?;

        // Step 4: Apply the selected optimization strategy
        let optimized_plan = match optimization_strategy {
            JoinOptimizationStrategy::StarJoin => {
                self.optimize_star_join(&join_graph, registry).await?
            }
            JoinOptimizationStrategy::ChainJoin => {
                self.optimize_chain_join(&join_graph, registry).await?
            }
            JoinOptimizationStrategy::BushyTree => {
                self.optimize_bushy_tree(&join_graph, registry).await?
            }
            JoinOptimizationStrategy::Dynamic => {
                self.dynamic_join_optimization(&join_graph, registry)
                    .await?
            }
        };

        // Step 5: Apply adaptive execution optimizations
        let adaptive_plan = if self.config.adaptive_execution_enabled {
            self.apply_adaptive_optimizations(optimized_plan, registry)
                .await?
        } else {
            optimized_plan
        };

        info!(
            "Generated join plan with {} operations",
            adaptive_plan.operations.len()
        );
        Ok(adaptive_plan)
    }

    /// Build join graph from triple patterns
    async fn build_join_graph(
        &self,
        patterns: &[TriplePattern],
        filters: &[PlanningFilterExpression],
    ) -> Result<JoinGraph> {
        let mut graph = JoinGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
        };

        // Create nodes for each pattern
        for (idx, pattern) in patterns.iter().enumerate() {
            let node = JoinNode {
                id: format!("pattern_{}", idx),
                pattern: pattern.clone(),
                variables: self.extract_variables(pattern),
                selectivity: self.estimate_selectivity(pattern).await?,
                estimated_cardinality: self.estimate_cardinality(pattern).await?,
                execution_cost: self.estimate_execution_cost(pattern).await?,
            };

            // Variables are now tracked per node
            graph.nodes.push(node);
        }

        // Create edges for variable connections
        for node1 in &graph.nodes {
            for node2 in &graph.nodes {
                if node1.id != node2.id {
                    let shared_vars = node1
                        .variables
                        .intersection(&node2.variables)
                        .cloned()
                        .collect::<Vec<_>>();

                    if !shared_vars.is_empty() {
                        let edge = JoinEdge {
                            from_node: node1.id.clone(),
                            to_node: node2.id.clone(),
                            shared_variables: shared_vars,
                            join_selectivity: self
                                .estimate_join_selectivity(&node1, &node2)
                                .await?,
                            estimated_cost: self
                                .estimate_join_cost(1000, 1000) // placeholder values
                                .await?,
                        };
                        graph.edges.push(edge);
                    }
                }
            }
        }

        // Apply filter pushdown analysis
        for filter in filters {
            self.analyze_filter_pushdown(&mut graph, filter).await?;
        }

        Ok(graph)
    }

    /// Detect special join patterns (star, chain, etc.)
    async fn detect_special_patterns(&self, graph: &JoinGraph) -> Result<SpecialJoinPatterns> {
        debug!("Detecting special join patterns in graph");

        let mut star_joins = Vec::new();
        let mut chain_joins = Vec::new();
        let mut cycles = Vec::new();

        // Star join detection
        if self.config.enable_star_join_detection {
            star_joins = self.detect_star_joins(graph).await?;
        }

        // Chain join detection
        if self.config.enable_chain_optimization {
            chain_joins = self.detect_chain_joins(graph).await?;
        }

        // Cycle detection for complex graph structures
        cycles = self.detect_cycles(graph).await?;

        Ok(SpecialJoinPatterns {
            star_joins,
            chain_joins,
            cycles,
            total_patterns: graph.nodes.len(),
        })
    }

    /// Star join detection algorithm
    async fn detect_star_joins(&self, graph: &JoinGraph) -> Result<Vec<StarJoinPattern>> {
        let mut star_patterns = Vec::new();

        for node in &graph.nodes {
            // Count connections to this node
            let connections: Vec<&JoinEdge> = graph
                .edges
                .iter()
                .filter(|e| e.from_node == node.id || e.to_node == node.id)
                .collect();

            // A star join has one central node connected to multiple others
            if connections.len() >= 3 {
                let connected_nodes: Vec<String> = connections
                    .iter()
                    .map(|e| {
                        if e.from_node == node.id {
                            e.to_node.clone()
                        } else {
                            e.from_node.clone()
                        }
                    })
                    .collect();

                let star_pattern = StarJoinPattern {
                    center_node: node.id.clone(),
                    connected_nodes,
                    estimated_benefit: self.calculate_star_join_benefit_from_edges(node, &connections).await?,
                    optimization_type: StarOptimizationType::MultiWayJoin,
                };

                star_patterns.push(star_pattern);
            }
        }

        // Sort by estimated benefit
        star_patterns.sort_by(|a, b| {
            b.estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!("Detected {} star join patterns", star_patterns.len());
        Ok(star_patterns)
    }

    /// Chain join detection algorithm
    async fn detect_chain_joins(&self, graph: &JoinGraph) -> Result<Vec<ChainJoinPattern>> {
        let mut chain_patterns = Vec::new();
        let mut visited = HashSet::new();

        for start_node in &graph.nodes {
            if visited.contains(&start_node.id) {
                continue;
            }

            let chain = self.find_longest_chain(graph, &start_node.id, &mut visited).await?;

            if chain.len() >= 3 {
                // Only consider chains of length 3 or more
                let chain_pattern = ChainJoinPattern {
                    node_sequence: chain.clone(),
                    estimated_benefit: self.calculate_chain_join_benefit(&chain).await?,
                    optimization_type: ChainOptimizationType::PipelinedExecution,
                };

                chain_patterns.push(chain_pattern);
            }
        }

        debug!("Detected {} chain join patterns", chain_patterns.len());
        Ok(chain_patterns)
    }

    /// Cycle detection using DFS
    async fn detect_cycles(&self, graph: &JoinGraph) -> Result<Vec<CyclePattern>> {
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node in &graph.nodes {
            if !visited.contains(&node.id) {
                self.dfs_cycle_detection(
                    graph,
                    &node.id,
                    &mut visited,
                    &mut rec_stack,
                    &mut cycles,
                    &mut Vec::new(),
                )
                .await?;
            }
        }

        debug!("Detected {} cycles in join graph", cycles.len());
        Ok(cycles)
    }

    /// DFS helper for cycle detection
    async fn dfs_cycle_detection(
        &self,
        graph: &JoinGraph,
        node_id: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
        cycles: &mut Vec<CyclePattern>,
        current_path: &mut Vec<String>,
    ) -> Result<()> {
        visited.insert(node_id.to_string());
        rec_stack.insert(node_id.to_string());
        current_path.push(node_id.to_string());

        // Find all adjacent nodes
        for edge in &graph.edges {
            let adjacent_node = if edge.from_node == node_id {
                &edge.to_node
            } else if edge.to_node == node_id {
                &edge.from_node
            } else {
                continue;
            };

            if !visited.contains(adjacent_node) {
                Box::pin(self.dfs_cycle_detection(
                    graph,
                    adjacent_node,
                    visited,
                    rec_stack,
                    cycles,
                    current_path,
                ))
                .await?;
            } else if rec_stack.contains(adjacent_node) {
                // Found a cycle
                if let Some(cycle_start) = current_path.iter().position(|n| n == adjacent_node) {
                    let cycle_nodes = current_path[cycle_start..].to_vec();
                    cycles.push(CyclePattern {
                        nodes: cycle_nodes,
                        complexity_score: current_path.len() as f64,
                    });
                }
            }
        }

        rec_stack.remove(node_id);
        current_path.pop();
        Ok(())
    }

    /// Bushy tree optimization for parallel execution
    async fn optimize_bushy_tree(
        &self,
        graph: &JoinGraph,
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        debug!("Optimizing with bushy tree algorithm");

        let patterns: Vec<TriplePattern> = graph.nodes.iter().map(|n| n.pattern.clone()).collect();

        if patterns.is_empty() {
            return Err(anyhow!("Cannot optimize empty pattern set"));
        }

        // Build optimal bushy tree using dynamic programming
        let tree = self.build_optimal_bushy_tree(&patterns, registry).await?;

        // Convert tree to execution plan
        let operations = self.tree_to_execution_operations(&tree).await?;

        Ok(JoinPlan {
            operations,
            estimated_cost: tree.total_cost,
            parallelization_opportunities: self.identify_parallelization_opportunities(&tree).await?,
            execution_strategy: JoinExecutionStrategy::BushyTree,
            memory_requirements: self.calculate_memory_requirements(&tree).await?,
        })
    }

    /// Build optimal bushy tree using dynamic programming
    async fn build_optimal_bushy_tree(
        &self,
        patterns: &[TriplePattern],
        registry: &ServiceRegistry,
    ) -> Result<BushyTreeNode> {
        let n = patterns.len();

        if n == 1 {
            return Ok(BushyTreeNode {
                node_type: BushyNodeType::Leaf,
                patterns: vec![patterns[0].clone()],
                left_child: None,
                right_child: None,
                total_cost: self.estimate_pattern_cost(&patterns[0]).await?,
                parallelization_factor: 1.0,
            });
        }

        // Dynamic programming table: dp[mask] = optimal tree for pattern subset
        let mut dp: HashMap<u32, BushyTreeNode> = HashMap::new();

        // Initialize single patterns
        for i in 0..n {
            let mask = 1u32 << i;
            dp.insert(
                mask,
                BushyTreeNode {
                    node_type: BushyNodeType::Leaf,
                    patterns: vec![patterns[i].clone()],
                    left_child: None,
                    right_child: None,
                    total_cost: self.estimate_pattern_cost(&patterns[i]).await?,
                    parallelization_factor: 1.0,
                },
            );
        }

        // Build larger subsets
        for subset_size in 2..=n {
            for mask in 1u32..(1u32 << n) {
                if mask.count_ones() != subset_size as u32 {
                    continue;
                }

                let mut best_cost = f64::INFINITY;
                let mut best_tree = None;

                // Try all possible ways to split this subset
                let mut submask = mask;
                while submask > 0 {
                    if submask != mask && dp.contains_key(&submask) {
                        let complement = mask ^ submask;
                        if complement > 0 && dp.contains_key(&complement) {
                            let left_tree = &dp[&submask];
                            let right_tree = &dp[&complement];

                            let join_cost = self
                                .estimate_bushy_join_cost(left_tree, right_tree, registry)
                                .await?;
                            let total_cost = left_tree.total_cost + right_tree.total_cost + join_cost;

                            if total_cost < best_cost {
                                best_cost = total_cost;

                                let combined_patterns: Vec<TriplePattern> = left_tree
                                    .patterns
                                    .iter()
                                    .chain(right_tree.patterns.iter())
                                    .cloned()
                                    .collect();

                                best_tree = Some(BushyTreeNode {
                                    node_type: BushyNodeType::InnerJoin,
                                    patterns: combined_patterns,
                                    left_child: Some(Box::new(left_tree.clone())),
                                    right_child: Some(Box::new(right_tree.clone())),
                                    total_cost,
                                    parallelization_factor: self
                                        .calculate_parallelization_factor(left_tree, right_tree)
                                        .await?,
                                });
                            }
                        }
                    }

                    submask = (submask - 1) & mask;
                }

                if let Some(tree) = best_tree {
                    dp.insert(mask, tree);
                }
            }
        }

        // Return the tree for all patterns
        let full_mask = (1u32 << n) - 1;
        dp.remove(&full_mask)
            .ok_or_else(|| anyhow!("Failed to build optimal bushy tree"))
    }

    /// Runtime statistics collection for adaptive execution
    pub async fn collect_runtime_statistics(
        &mut self,
        execution_result: &JoinExecutionResult,
    ) -> Result<()> {
        debug!("Collecting runtime statistics");

        let mut stats = self.statistics.write().await;

        // Update execution counts
        stats.total_executions += 1;

        // Update strategy performance
        let strategy_stats = stats
            .strategy_performance
            .entry(execution_result.strategy.clone())
            .or_insert_with(StrategyPerformance::default);

        strategy_stats.execution_count += 1;
        strategy_stats.total_execution_time += execution_result.execution_time;
        strategy_stats.avg_execution_time =
            strategy_stats.total_execution_time / strategy_stats.execution_count as f64;

        if execution_result.success {
            strategy_stats.success_count += 1;
        }
        strategy_stats.success_rate =
            strategy_stats.success_count as f64 / strategy_stats.execution_count as f64;

        // Update cardinality statistics
        strategy_stats.cardinality_samples.push(execution_result.result_cardinality);
        if strategy_stats.cardinality_samples.len() > 1000 {
            strategy_stats.cardinality_samples.remove(0); // Keep only recent samples
        }

        // Update memory usage statistics  
        strategy_stats.memory_usage_samples.push(execution_result.memory_used);
        if strategy_stats.memory_usage_samples.len() > 1000 {
            strategy_stats.memory_usage_samples.remove(0);
        }

        // Trigger adaptive reconfiguration if needed
        if stats.total_executions % 100 == 0 {
            self.adaptive_controller.analyze_and_adapt(&stats).await?;
        }

        Ok(())
    }

    /// Dynamic algorithm switching based on runtime feedback
    pub async fn recommend_execution_strategy(
        &self,
        patterns: &[TriplePattern],
        estimated_cardinality: u64,
        registry: &ServiceRegistry,
    ) -> Result<JoinExecutionStrategy> {
        debug!(
            "Recommending execution strategy for {} patterns, estimated cardinality: {}",
            patterns.len(),
            estimated_cardinality
        );

        let stats = self.statistics.read().await;

        // Find the best performing strategy for similar cardinality
        let mut best_strategy = JoinExecutionStrategy::Sequential;
        let mut best_score = 0.0;

        for (strategy, performance) in &stats.strategy_performance {
            let cardinality_match = self
                .calculate_cardinality_match_score(estimated_cardinality, &performance.cardinality_samples)
                .await?;

            if cardinality_match > 0.5 {
                // Only consider strategies that have worked for similar cardinalities
                let score = performance.success_rate / (performance.avg_execution_time + 1.0);
                if score > best_score {
                    best_score = score;
                    best_strategy = strategy.clone();
                }
            }
        }

        // Fallback to heuristic-based selection if no historical data
        if best_score == 0.0 {
            best_strategy = self.heuristic_strategy_selection(patterns, estimated_cardinality).await?;
        }

        debug!("Recommended strategy: {:?}", best_strategy);
        Ok(best_strategy)
    }

    /// Build a join graph from query patterns (compatibility method)
    async fn build_join_graph_from_patterns(&self, patterns: &[TriplePattern]) -> Result<JoinGraph> {
        let mut graph = JoinGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
        };

        // Create nodes for each pattern
        for (i, pattern) in patterns.iter().enumerate() {
            let variables = self.get_pattern_variables(pattern);
            let node = JoinNode {
                id: format!("pattern_{}", i),
                pattern: pattern.clone(),
                variables,
                estimated_cardinality: self.estimate_pattern_cardinality(pattern).await?,
                selectivity: self.estimate_pattern_selectivity(pattern).await?,
                execution_cost: self.estimate_pattern_cost(pattern).await?,
            };
            graph.nodes.push(node);
        }

        // Create edges between nodes that share variables
        for i in 0..graph.nodes.len() {
            for j in i + 1..graph.nodes.len() {
                let node1 = &graph.nodes[i];
                let node2 = &graph.nodes[j];
                let shared_vars: Vec<String> = node1
                    .variables
                    .intersection(&node2.variables)
                    .cloned()
                    .collect();
                if !shared_vars.is_empty() {
                        let edge = JoinEdge {
                            from_node: node1.id.clone(),
                            to_node: node2.id.clone(),
                            shared_variables: shared_vars,
                            join_selectivity: self
                                .estimate_join_selectivity(&node1, &node2)
                                .await?,
                            estimated_cost: self
                                .estimate_join_cost(node1.estimated_cardinality, node2.estimated_cardinality)
                                .await?,
                        };
                        graph.edges.push(edge);
                    }
            }
        }

        Ok(graph)
    }

    /// Detect special join patterns (star, chain, etc.) - compatibility method
    async fn detect_special_patterns_compatibility(&self, graph: &JoinGraph) -> Result<SpecialJoinPatterns> {
        let star_joins = if self.config.enable_star_join_detection {
            self.detect_star_joins(graph).await?
        } else {
            Vec::new()
        };

        let chain_joins = if self.config.enable_chain_optimization {
            self.detect_chain_joins(graph).await?
        } else {
            Vec::new()
        };

        let cycles = self.detect_cycles(graph).await?;

        Ok(SpecialJoinPatterns {
            star_joins,
            chain_joins,
            cycles,
            total_patterns: graph.nodes.len(),
        })
    }

    /// Detect star join center variable
    async fn detect_star_center(&self, graph: &JoinGraph) -> Result<Option<String>> {
        let mut variable_counts: HashMap<String, usize> = HashMap::new();

        // Count how many patterns each variable appears in
        for node in &graph.nodes {
            for var in &node.variables {
                *variable_counts.entry(var.clone()).or_insert(0) += 1;
            }
        }

        // Find variable that appears in most patterns (star center candidate)
        let max_count = variable_counts.values().max().copied().unwrap_or(0);
        if max_count >= 3 {
            // Need at least 3 connections for a star
            for (var, count) in variable_counts {
                if count == max_count {
                    return Ok(Some(var));
                }
            }
        }

        Ok(None)
    }

    /// Detect chain join pattern
    async fn detect_chain_pattern(&self, graph: &JoinGraph) -> Result<Option<Vec<String>>> {
        // Find a path that visits all nodes with minimal backtracking
        if graph.nodes.len() < 3 {
            return Ok(None);
        }

        // Try to find a linear chain
        for start_node in &graph.nodes {
            if let Some(chain) = self.find_chain_from_node(graph, &start_node.id).await? {
                if chain.len() == graph.nodes.len() {
                    return Ok(Some(chain));
                }
            }
        }

        Ok(None)
    }

    /// Find chain starting from a specific node
    async fn find_chain_from_node(
        &self,
        graph: &JoinGraph,
        start_id: &str,
    ) -> Result<Option<Vec<String>>> {
        let mut visited = HashSet::new();
        let mut chain = Vec::new();
        let mut current = start_id.to_string();

        loop {
            if visited.contains(&current) {
                break;
            }

            visited.insert(current.clone());
            chain.push(current.clone());

            // Find next node with exactly one unvisited connection
            let mut next_candidates = Vec::new();
            for edge in &graph.edges {
                if edge.from_node == current && !visited.contains(&edge.to_node) {
                    next_candidates.push(edge.to_node.clone());
                }
                if edge.to_node == current && !visited.contains(&edge.from_node) {
                    next_candidates.push(edge.from_node.clone());
                }
            }

            if next_candidates.len() == 1 {
                current = next_candidates[0].clone();
            } else {
                break;
            }
        }

        if chain.len() >= 3 {
            Ok(Some(chain))
        } else {
            Ok(None)
        }
    }

    /// Calculate graph complexity score
    async fn calculate_graph_complexity(&self, graph: &JoinGraph) -> Result<f64> {
        let node_count = graph.nodes.len() as f64;
        let edge_count = graph.edges.len() as f64;
        let all_variables: HashSet<String> = graph.nodes.iter()
            .flat_map(|node| node.variables.iter().cloned())
            .collect();
        let variable_count = all_variables.len() as f64;

        // Complexity is based on graph density and variable sharing
        let density = edge_count / (node_count * (node_count - 1.0) / 2.0);
        let complexity = node_count * density + variable_count * 0.5;

        Ok(complexity)
    }

    /// Select optimization strategy based on detected patterns
    async fn select_optimization_strategy(
        &self,
        patterns: &SpecialJoinPatterns,
    ) -> Result<JoinOptimizationStrategy> {
        if !patterns.star_joins.is_empty() && self.config.enable_star_join_detection {
            Ok(JoinOptimizationStrategy::StarJoin)
        } else if !patterns.chain_joins.is_empty() && self.config.enable_chain_optimization {
            Ok(JoinOptimizationStrategy::ChainJoin)
        } else if self.config.enable_bushy_trees && patterns.total_patterns > 5 {
            Ok(JoinOptimizationStrategy::BushyTree)
        } else {
            Ok(JoinOptimizationStrategy::Dynamic)
        }
    }

    /// Optimize star join pattern
    async fn optimize_star_join(
        &self,
        graph: &JoinGraph,
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        info!("Applying star join optimization");

        let mut operations = Vec::new();

        if let Some(center_var) = graph.nodes.first() {
            // Step 1: Execute the most selective pattern first (center)
            let center_op = JoinOperation {
                id: "star_center".to_string(),
                operation_type: JoinOperationType::InitialScan,
                left_input: None,
                right_input: Some(center_var.id.clone()),
                join_algorithm: JoinAlgorithm::HashJoin,
                join_variables: center_var.variables.clone(),
                estimated_cost: center_var.execution_cost,
                estimated_cardinality: center_var.estimated_cardinality,
                parallelizable: false,
            };
            operations.push(center_op);

            // Step 2: Join remaining patterns to the center
            for (idx, node) in graph.nodes.iter().skip(1).enumerate() {
                let join_op = JoinOperation {
                    id: format!("star_join_{}", idx),
                    operation_type: JoinOperationType::Join,
                    left_input: Some("star_center".to_string()),
                    right_input: Some(node.id.clone()),
                    join_algorithm: self
                        .select_join_algorithm(
                            node.estimated_cardinality,
                            center_var.estimated_cardinality,
                        )
                        .await?,
                    join_variables: node
                        .variables
                        .intersection(&center_var.variables)
                        .cloned()
                        .collect(),
                    estimated_cost: self
                        .estimate_join_cost(
                            center_var.estimated_cardinality,
                            node.estimated_cardinality,
                        )
                        .await?,
                    estimated_cardinality: self
                        .estimate_join_result_cardinality(
                            center_var.estimated_cardinality,
                            node.estimated_cardinality,
                        )
                        .await?,
                    parallelizable: true,
                };
                operations.push(join_op);
            }
        }

        // Calculate values before moving operations
        let estimated_total_cost: f64 = operations.iter().map(|op| op.estimated_cost).sum();
        let expected_parallelism = self.calculate_parallelization_opportunities_for_operations(&operations).await?;
        let memory_requirements = self.estimate_memory_requirements(&operations).await?;

        Ok(JoinPlan {
            operations,
            estimated_cost: estimated_total_cost,
            parallelization_opportunities: expected_parallelism,
            execution_strategy: JoinExecutionStrategy::StarJoin,
            memory_requirements,
        })
    }

    /// Optimize chain join pattern
    async fn optimize_chain_join(
        &self,
        graph: &JoinGraph,
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        info!("Applying chain join optimization");

        let mut operations = Vec::new();

        // For chain joins, use left-deep tree with most selective joins first
        let mut sorted_nodes: Vec<_> = graph.nodes.iter().collect();
        sorted_nodes.sort_by(|a, b| a.selectivity.partial_cmp(&b.selectivity).unwrap());

        if let Some(first_node) = sorted_nodes.first() {
            // Initial scan
            let initial_op = JoinOperation {
                id: "chain_start".to_string(),
                operation_type: JoinOperationType::InitialScan,
                left_input: None,
                right_input: Some(first_node.id.clone()),
                join_algorithm: JoinAlgorithm::SortMergeJoin,
                join_variables: first_node.variables.clone(),
                estimated_cost: first_node.execution_cost,
                estimated_cardinality: first_node.estimated_cardinality,
                parallelizable: false,
            };
            operations.push(initial_op);

            // Chain subsequent joins
            let mut current_cardinality = first_node.estimated_cardinality;
            for (idx, node) in sorted_nodes.iter().skip(1).enumerate() {
                let join_op = JoinOperation {
                    id: format!("chain_join_{}", idx),
                    operation_type: JoinOperationType::Join,
                    left_input: Some(if idx == 0 {
                        "chain_start".to_string()
                    } else {
                        format!("chain_join_{}", idx - 1)
                    }),
                    right_input: Some(node.id.clone()),
                    join_algorithm: JoinAlgorithm::SortMergeJoin,
                    join_variables: self.find_common_variables(
                        &operations.last().unwrap().join_variables,
                        &node.variables,
                    ),
                    estimated_cost: self
                        .estimate_join_cost(current_cardinality, node.estimated_cardinality)
                        .await?,
                    estimated_cardinality: self
                        .estimate_join_result_cardinality(
                            current_cardinality,
                            node.estimated_cardinality,
                        )
                        .await?,
                    parallelizable: false, // Chain joins are inherently sequential
                };
                current_cardinality = join_op.estimated_cardinality;
                operations.push(join_op);
            }
        }

        // Calculate values before moving operations
        let estimated_total_cost: f64 = operations.iter().map(|op| op.estimated_cost).sum();
        let memory_requirements = self.estimate_memory_requirements(&operations).await?;

        Ok(JoinPlan {
            operations,
            estimated_cost: estimated_total_cost,
            parallelization_opportunities: vec![], // Chain joins are sequential
            execution_strategy: JoinExecutionStrategy::ChainJoin,
            memory_requirements,
        })
    }


    /// Dynamic join optimization based on runtime statistics
    async fn dynamic_join_optimization(
        &self,
        graph: &JoinGraph,
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        info!("Applying dynamic join optimization");

        // Use cost-based optimization with runtime feedback
        let mut operations = Vec::new();

        // Create a flexible join plan that can adapt at runtime
        let nodes: Vec<_> = graph.nodes.iter().collect();

        for (idx, node) in nodes.iter().enumerate() {
            let join_op = JoinOperation {
                id: format!("dynamic_join_{}", idx),
                operation_type: if idx == 0 {
                    JoinOperationType::InitialScan
                } else {
                    JoinOperationType::Join
                },
                left_input: if idx > 0 {
                    Some(format!("dynamic_join_{}", idx - 1))
                } else {
                    None
                },
                right_input: Some(node.id.clone()),
                join_algorithm: JoinAlgorithm::Adaptive, // Will choose algorithm at runtime
                join_variables: node.variables.clone(),
                estimated_cost: node.execution_cost,
                estimated_cardinality: node.estimated_cardinality,
                parallelizable: true,
            };
            operations.push(join_op);
        }

        // Calculate values before moving operations
        let estimated_total_cost: f64 = operations.iter().map(|op| op.estimated_cost).sum();
        let expected_parallelism = self.calculate_parallelization_opportunities_for_operations(&operations).await?;
        let memory_requirements = self.estimate_memory_requirements(&operations).await?;

        Ok(JoinPlan {
            operations,
            estimated_cost: estimated_total_cost,
            parallelization_opportunities: expected_parallelism,
            execution_strategy: JoinExecutionStrategy::Dynamic,
            memory_requirements,
        })
    }

    /// Apply adaptive execution optimizations
    async fn apply_adaptive_optimizations(
        &self,
        mut plan: JoinPlan,
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        info!("Applying adaptive execution optimizations");

        // Add runtime adaptation points
        for operation in &mut plan.operations {
            if operation.parallelizable {
                operation.join_algorithm = JoinAlgorithm::Adaptive;
            }
        }

        // Add monitoring and reoptimization triggers
        // Note: parallelization_opportunities is a Vec, not a scalar value to multiply

        Ok(plan)
    }

    // Helper methods for cost estimation and algorithm selection

    async fn estimate_selectivity(&self, pattern: &TriplePattern) -> Result<f64> {
        // Simplified selectivity estimation
        let mut selectivity: f64 = 1.0;

        if pattern
            .subject
            .as_ref()
            .map_or(false, |s| s.starts_with('?'))
        {
            selectivity *= 0.1;
        }
        if pattern
            .predicate
            .as_ref()
            .map_or(false, |s| s.starts_with('?'))
        {
            selectivity *= 0.05;
        }
        if pattern
            .object
            .as_ref()
            .map_or(false, |s| s.starts_with('?'))
        {
            selectivity *= 0.1;
        }

        Ok(selectivity.max(0.001))
    }

    async fn estimate_cardinality(&self, pattern: &TriplePattern) -> Result<u64> {
        // Base cardinality estimation
        let base_cardinality = 100000u64;
        let selectivity = self.estimate_selectivity(pattern).await?;
        Ok((base_cardinality as f64 * selectivity) as u64)
    }

    async fn estimate_execution_cost(&self, pattern: &TriplePattern) -> Result<f64> {
        let cardinality = self.estimate_cardinality(pattern).await?;
        Ok(cardinality as f64 * 0.001) // Cost is proportional to cardinality
    }

    async fn estimate_join_selectivity(&self, node1: &JoinNode, node2: &JoinNode) -> Result<f64> {
        // Simplified join selectivity based on shared variables
        let shared_vars = node1.variables.intersection(&node2.variables).count();
        Ok(1.0 / (shared_vars.max(1) as f64))
    }

    async fn estimate_join_result_size(&self, node1: &JoinNode, node2: &JoinNode) -> Result<u64> {
        let selectivity = self.estimate_join_selectivity(node1, node2).await?;
        Ok((node1.estimated_cardinality * node2.estimated_cardinality) as u64)
    }

    async fn select_join_algorithm(
        &self,
        left_size: u64,
        right_size: u64,
    ) -> Result<JoinAlgorithm> {
        if left_size < 1000 && right_size < 1000 {
            Ok(JoinAlgorithm::NestedLoopJoin)
        } else if left_size.min(right_size) < 10000 {
            Ok(JoinAlgorithm::HashJoin)
        } else {
            Ok(JoinAlgorithm::SortMergeJoin)
        }
    }

    async fn estimate_join_cost(&self, left_size: u64, right_size: u64) -> Result<f64> {
        // Cost model: O(M + N) for hash join, O(M * N) for nested loop
        if left_size < 1000 && right_size < 1000 {
            Ok((left_size * right_size) as f64 * 0.001) // Nested loop
        } else {
            Ok((left_size + right_size) as f64 * 0.001) // Hash join
        }
    }

    async fn estimate_join_result_cardinality(
        &self,
        left_size: u64,
        right_size: u64,
    ) -> Result<u64> {
        // Simplified estimation assuming some selectivity
        Ok((left_size * right_size / 10).max(1))
    }

    async fn enumerate_join_orders(&self, graph: &JoinGraph) -> Result<Vec<JoinOrderOption>> {
        // Simplified join order enumeration
        let mut options = Vec::new();

        for edge in &graph.edges {
            // Find nodes by ID in the Vec
            let from_node = graph.nodes.iter().find(|n| n.id == edge.from_node);
            let to_node = graph.nodes.iter().find(|n| n.id == edge.to_node);
            
            if let (Some(from), Some(to)) = (from_node, to_node) {
                options.push(JoinOrderOption {
                    left_node: edge.from_node.clone(),
                    right_node: edge.to_node.clone(),
                    join_variables: edge.shared_variables.iter().cloned().collect(),
                    estimated_cost: self
                        .estimate_join_cost(
                            from.estimated_cardinality,
                            to.estimated_cardinality,
                        )
                        .await?,
                    estimated_cardinality: (edge.estimated_cost * 1000.0) as u64, // Convert cost to cardinality estimate
                });
            }
        }

        Ok(options)
    }

    async fn select_optimal_join_order(
        &self,
        orders: Vec<JoinOrderOption>,
    ) -> Result<Vec<JoinOrderOption>> {
        // Select the order with minimum total cost
        let mut optimal = orders;
        optimal.sort_by(|a, b| a.estimated_cost.partial_cmp(&b.estimated_cost).unwrap());
        Ok(optimal)
    }

    async fn select_optimal_join_algorithm(
        &self,
        join_pair: &JoinOrderOption,
    ) -> Result<JoinAlgorithm> {
        if join_pair.estimated_cardinality < 10000 {
            Ok(JoinAlgorithm::HashJoin)
        } else {
            Ok(JoinAlgorithm::SortMergeJoin)
        }
    }

    async fn calculate_parallelism(&self, operations: &[JoinOperation]) -> Result<f64> {
        let parallelizable_ops = operations.iter().filter(|op| op.parallelizable).count();
        Ok(parallelizable_ops as f64 / operations.len() as f64)
    }

    /// Calculate parallelization opportunities for operations
    async fn calculate_parallelization_opportunities_for_operations(
        &self,
        operations: &[JoinOperation],
    ) -> Result<Vec<ParallelizationOpportunity>> {
        let mut opportunities = Vec::new();
        
        // Find groups of parallelizable operations
        let parallelizable_ops: Vec<_> = operations
            .iter()
            .filter(|op| op.parallelizable)
            .collect();
        
        if !parallelizable_ops.is_empty() {
            let op_ids: Vec<String> = parallelizable_ops
                .iter()
                .map(|op| op.id.clone())
                .collect();
            
            let total_memory: u64 = parallelizable_ops
                .iter()
                .map(|op| op.estimated_cardinality * 100) // Estimate memory per row
                .sum();
            
            let parallelism_factor = parallelizable_ops.len() as f64 / operations.len() as f64;
            
            opportunities.push(ParallelizationOpportunity {
                operation_ids: op_ids,
                parallelism_factor,
                memory_requirements: total_memory,
            });
        }
        
        Ok(opportunities)
    }

    async fn estimate_memory_requirements(&self, operations: &[JoinOperation]) -> Result<u64> {
        let total_cardinality: u64 = operations.iter().map(|op| op.estimated_cardinality).sum();
        Ok(total_cardinality * 100) // Assume 100 bytes per row
    }

    fn extract_variables(&self, pattern: &TriplePattern) -> HashSet<String> {
        let mut variables = HashSet::new();

        if let Some(ref subject) = pattern.subject {
            if subject.starts_with('?') {
                variables.insert(subject.clone());
            }
        }
        if let Some(ref predicate) = pattern.predicate {
            if predicate.starts_with('?') {
                variables.insert(predicate.clone());
            }
        }
        if let Some(ref object) = pattern.object {
            if object.starts_with('?') {
                variables.insert(object.clone());
            }
        }

        variables
    }

    /// Extract variables from TriplePattern for compatibility
    fn get_pattern_variables(&self, pattern: &TriplePattern) -> HashSet<String> {
        self.extract_variables(pattern)
    }

    /// Estimate pattern cardinality compatibility method
    async fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> Result<u64> {
        self.estimate_cardinality(pattern).await
    }

    /// Estimate pattern selectivity compatibility method
    async fn estimate_pattern_selectivity(&self, pattern: &TriplePattern) -> Result<f64> {
        self.estimate_selectivity(pattern).await
    }

    fn find_common_variables(
        &self,
        vars1: &HashSet<String>,
        vars2: &HashSet<String>,
    ) -> HashSet<String> {
        vars1.intersection(vars2).cloned().collect()
    }

    /// Analyze filter pushdown opportunities for improved performance
    async fn analyze_filter_pushdown(
        &self,
        graph: &mut JoinGraph,
        filter: &PlanningFilterExpression,
    ) -> Result<()> {
        debug!("Analyzing filter pushdown for filter expression");

        // Parse the filter to identify variables and selectivity
        let filter_variables = self.extract_filter_variables_from_expression(filter);
        let filter_selectivity = self.estimate_filter_selectivity_from_expression(filter).await?;

        // Find nodes that can benefit from early filtering
        for node in graph.nodes.iter_mut() {
            // Check if any filter variables overlap with node variables
            let overlap: HashSet<_> = filter_variables
                .intersection(&node.variables)
                .cloned()
                .collect();

            if !overlap.is_empty() {
                // Update node selectivity with filter benefit
                let original_cardinality = node.estimated_cardinality;
                node.estimated_cardinality = 
                    (original_cardinality as f64 * filter_selectivity) as u64;
                
                // Reduce execution cost due to early filtering
                node.execution_cost *= filter_selectivity;
                
                info!(
                    "Applied filter pushdown to node {}: cardinality {} -> {}, cost reduction: {:.2}%",
                    node.id,
                    original_cardinality,
                    node.estimated_cardinality,
                    (1.0 - filter_selectivity) * 100.0
                );
            }
        }

        Ok(())
    }

    /// Calculate the potential benefit of star join optimization
    async fn calculate_star_join_benefit(
        &self,
        center_node: &JoinNode,
        connected_nodes: &[&JoinNode],
    ) -> Result<f64> {
        if connected_nodes.is_empty() {
            return Ok(0.0);
        }

        // Calculate cost without star optimization (linear joins)
        let mut linear_cost = center_node.execution_cost;
        let mut current_cardinality = center_node.estimated_cardinality;

        for node in connected_nodes {
            // Sequential join cost
            let join_cost = self.estimate_join_cost(current_cardinality, node.estimated_cardinality).await?;
            linear_cost += join_cost + node.execution_cost;
            
            // Update cardinality for next join
            current_cardinality = self.estimate_join_result_cardinality(
                current_cardinality,
                node.estimated_cardinality
            ).await?;
        }

        // Calculate cost with star optimization (parallel execution possible)
        let mut star_cost = center_node.execution_cost;
        
        // All leaf nodes can be executed in parallel with the center
        let max_leaf_cost = connected_nodes
            .iter()
            .map(|n| n.execution_cost)
            .fold(0.0, f64::max);
        
        star_cost += max_leaf_cost; // Parallel execution

        // Join cost for star pattern (can also be parallelized)
        let total_join_cost: f64 = connected_nodes
            .iter()
            .map(|node| {
                // Each node joins with center result
                let join_cost = futures::executor::block_on(
                    self.estimate_join_cost(center_node.estimated_cardinality, node.estimated_cardinality)
                );
                join_cost.unwrap_or(1000.0) // Fallback cost
            })
            .sum();

        star_cost += total_join_cost / connected_nodes.len() as f64; // Parallel join execution

        // Calculate benefit as cost reduction percentage
        let benefit = if linear_cost > 0.0 {
            (linear_cost - star_cost) / linear_cost
        } else {
            0.0
        };

        debug!(
            "Star join benefit calculation: linear_cost={:.2}, star_cost={:.2}, benefit={:.2}%",
            linear_cost,
            star_cost,
            benefit * 100.0
        );

        Ok(benefit.max(0.0)) // Ensure non-negative benefit
    }

    /// Extract variables from a FilterExpression
    fn extract_filter_variables_from_expression(&self, _filter: &PlanningFilterExpression) -> HashSet<String> {
        // Simplified implementation - in a real scenario this would parse the filter
        // For now, return empty set since the specific FilterExpression structure is unclear
        HashSet::new()
    }

    /// Extract variables from a pattern
    fn extract_variables_from_pattern(&self, _pattern: &str) -> HashSet<String> {
        // Simplified pattern variable extraction
        let mut variables = HashSet::new();
        // TODO: Implement proper pattern parsing
        variables
    }

    /// Extract variables from a term
    fn extract_variables_from_term(&self, term: &str) -> HashSet<String> {
        let mut variables = HashSet::new();
        if term.starts_with('?') {
            variables.insert(term.to_string());
        }
        variables
    }

    /// Estimate the selectivity of a FilterExpression
    async fn estimate_filter_selectivity_from_expression(&self, _filter: &PlanningFilterExpression) -> Result<f64> {
        // Simplified implementation - default selectivity
        // In a real scenario this would analyze the filter structure
        let selectivity: f64 = 0.5; // Default medium selectivity

        debug!("Estimated filter selectivity: {:.2}", selectivity);
        Ok(selectivity.min(1.0).max(0.001)) // Clamp between 0.001 and 1.0
    }

    /// Calculate the potential benefit of star join optimization from edges
    async fn calculate_star_join_benefit_from_edges(
        &self,
        center_node: &JoinNode,
        connections: &[&JoinEdge],
    ) -> Result<f64> {
        if connections.is_empty() {
            return Ok(0.0);
        }

        // Extract connected node IDs from edges
        let connected_node_ids: Vec<String> = connections
            .iter()
            .map(|edge| {
                if edge.from_node == center_node.id {
                    edge.to_node.clone()
                } else {
                    edge.from_node.clone()
                }
            })
            .collect();

        // For simplification, assume all connected nodes have similar characteristics
        // In a real implementation, you would look up actual node statistics
        let estimated_connected_cardinality = 1000u64; // Default estimate
        let estimated_connected_cost = 50.0; // Default estimate

        // Calculate cost without star optimization (linear joins)
        let mut linear_cost = center_node.execution_cost;
        let mut current_cardinality = center_node.estimated_cardinality;

        for _ in &connected_node_ids {
            // Sequential join cost
            let join_cost = self.estimate_join_cost(current_cardinality, estimated_connected_cardinality).await?;
            linear_cost += join_cost + estimated_connected_cost;
            
            // Update cardinality for next join
            current_cardinality = self.estimate_join_result_cardinality(
                current_cardinality,
                estimated_connected_cardinality
            ).await?;
        }

        // Calculate cost with star optimization (parallel execution possible)
        let mut star_cost = center_node.execution_cost;
        
        // All leaf nodes can be executed in parallel with the center
        star_cost += estimated_connected_cost; // Parallel execution

        // Join cost for star pattern (can also be parallelized)
        let total_join_cost: f64 = connected_node_ids
            .iter()
            .map(|_| {
                // Each node joins with center result
                let join_cost = futures::executor::block_on(
                    self.estimate_join_cost(center_node.estimated_cardinality, estimated_connected_cardinality)
                );
                join_cost.unwrap_or(1000.0) // Fallback cost
            })
            .sum();

        star_cost += total_join_cost / connected_node_ids.len() as f64; // Parallel join execution

        // Calculate benefit as cost reduction percentage
        let benefit = if linear_cost > 0.0 {
            (linear_cost - star_cost) / linear_cost
        } else {
            0.0
        };

        debug!(
            "Star join benefit calculation: linear_cost={:.2}, star_cost={:.2}, benefit={:.2}%",
            linear_cost,
            star_cost,
            benefit * 100.0
        );

        Ok(benefit.max(0.0)) // Ensure non-negative benefit
    }

    /// Extract variables from a filter expression (legacy string-based method)
    fn extract_filter_variables(&self, filter: &str) -> HashSet<String> {
        let mut variables = HashSet::new();
        
        // Simple regex-based variable extraction (variables start with ?)
        for part in filter.split_whitespace() {
            if part.starts_with('?') {
                variables.insert(part.trim_matches(|c: char| !c.is_alphanumeric() && c != '_').to_string());
            }
        }
        
        variables
    }

    /// Estimate the selectivity of a filter (legacy string-based method)
    async fn estimate_filter_selectivity(&self, filter: &str) -> Result<f64> {
        // Simplified selectivity estimation based on filter patterns
        let selectivity = if filter.contains("=") {
            // Equality filters are typically very selective
            0.1
        } else if filter.contains("REGEX") || filter.contains("CONTAINS") {
            // String pattern filters have medium selectivity
            0.3
        } else if filter.contains(">") || filter.contains("<") {
            // Range filters have variable selectivity
            0.5
        } else {
            // Default selectivity for unknown filter types
            0.8
        };

        debug!("Estimated filter selectivity for '{}': {:.2}", filter, selectivity);
        Ok(selectivity)
    }

    /// Find the longest chain starting from a specific node
    async fn find_longest_chain(
        &self,
        graph: &JoinGraph,
        start_id: &str,
        visited: &mut HashSet<String>,
    ) -> Result<Vec<String>> {
        let mut chain = Vec::new();
        let mut current_id = start_id.to_string();
        
        while !visited.contains(&current_id) {
            visited.insert(current_id.clone());
            chain.push(current_id.clone());
            
            // Find next node in chain
            let mut next_candidates = Vec::new();
            for edge in &graph.edges {
                if edge.from_node == current_id && !visited.contains(&edge.to_node) {
                    next_candidates.push(edge.to_node.clone());
                } else if edge.to_node == current_id && !visited.contains(&edge.from_node) {
                    next_candidates.push(edge.from_node.clone());
                }
            }
            
            if next_candidates.len() == 1 {
                current_id = next_candidates[0].clone();
            } else {
                break;
            }
        }
        
        Ok(chain)
    }

    /// Calculate the benefit of chain join optimization
    async fn calculate_chain_join_benefit(&self, chain: &[String]) -> Result<f64> {
        if chain.len() < 3 {
            return Ok(0.0);
        }
        let base_cost = chain.len() as f64 * 1000.0;
        let optimized_cost = chain.len() as f64 * 500.0;
        let benefit = (base_cost - optimized_cost) / base_cost;
        Ok(benefit.max(0.0))
    }

    /// Convert bushy tree to execution operations
    async fn tree_to_execution_operations(&self, tree: &BushyTreeNode) -> Result<Vec<JoinOperation>> {
        let mut operations = Vec::new();
        let mut counter = 0;
        self.tree_to_operations_recursive(tree, &mut operations, &mut counter).await?;
        Ok(operations)
    }
    
    fn tree_to_operations_recursive<'a>(
        &'a self,
        node: &'a BushyTreeNode,
        operations: &'a mut Vec<JoinOperation>,
        counter: &'a mut usize,
    ) -> futures::future::BoxFuture<'a, Result<String>> {
        Box::pin(async move {
        match &node.node_type {
            BushyNodeType::Leaf => {
                let op_id = format!("scan_{}", counter);
                *counter += 1;
                // Use first pattern as identifier
                let pattern_id = if !node.patterns.is_empty() {
                    format!("pattern_{}", counter)
                } else {
                    format!("unknown_pattern_{}", counter)
                };
                let operation = JoinOperation {
                    id: op_id.clone(),
                    operation_type: JoinOperationType::InitialScan,
                    left_input: None,
                    right_input: Some(pattern_id),
                    join_algorithm: JoinAlgorithm::HashJoin,
                    join_variables: HashSet::new(),
                    estimated_cost: node.total_cost,
                    estimated_cardinality: 1000, // Default cardinality
                    parallelizable: false,
                };
                operations.push(operation);
                Ok(op_id)
            }
            BushyNodeType::InnerJoin => {
                if let (Some(left), Some(right)) = (&node.left_child, &node.right_child) {
                    let left_id = self.tree_to_operations_recursive(left, operations, counter).await?;
                    let right_id = self.tree_to_operations_recursive(right, operations, counter).await?;
                    let op_id = format!("join_{}", counter);
                    *counter += 1;
                    let operation = JoinOperation {
                        id: op_id.clone(),
                        operation_type: JoinOperationType::Join,
                        left_input: Some(left_id),
                        right_input: Some(right_id),
                        join_algorithm: JoinAlgorithm::HashJoin,
                        join_variables: HashSet::new(), // Would need to extract from patterns
                        estimated_cost: node.total_cost,
                        estimated_cardinality: 1000, // Default cardinality
                        parallelizable: true,
                    };
                    operations.push(operation);
                    Ok(op_id)
                } else {
                    Err(anyhow!("InnerJoin node missing children"))
                }
            }
            BushyNodeType::LeftJoin => {
                if let (Some(left), Some(right)) = (&node.left_child, &node.right_child) {
                    let left_id = self.tree_to_operations_recursive(left, operations, counter).await?;
                    let right_id = self.tree_to_operations_recursive(right, operations, counter).await?;
                    let op_id = format!("left_join_{}", counter);
                    *counter += 1;
                    let operation = JoinOperation {
                        id: op_id.clone(),
                        operation_type: JoinOperationType::Join,
                        left_input: Some(left_id),
                        right_input: Some(right_id),
                        join_algorithm: JoinAlgorithm::HashJoin,
                        join_variables: HashSet::new(),
                        estimated_cost: node.total_cost,
                        estimated_cardinality: 1000,
                        parallelizable: true,
                    };
                    operations.push(operation);
                    Ok(op_id)
                } else {
                    Err(anyhow!("LeftJoin node missing children"))
                }
            }
            BushyNodeType::Union => {
                if let (Some(left), Some(right)) = (&node.left_child, &node.right_child) {
                    let left_id = self.tree_to_operations_recursive(left, operations, counter).await?;
                    let right_id = self.tree_to_operations_recursive(right, operations, counter).await?;
                    let op_id = format!("union_{}", counter);
                    *counter += 1;
                    let operation = JoinOperation {
                        id: op_id.clone(),
                        operation_type: JoinOperationType::Union,
                        left_input: Some(left_id),
                        right_input: Some(right_id),
                        join_algorithm: JoinAlgorithm::HashJoin,
                        join_variables: HashSet::new(),
                        estimated_cost: node.total_cost,
                        estimated_cardinality: 1000,
                        parallelizable: true,
                    };
                    operations.push(operation);
                    Ok(op_id)
                } else {
                    Err(anyhow!("Union node missing children"))
                }
            }
        }
        })
    }

    /// Identify parallelization opportunities in bushy tree
    async fn identify_parallelization_opportunities(&self, tree: &BushyTreeNode) -> Result<Vec<ParallelizationOpportunity>> {
        let mut opportunities = Vec::new();
        self.collect_parallelization_opportunities(tree, &mut opportunities).await?;
        Ok(opportunities)
    }
    
    fn collect_parallelization_opportunities<'a>(
        &'a self,
        node: &'a BushyTreeNode,
        opportunities: &'a mut Vec<ParallelizationOpportunity>,
    ) -> futures::future::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
        match &node.node_type {
            BushyNodeType::Leaf => {
                // Leaf nodes can be executed in parallel
                opportunities.push(ParallelizationOpportunity {
                    operation_ids: vec![format!("leaf_{}", node.patterns.len())],
                    parallelism_factor: node.parallelization_factor,
                    memory_requirements: (node.total_cost * 1000.0) as u64,
                });
            }
            BushyNodeType::InnerJoin => {
                // Check left and right children
                if let Some(left) = &node.left_child {
                    self.collect_parallelization_opportunities(left, opportunities).await?;
                }
                if let Some(right) = &node.right_child {
                    self.collect_parallelization_opportunities(right, opportunities).await?;
                }
                
                // Add parallelization opportunity for large joins
                if node.total_cost > 10000.0 {
                    opportunities.push(ParallelizationOpportunity {
                        operation_ids: vec![format!("parallel_join_{}", opportunities.len())],
                        parallelism_factor: node.parallelization_factor,
                        memory_requirements: (node.total_cost * 500.0) as u64,
                    });
                }
            }
            BushyNodeType::LeftJoin => {
                // Handle left join parallelization
                if let Some(left) = &node.left_child {
                    self.collect_parallelization_opportunities(left, opportunities).await?;
                }
                if let Some(right) = &node.right_child {
                    self.collect_parallelization_opportunities(right, opportunities).await?;
                }
            }
            BushyNodeType::Union => {
                // Union operations are naturally parallelizable
                opportunities.push(ParallelizationOpportunity {
                    operation_ids: vec![format!("union_{}", opportunities.len())],
                    parallelism_factor: 2.0, // Can execute both sides in parallel
                    memory_requirements: (node.total_cost * 200.0) as u64,
                });
                
                if let Some(left) = &node.left_child {
                    self.collect_parallelization_opportunities(left, opportunities).await?;
                }
                if let Some(right) = &node.right_child {
                    self.collect_parallelization_opportunities(right, opportunities).await?;
                }
            }
        }
        Ok(())
        })
    }

    /// Calculate memory requirements for the tree
    async fn calculate_memory_requirements(&self, tree: &BushyTreeNode) -> Result<u64> {
        // Use total_cost as proxy for memory requirements since estimated_cardinality doesn't exist
        Ok((tree.total_cost * 100.0) as u64)
    }

    /// Estimate pattern cost
    async fn estimate_pattern_cost(&self, pattern: &TriplePattern) -> Result<f64> {
        let cardinality = self.estimate_cardinality(pattern).await?;
        Ok(cardinality as f64 * 0.001)
    }

    /// Estimate bushy join cost
    async fn estimate_bushy_join_cost(&self, left: &BushyTreeNode, right: &BushyTreeNode, _registry: &ServiceRegistry) -> Result<f64> {
        // Use total_cost since estimated_cardinality doesn't exist
        Ok((left.total_cost + right.total_cost) * 0.001)
    }

    /// Calculate parallelization factor
    async fn calculate_parallelization_factor(&self, left: &BushyTreeNode, right: &BushyTreeNode) -> Result<f64> {
        Ok((left.parallelization_factor + right.parallelization_factor) / 2.0)
    }

    /// Calculate cardinality match score
    async fn calculate_cardinality_match_score(&self, target: u64, samples: &[u64]) -> Result<f64> {
        if samples.is_empty() {
            return Ok(0.0);
        }
        let avg: f64 = samples.iter().map(|&x| x as f64).sum::<f64>() / samples.len() as f64;
        let diff = (target as f64 - avg).abs();
        let score = 1.0 / (1.0 + diff / avg);
        Ok(score)
    }

    /// Heuristic strategy selection
    async fn heuristic_strategy_selection(&self, patterns: &[TriplePattern], cardinality: u64) -> Result<JoinExecutionStrategy> {
        if patterns.len() <= 2 {
            Ok(JoinExecutionStrategy::Sequential)
        } else if cardinality > 100000 {
            Ok(JoinExecutionStrategy::Parallel)
        } else {
            Ok(JoinExecutionStrategy::Dynamic)
        }
    }
}

/// Adaptive execution controller for runtime optimization
#[derive(Debug)]
pub struct AdaptiveExecutionController {
    runtime_statistics: HashMap<String, RuntimeStatistics>,
    reoptimization_threshold: f64,
}

impl AdaptiveExecutionController {
    pub fn new() -> Self {
        Self {
            runtime_statistics: HashMap::new(),
            reoptimization_threshold: 2.0,
        }
    }

    /// Monitor execution and trigger reoptimization if needed
    pub async fn monitor_execution(
        &mut self,
        operation_id: &str,
        actual_cardinality: u64,
        actual_cost: f64,
        estimated_cardinality: u64,
        estimated_cost: f64,
    ) -> Result<bool> {
        let stats = self
            .runtime_statistics
            .entry(operation_id.to_string())
            .or_insert_with(RuntimeStatistics::new);

        stats.update(
            actual_cardinality,
            actual_cost,
            estimated_cardinality,
            estimated_cost,
        );

        // Check if reoptimization is needed
        let cost_ratio = actual_cost / estimated_cost;
        let cardinality_ratio = actual_cardinality as f64 / estimated_cardinality as f64;

        let should_reoptimize = cost_ratio > self.reoptimization_threshold
            || cardinality_ratio > self.reoptimization_threshold
            || cardinality_ratio < (1.0 / self.reoptimization_threshold);

        if should_reoptimize {
            warn!("Reoptimization triggered for operation {}: cost_ratio={:.2}, cardinality_ratio={:.2}", 
                  operation_id, cost_ratio, cardinality_ratio);
        }

        Ok(should_reoptimize)
    }

    /// Get runtime statistics for an operation
    pub fn get_statistics(&self, operation_id: &str) -> Option<&RuntimeStatistics> {
        self.runtime_statistics.get(operation_id)
    }

    /// Analyze statistics and adapt configuration
    pub async fn analyze_and_adapt(&mut self, _stats: &JoinStatistics) -> Result<()> {
        // Placeholder for adaptive logic
        debug!("Analyzing and adapting execution strategies");
        Ok(())
    }
}

// Supporting data structures

#[derive(Debug, Clone)]
pub struct JoinGraph {
    pub nodes: Vec<JoinNode>,
    pub edges: Vec<JoinEdge>,
}

// Pattern types imported from service_optimizer::types

#[derive(Debug, Clone)]
pub struct JoinNode {
    pub id: String,
    pub pattern: TriplePattern,
    pub variables: HashSet<String>,
    pub selectivity: f64,
    pub estimated_cardinality: u64,
    pub execution_cost: f64,
}

// JoinEdge and SpecialJoinPatterns imported from service_optimizer

#[derive(Debug, Clone, PartialEq)]
pub enum JoinOptimizationStrategy {
    StarJoin,
    ChainJoin,
    BushyTree,
    Dynamic,
}

// JoinPlan imported from service_optimizer

// JoinOperation, JoinOperationType, and JoinAlgorithm imported from service_optimizer::types

#[derive(Debug, Clone)]
pub struct JoinOrderOption {
    pub left_node: String,
    pub right_node: String,
    pub join_variables: HashSet<String>,
    pub estimated_cost: f64,
    pub estimated_cardinality: u64,
}

#[derive(Debug)]
pub struct JoinCostModel {
    // Cost model parameters would go here
}

impl JoinCostModel {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, Default)]
pub struct JoinStatistics {
    pub total_joins_executed: u64,
    pub total_executions: u64,
    pub average_join_cost: f64,
    pub join_algorithm_performance: HashMap<JoinAlgorithm, f64>,
    pub strategy_performance: HashMap<JoinExecutionStrategy, StrategyPerformance>,
}

#[derive(Debug)]
pub struct RuntimeStatistics {
    pub executions: u64,
    pub total_actual_cost: f64,
    pub total_estimated_cost: f64,
    pub total_actual_cardinality: u64,
    pub total_estimated_cardinality: u64,
    pub average_cost_error: f64,
    pub average_cardinality_error: f64,
}

impl RuntimeStatistics {
    pub fn new() -> Self {
        Self {
            executions: 0,
            total_actual_cost: 0.0,
            total_estimated_cost: 0.0,
            total_actual_cardinality: 0,
            total_estimated_cardinality: 0,
            average_cost_error: 0.0,
            average_cardinality_error: 0.0,
        }
    }

    pub fn update(
        &mut self,
        actual_cardinality: u64,
        actual_cost: f64,
        estimated_cardinality: u64,
        estimated_cost: f64,
    ) {
        self.executions += 1;
        self.total_actual_cost += actual_cost;
        self.total_estimated_cost += estimated_cost;
        self.total_actual_cardinality += actual_cardinality;
        self.total_estimated_cardinality += estimated_cardinality;

        self.average_cost_error =
            (self.total_actual_cost - self.total_estimated_cost).abs() / self.total_estimated_cost;
        self.average_cardinality_error =
            (self.total_actual_cardinality as f64 - self.total_estimated_cardinality as f64).abs()
                / self.total_estimated_cardinality as f64;
    }
}
