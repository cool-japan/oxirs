//! Join Optimizer - Modular Implementation
//!
//! This module provides a modular join optimizer implementation for federated queries.
//! The implementation is broken down into focused modules for better maintainability
//! and adherence to the 2000-line file policy.

pub mod config;
pub mod types;

// Re-export public types
pub use config::*;
pub use types::*;

use anyhow::Result;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::info;

use crate::{
    planner::planning::{FilterExpression as PlanningFilterExpression, TriplePattern},
    service_optimizer::{JoinAlgorithm, JoinOperation, JoinOperationType, JoinPlan},
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
                id: format!("pattern_{idx}"),
                pattern: pattern.clone(),
                variables: self.extract_variables(pattern),
                selectivity: self.estimate_selectivity(pattern).await?,
                estimated_cardinality: self.estimate_cardinality(pattern).await?,
                execution_cost: self.estimate_execution_cost(pattern).await?,
            };

            graph.nodes.push(node);
        }

        // Create edges for variable connections
        for node1 in &graph.nodes {
            for node2 in &graph.nodes {
                if node1.id != node2.id {
                    let shared_vars: Vec<String> = node1
                        .variables
                        .intersection(&node2.variables)
                        .cloned()
                        .collect();

                    if !shared_vars.is_empty() {
                        let selectivity = self.estimate_join_selectivity(node1, node2).await?;
                        let cost = self.estimate_join_cost(node1, node2).await?;
                        let edge = crate::service_optimizer::JoinEdge {
                            from_node: node1.id.clone(),
                            to_node: node2.id.clone(),
                            shared_variables: shared_vars.clone(),
                            join_selectivity: selectivity,
                            estimated_cost: cost,
                            // Legacy field aliases for compatibility
                            from: node1.id.clone(),
                            to: node2.id.clone(),
                            join_variables: shared_vars,
                            selectivity,
                        };
                        graph.edges.push(edge);
                    }
                }
            }
        }

        Ok(graph)
    }

    /// Extract variables from a triple pattern
    fn extract_variables(&self, pattern: &TriplePattern) -> HashSet<String> {
        let mut variables = HashSet::new();

        // Extract variables from subject, predicate, object
        if let Some(subject) = &pattern.subject {
            if subject.starts_with('?') {
                variables.insert(subject.clone());
            }
        }
        if let Some(predicate) = &pattern.predicate {
            if predicate.starts_with('?') {
                variables.insert(predicate.clone());
            }
        }
        if let Some(object) = &pattern.object {
            if object.starts_with('?') {
                variables.insert(object.clone());
            }
        }

        variables
    }

    /// Estimate selectivity of a triple pattern
    async fn estimate_selectivity(&self, pattern: &TriplePattern) -> Result<f64> {
        // Simplified selectivity estimation
        let mut selectivity = 1.0;

        // Less selective if more variables
        let var_count = self.extract_variables(pattern).len();
        selectivity = match var_count {
            0 => 0.001, // Ground triple
            1 => 0.01,  // One variable
            2 => 0.1,   // Two variables
            3 => 0.5,   // Three variables
            _ => 0.8,   // More variables
        };

        Ok(selectivity)
    }

    /// Estimate cardinality of a triple pattern
    async fn estimate_cardinality(&self, pattern: &TriplePattern) -> Result<u64> {
        // Simplified cardinality estimation based on pattern type
        let selectivity = self.estimate_selectivity(pattern).await?;
        let estimated_total_triples = 1_000_000u64; // Base estimate

        Ok((estimated_total_triples as f64 * selectivity) as u64)
    }

    /// Estimate execution cost of a triple pattern
    async fn estimate_execution_cost(&self, pattern: &TriplePattern) -> Result<f64> {
        let cardinality = self.estimate_cardinality(pattern).await?;
        Ok(cardinality as f64 * 0.001) // Simple cost model
    }

    /// Estimate cost of joining two nodes
    async fn estimate_join_cost(&self, node1: &JoinNode, node2: &JoinNode) -> Result<f64> {
        let join_cost = self.cost_model.estimate_join_cost(
            node1.estimated_cardinality,
            node2.estimated_cardinality,
            &JoinAlgorithm::HashJoin,
            Duration::from_millis(10),
        );

        Ok(join_cost)
    }

    /// Estimate selectivity of joining two nodes
    async fn estimate_join_selectivity(&self, node1: &JoinNode, node2: &JoinNode) -> Result<f64> {
        // Simplified join selectivity estimation
        let shared_vars: Vec<String> = node1
            .variables
            .intersection(&node2.variables)
            .cloned()
            .collect();

        // More shared variables typically means higher selectivity
        let selectivity = match shared_vars.len() {
            0 => 1.0,   // Cartesian product
            1 => 0.1,   // One join variable
            2 => 0.01,  // Two join variables
            _ => 0.001, // Multiple join variables
        };

        Ok(selectivity)
    }

    /// Detect special join patterns in the graph
    async fn detect_special_patterns(&self, graph: &JoinGraph) -> Result<Vec<JoinPatternAnalysis>> {
        let mut patterns = Vec::new();

        // Detect star patterns
        for node in &graph.nodes {
            let connected_count = graph
                .edges
                .iter()
                .filter(|edge| edge.from_node == node.id || edge.to_node == node.id)
                .count();

            if connected_count >= 3 {
                patterns.push(JoinPatternAnalysis {
                    pattern_type: JoinPatternType::Star {
                        center_variable: node.id.clone(),
                    },
                    complexity_score: connected_count as f64,
                    optimization_opportunities: vec![OptimizationOpportunity {
                        opportunity_type: OptimizationType::JoinReordering,
                        description: "Star join optimization".to_string(),
                        estimated_improvement: 0.3,
                        implementation_complexity: ComplexityLevel::Medium,
                    }],
                    estimated_benefit: 0.3,
                });
            }
        }

        Ok(patterns)
    }

    /// Select optimization strategy based on detected patterns
    async fn select_optimization_strategy(
        &self,
        patterns: &[JoinPatternAnalysis],
    ) -> Result<JoinOptimizationStrategy> {
        // Simple strategy selection
        for pattern in patterns {
            match pattern.pattern_type {
                JoinPatternType::Star { .. } if self.config.enable_star_join_detection => {
                    return Ok(JoinOptimizationStrategy::StarJoin);
                }
                JoinPatternType::Chain { .. } if self.config.enable_chain_optimization => {
                    return Ok(JoinOptimizationStrategy::ChainJoin);
                }
                _ => {}
            }
        }

        if self.config.enable_bushy_trees {
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

        // Create join operations for star pattern
        for edge in &graph.edges {
            operations.push(JoinOperation {
                id: format!("join_{}", operations.len()),
                operation_type: JoinOperationType::HashJoin,
                left_input: Some(edge.from.clone()),
                right_input: Some(edge.to.clone()),
                join_algorithm: JoinAlgorithm::HashJoin,
                join_variables: edge.join_variables.iter().cloned().collect(),
                estimated_cost: edge.estimated_cost,
                estimated_cardinality: (edge.selectivity * 1000.0) as u64,
                parallelizable: true,
                join_condition: format!("ON {}", edge.join_variables.join(" = ")),
            });
        }

        Ok(JoinPlan {
            operations,
            estimated_cost: graph.edges.iter().map(|e| e.estimated_cost).sum(),
            parallelization_opportunities: Vec::new(),
            execution_strategy: crate::service_optimizer::JoinExecutionStrategy::Sequential,
            memory_requirements: 1024 * 1024, // 1MB default
            estimated_total_cost: graph.edges.iter().map(|e| e.estimated_cost).sum(),
        })
    }

    /// Optimize chain join pattern  
    async fn optimize_chain_join(
        &self,
        graph: &JoinGraph,
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        info!("Applying chain join optimization");
        self.optimize_star_join(graph, registry).await // Simplified implementation
    }

    /// Optimize bushy tree pattern
    async fn optimize_bushy_tree(
        &self,
        graph: &JoinGraph,
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        info!("Applying bushy tree optimization");
        self.optimize_star_join(graph, registry).await // Simplified implementation
    }

    /// Dynamic join optimization
    async fn dynamic_join_optimization(
        &self,
        graph: &JoinGraph,
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        info!("Applying dynamic join optimization");
        self.optimize_star_join(graph, registry).await // Simplified implementation
    }

    /// Apply adaptive optimizations to the plan
    async fn apply_adaptive_optimizations(
        &self,
        plan: JoinPlan,
        registry: &ServiceRegistry,
    ) -> Result<JoinPlan> {
        info!("Applying adaptive optimizations");

        // Record statistics
        let mut stats = self.statistics.write().await;
        stats.total_joins += plan.operations.len() as u64;

        Ok(plan)
    }

    /// Get optimizer statistics
    pub async fn get_statistics(&self) -> JoinStatistics {
        self.statistics.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_optimizer_creation() {
        let config = JoinOptimizerConfig::default();
        let optimizer = DistributedJoinOptimizer::new(config);
        assert!(optimizer.config.enable_star_join_detection);
    }

    #[test]
    fn test_cost_model() {
        let cost_model = JoinCostModel::new();
        let cost = cost_model.estimate_join_cost(
            1000,
            2000,
            &JoinAlgorithm::HashJoin,
            Duration::from_millis(10),
        );
        assert!(cost > 0.0);
    }

    #[test]
    fn test_adaptive_controller() {
        let controller = AdaptiveExecutionController::new();
        assert!(controller.should_adapt(0.5, 1.0)); // 50% performance degradation
        assert!(!controller.should_adapt(1.0, 1.0)); // No degradation
    }
}
