//! Constraint ordering optimization for SHACL validation
//!
//! This module provides advanced constraint ordering optimization to improve
//! validation performance through intelligent constraint execution planning.

use crate::ShaclAiError;
use chrono::Utc;
use scirs2_core::random::Random;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use super::config::OptimizationStrategy;
use super::types::{ConstraintDependencyGraph, ConstraintPerformanceStats};

/// Constraint execution order optimizer
#[derive(Debug, Clone)]
pub struct ConstraintOrderOptimizer {
    performance_history: Arc<RwLock<HashMap<String, ConstraintPerformanceStats>>>,
    dependency_graph: Arc<RwLock<ConstraintDependencyGraph>>,
    optimization_strategy: OptimizationStrategy,
}

impl ConstraintOrderOptimizer {
    pub fn new(strategy: OptimizationStrategy) -> Self {
        Self {
            performance_history: Arc::new(RwLock::new(HashMap::new())),
            dependency_graph: Arc::new(RwLock::new(ConstraintDependencyGraph {
                dependencies: HashMap::new(),
                execution_costs: HashMap::new(),
                selectivity_scores: HashMap::new(),
            })),
            optimization_strategy: strategy,
        }
    }

    /// Optimize constraint execution order
    pub fn optimize_constraint_order(
        &self,
        constraints: &[String],
    ) -> Result<Vec<String>, ShaclAiError> {
        match self.optimization_strategy {
            OptimizationStrategy::Selectivity => self.optimize_by_selectivity(constraints),
            OptimizationStrategy::Cost => self.optimize_by_cost(constraints),
            OptimizationStrategy::Dependency => self.optimize_by_dependency(constraints),
            OptimizationStrategy::MachineLearning => self.optimize_by_ml(constraints),
            OptimizationStrategy::Hybrid => self.optimize_hybrid(constraints),
            OptimizationStrategy::Genetic => self.optimize_genetic(constraints),
        }
    }

    /// Optimize by selectivity (most selective constraints first)
    fn optimize_by_selectivity(&self, constraints: &[String]) -> Result<Vec<String>, ShaclAiError> {
        let graph = self.dependency_graph.read().map_err(|e| {
            ShaclAiError::Optimization(format!("Failed to read dependency graph: {e}"))
        })?;

        let mut constraint_selectivity: Vec<(String, f64)> = constraints
            .iter()
            .map(|c| {
                let selectivity = graph.selectivity_scores.get(c).copied().unwrap_or(0.5);
                (c.clone(), selectivity)
            })
            .collect();

        // Sort by selectivity in descending order (most selective first)
        constraint_selectivity
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(constraint_selectivity.into_iter().map(|(c, _)| c).collect())
    }

    /// Optimize by execution cost (fastest constraints first)
    fn optimize_by_cost(&self, constraints: &[String]) -> Result<Vec<String>, ShaclAiError> {
        let performance_history = self.performance_history.read().map_err(|e| {
            ShaclAiError::Optimization(format!("Failed to read performance history: {e}"))
        })?;

        let mut constraint_costs: Vec<(String, f64)> = constraints
            .iter()
            .map(|c| {
                let cost = performance_history
                    .get(c)
                    .map(|stats| stats.average_execution_time_ms)
                    .unwrap_or(100.0); // Default cost
                (c.clone(), cost)
            })
            .collect();

        // Sort by cost in ascending order (fastest first)
        constraint_costs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(constraint_costs.into_iter().map(|(c, _)| c).collect())
    }

    /// Optimize by dependency constraints (topological sort)
    fn optimize_by_dependency(&self, constraints: &[String]) -> Result<Vec<String>, ShaclAiError> {
        let graph = self.dependency_graph.read().map_err(|e| {
            ShaclAiError::Optimization(format!("Failed to read dependency graph: {e}"))
        })?;

        // Topological sort implementation
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

        for constraint in constraints {
            if !visited.contains(constraint) {
                self.topological_sort_visit(
                    constraint,
                    &graph.dependencies,
                    &mut visited,
                    &mut temp_visited,
                    &mut result,
                )?;
            }
        }

        Ok(result)
    }

    /// Recursive helper for topological sort
    #[allow(clippy::only_used_in_recursion)]
    fn topological_sort_visit(
        &self,
        constraint: &str,
        dependencies: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        temp_visited: &mut HashSet<String>,
        result: &mut Vec<String>,
    ) -> Result<(), ShaclAiError> {
        if temp_visited.contains(constraint) {
            return Err(ShaclAiError::Optimization(
                "Circular dependency detected in constraints".to_string(),
            ));
        }

        if visited.contains(constraint) {
            return Ok(());
        }

        temp_visited.insert(constraint.to_string());

        if let Some(deps) = dependencies.get(constraint) {
            for dep in deps {
                self.topological_sort_visit(dep, dependencies, visited, temp_visited, result)?;
            }
        }

        temp_visited.remove(constraint);
        visited.insert(constraint.to_string());
        result.push(constraint.to_string());

        Ok(())
    }

    /// Optimize using machine learning predictions
    fn optimize_by_ml(&self, constraints: &[String]) -> Result<Vec<String>, ShaclAiError> {
        // Simplified ML-based optimization
        // In a real implementation, this would use trained models
        let performance_history = self.performance_history.read().map_err(|e| {
            ShaclAiError::Optimization(format!("Failed to read performance history: {e}"))
        })?;

        let mut scored_constraints: Vec<(String, f64)> = constraints
            .iter()
            .map(|c| {
                let stats = performance_history.get(c);
                let score = match stats {
                    Some(s) => {
                        // Combine multiple factors for ML score
                        let time_factor = 1.0 / (s.average_execution_time_ms + 1.0);
                        let selectivity_factor = s.selectivity;
                        let success_factor = s.success_rate;

                        time_factor * 0.4 + selectivity_factor * 0.4 + success_factor * 0.2
                    }
                    None => 0.5, // Default score
                };
                (c.clone(), score)
            })
            .collect();

        // Sort by ML score in descending order
        scored_constraints
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_constraints.into_iter().map(|(c, _)| c).collect())
    }

    /// Genetic algorithm optimization
    fn optimize_genetic(&self, constraints: &[String]) -> Result<Vec<String>, ShaclAiError> {
        // Simplified genetic algorithm implementation
        // This would be a full genetic algorithm in a real implementation
        let mut best_order = constraints.to_vec();
        let mut best_score = self.evaluate_constraint_order(&best_order)?;

        // Simple hill climbing as a placeholder for genetic algorithm
        let mut rng = Random::default();
        for _ in 0..100 {
            let mut candidate = best_order.clone();
            if candidate.len() >= 2 {
                let i = rng.random_range(0, candidate.len());
                let j = rng.random_range(0, candidate.len());
                candidate.swap(i, j);

                let score = self.evaluate_constraint_order(&candidate)?;
                if score > best_score {
                    best_order = candidate;
                    best_score = score;
                }
            }
        }

        Ok(best_order)
    }

    /// Hybrid optimization combining multiple strategies
    fn optimize_hybrid(&self, constraints: &[String]) -> Result<Vec<String>, ShaclAiError> {
        // Get results from different strategies
        let selectivity_order = self.optimize_by_selectivity(constraints)?;
        let cost_order = self.optimize_by_cost(constraints)?;
        let ml_order = self.optimize_by_ml(constraints)?;

        // Combine strategies using weighted ranking
        let mut constraint_scores: HashMap<String, f64> = HashMap::new();

        // Weight: selectivity 40%, cost 35%, ML 25%
        for (index, constraint) in selectivity_order.iter().enumerate() {
            *constraint_scores.entry(constraint.clone()).or_insert(0.0) +=
                0.4 * (constraints.len() - index) as f64;
        }

        for (index, constraint) in cost_order.iter().enumerate() {
            *constraint_scores.entry(constraint.clone()).or_insert(0.0) +=
                0.35 * (constraints.len() - index) as f64;
        }

        for (index, constraint) in ml_order.iter().enumerate() {
            *constraint_scores.entry(constraint.clone()).or_insert(0.0) +=
                0.25 * (constraints.len() - index) as f64;
        }

        // Sort by combined score
        let mut scored_constraints: Vec<(String, f64)> = constraint_scores.into_iter().collect();
        scored_constraints
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_constraints.into_iter().map(|(c, _)| c).collect())
    }

    /// Evaluate the quality of a constraint order
    fn evaluate_constraint_order(&self, order: &[String]) -> Result<f64, ShaclAiError> {
        let performance_history = self.performance_history.read().map_err(|e| {
            ShaclAiError::Optimization(format!("Failed to read performance history: {e}"))
        })?;

        let mut score = 0.0;
        let mut cumulative_selectivity = 1.0;

        for constraint in order {
            if let Some(stats) = performance_history.get(constraint) {
                // Score based on execution time and selectivity position
                let time_score = 1000.0 / (stats.average_execution_time_ms + 1.0);
                let selectivity_score = stats.selectivity * cumulative_selectivity;

                score += time_score + selectivity_score * 100.0;
                cumulative_selectivity *= 1.0 - stats.selectivity;
            }
        }

        Ok(score)
    }

    /// Update performance statistics for a constraint
    pub fn update_performance_stats(
        &self,
        constraint_id: &str,
        execution_time_ms: f64,
        success: bool,
        memory_usage_mb: f64,
        selectivity: f64,
    ) -> Result<(), ShaclAiError> {
        let mut history = self.performance_history.write().map_err(|e| {
            ShaclAiError::Optimization(format!("Failed to write performance history: {e}"))
        })?;

        let stats = history.entry(constraint_id.to_string()).or_insert_with(|| {
            ConstraintPerformanceStats {
                constraint_id: constraint_id.to_string(),
                average_execution_time_ms: execution_time_ms,
                success_rate: if success { 1.0 } else { 0.0 },
                memory_usage_mb,
                cpu_usage_percent: 0.0,
                selectivity,
                execution_count: 1,
                last_updated: Utc::now(),
            }
        });

        // Update moving averages
        let alpha = 0.1; // Exponential moving average factor
        stats.average_execution_time_ms =
            alpha * execution_time_ms + (1.0 - alpha) * stats.average_execution_time_ms;
        stats.success_rate =
            alpha * (if success { 1.0 } else { 0.0 }) + (1.0 - alpha) * stats.success_rate;
        stats.memory_usage_mb = alpha * memory_usage_mb + (1.0 - alpha) * stats.memory_usage_mb;
        stats.selectivity = alpha * selectivity + (1.0 - alpha) * stats.selectivity;

        stats.execution_count += 1;
        stats.last_updated = Utc::now();

        Ok(())
    }

    /// Add dependency between constraints
    pub fn add_dependency(&self, dependent: &str, dependency: &str) -> Result<(), ShaclAiError> {
        let mut graph = self.dependency_graph.write().map_err(|e| {
            ShaclAiError::Optimization(format!("Failed to write dependency graph: {e}"))
        })?;

        graph
            .dependencies
            .entry(dependent.to_string())
            .or_insert_with(Vec::new)
            .push(dependency.to_string());

        Ok(())
    }

    /// Update selectivity score for a constraint
    pub fn update_selectivity(
        &self,
        constraint_id: &str,
        selectivity: f64,
    ) -> Result<(), ShaclAiError> {
        let mut graph = self.dependency_graph.write().map_err(|e| {
            ShaclAiError::Optimization(format!("Failed to write dependency graph: {e}"))
        })?;

        graph
            .selectivity_scores
            .insert(constraint_id.to_string(), selectivity);

        Ok(())
    }
}
