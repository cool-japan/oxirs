//! Advanced Join Algorithm Selection and Execution
//!
//! This module provides intelligent join algorithm selection based on cost estimates
//! and adaptive execution strategies for optimal performance.

use crate::algebra::{Solution, Variable};
use crate::cost_model::CostModel;
use crate::executor::parallel_optimized::{CacheFriendlyHashJoin, SortMergeJoin};
use anyhow::Result;
use std::collections::HashSet;

/// Intelligent join algorithm selector
pub struct JoinAlgorithmSelector {
    #[allow(dead_code)]
    cost_model: CostModel,
    hash_join: CacheFriendlyHashJoin,
    sort_merge_join: SortMergeJoin,
    memory_threshold: usize,
}

impl JoinAlgorithmSelector {
    /// Create a new join algorithm selector
    pub fn new(cost_model: CostModel, memory_threshold: usize) -> Self {
        Self {
            cost_model,
            hash_join: CacheFriendlyHashJoin::new(16), // 16 partitions
            sort_merge_join: SortMergeJoin::new(memory_threshold),
            memory_threshold,
        }
    }

    /// Select and execute optimal join algorithm
    pub fn execute_optimal_join(
        &mut self,
        left_solutions: Vec<Solution>,
        right_solutions: Vec<Solution>,
        join_variables: &[Variable],
    ) -> Result<(Vec<Solution>, JoinExecutionStats)> {
        let start_time = std::time::Instant::now();

        // Analyze join characteristics
        let join_info =
            self.analyze_join_characteristics(&left_solutions, &right_solutions, join_variables);

        // Select optimal algorithm
        let selected_algorithm = self.select_join_algorithm(&join_info)?;

        // Execute selected algorithm
        let result = match selected_algorithm {
            OptimalJoinAlgorithm::HashJoin => {
                self.hash_join
                    .join_parallel(left_solutions, right_solutions, join_variables)?
            }
            OptimalJoinAlgorithm::SortMergeJoin => {
                self.sort_merge_join
                    .join(left_solutions, right_solutions, join_variables)?
            }
            OptimalJoinAlgorithm::NestedLoopJoin => {
                self.execute_nested_loop_join(left_solutions, right_solutions, join_variables)?
            }
            OptimalJoinAlgorithm::IndexJoin => {
                // Fall back to hash join for now
                self.hash_join
                    .join_parallel(left_solutions, right_solutions, join_variables)?
            }
        };

        let execution_time = start_time.elapsed();
        let stats = JoinExecutionStats {
            algorithm_used: selected_algorithm,
            execution_time,
            input_cardinalities: (join_info.left_cardinality, join_info.right_cardinality),
            output_cardinality: result.len(),
            memory_used: self.estimate_memory_usage(&result),
            join_selectivity: result.len() as f64
                / (join_info.left_cardinality as f64 * join_info.right_cardinality as f64).max(1.0),
        };

        Ok((result, stats))
    }

    /// Analyze characteristics of the join operation
    fn analyze_join_characteristics(
        &self,
        left_solutions: &[Solution],
        right_solutions: &[Solution],
        join_variables: &[Variable],
    ) -> JoinCharacteristics {
        let left_cardinality = left_solutions.len();
        let right_cardinality = right_solutions.len();

        // Estimate selectivity based on distinct values in join columns
        let left_distinct = self.estimate_distinct_values(left_solutions, join_variables);
        let right_distinct = self.estimate_distinct_values(right_solutions, join_variables);

        let estimated_selectivity = if left_distinct > 0 && right_distinct > 0 {
            1.0 / (left_distinct.max(right_distinct) as f64)
        } else {
            0.1 // Default selectivity
        };

        // Check if data is pre-sorted
        let left_sorted = self.is_sorted_by_join_keys(left_solutions, join_variables);
        let right_sorted = self.is_sorted_by_join_keys(right_solutions, join_variables);

        // Estimate memory requirements
        let memory_requirement = (left_cardinality + right_cardinality) * 100; // Rough estimate

        JoinCharacteristics {
            left_cardinality,
            right_cardinality,
            left_distinct_values: left_distinct,
            right_distinct_values: right_distinct,
            estimated_selectivity,
            left_sorted,
            right_sorted,
            memory_requirement,
            join_variable_count: join_variables.len(),
        }
    }

    /// Select optimal join algorithm based on characteristics
    fn select_join_algorithm(
        &mut self,
        join_info: &JoinCharacteristics,
    ) -> Result<OptimalJoinAlgorithm> {
        // Rule-based selection with cost model validation
        let candidate_algorithm =
            if join_info.left_cardinality < 1000 || join_info.right_cardinality < 1000 {
                // Small inputs: nested loop is often fastest
                OptimalJoinAlgorithm::NestedLoopJoin
            } else if join_info.left_sorted && join_info.right_sorted {
                // Both sides sorted: sort-merge is optimal
                OptimalJoinAlgorithm::SortMergeJoin
            } else if join_info.memory_requirement > self.memory_threshold {
                // Large memory requirement: sort-merge with external sorting
                OptimalJoinAlgorithm::SortMergeJoin
            } else if join_info.estimated_selectivity < 0.01 {
                // Very selective join: hash join is good
                OptimalJoinAlgorithm::HashJoin
            } else {
                // Default: hash join
                OptimalJoinAlgorithm::HashJoin
            };

        // TODO: Validate with cost model and potentially override
        // For now, return the rule-based selection
        Ok(candidate_algorithm)
    }

    /// Estimate number of distinct values in join columns
    fn estimate_distinct_values(
        &self,
        solutions: &[Solution],
        join_variables: &[Variable],
    ) -> usize {
        let mut distinct_values = HashSet::new();

        for solution in solutions {
            for binding in solution {
                for var in join_variables {
                    if let Some(term) = binding.get(var) {
                        distinct_values.insert(term.clone());
                    }
                }
            }
        }

        distinct_values.len()
    }

    /// Check if solutions are sorted by join key variables
    fn is_sorted_by_join_keys(&self, solutions: &[Solution], join_variables: &[Variable]) -> bool {
        if solutions.len() <= 1 {
            return true;
        }

        // Check a sample of the data for sortedness
        let sample_size = (solutions.len() / 10).clamp(10, 100);
        let step = solutions.len() / sample_size;

        for i in 1..sample_size {
            let idx = i * step;
            if idx >= solutions.len() {
                break;
            }

            if self.compare_solutions_by_join_key(
                &solutions[idx - step],
                &solutions[idx],
                join_variables,
            ) == std::cmp::Ordering::Greater
            {
                return false;
            }
        }

        true
    }

    /// Compare solutions by join key (simplified version)
    fn compare_solutions_by_join_key(
        &self,
        left: &Solution,
        right: &Solution,
        join_variables: &[Variable],
    ) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        let left_binding = left.first();
        let right_binding = right.first();

        match (left_binding, right_binding) {
            (Some(l_binding), Some(r_binding)) => {
                for var in join_variables {
                    let left_term = l_binding.get(var);
                    let right_term = r_binding.get(var);

                    let cmp = match (left_term, right_term) {
                        (Some(l), Some(r)) => {
                            // Simple string comparison for now
                            format!("{l}").cmp(&format!("{r}"))
                        }
                        (Some(_), None) => Ordering::Greater,
                        (None, Some(_)) => Ordering::Less,
                        (None, None) => Ordering::Equal,
                    };

                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            }
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (None, None) => Ordering::Equal,
        }
    }

    /// Execute nested loop join
    fn execute_nested_loop_join(
        &self,
        left_solutions: Vec<Solution>,
        right_solutions: Vec<Solution>,
        join_variables: &[Variable],
    ) -> Result<Vec<Solution>> {
        let mut result = Vec::new();

        for left_solution in &left_solutions {
            for right_solution in &right_solutions {
                if let Some(merged) =
                    self.try_merge_solutions(left_solution, right_solution, join_variables)?
                {
                    result.push(merged);
                }
            }
        }

        Ok(result)
    }

    /// Try to merge two solutions if they are compatible on join variables
    fn try_merge_solutions(
        &self,
        left: &Solution,
        right: &Solution,
        join_variables: &[Variable],
    ) -> Result<Option<Solution>> {
        let mut result = Vec::new();

        for left_binding in left {
            for right_binding in right {
                // Check compatibility on join variables
                let mut compatible = true;
                for var in join_variables {
                    if let (Some(left_term), Some(right_term)) =
                        (left_binding.get(var), right_binding.get(var))
                    {
                        if left_term != right_term {
                            compatible = false;
                            break;
                        }
                    }
                }

                if compatible {
                    // Merge bindings
                    let mut merged_binding = left_binding.clone();
                    for (var, term) in right_binding {
                        // Only add if not already present (join variables will be the same)
                        if !merged_binding.contains_key(var) {
                            merged_binding.insert(var.clone(), term.clone());
                        }
                    }
                    result.push(merged_binding);
                }
            }
        }

        if result.is_empty() {
            Ok(None)
        } else {
            Ok(Some(result))
        }
    }

    /// Estimate memory usage of a solution set
    fn estimate_memory_usage(&self, solutions: &[Solution]) -> usize {
        solutions.len() * 1024 // Rough estimate: 1KB per solution
    }
}

/// Join algorithm options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimalJoinAlgorithm {
    HashJoin,
    SortMergeJoin,
    NestedLoopJoin,
    IndexJoin,
}

/// Join characteristics for algorithm selection
#[derive(Debug, Clone)]
pub struct JoinCharacteristics {
    pub left_cardinality: usize,
    pub right_cardinality: usize,
    pub left_distinct_values: usize,
    pub right_distinct_values: usize,
    pub estimated_selectivity: f64,
    pub left_sorted: bool,
    pub right_sorted: bool,
    pub memory_requirement: usize,
    pub join_variable_count: usize,
}

/// Join execution statistics
#[derive(Debug, Clone)]
pub struct JoinExecutionStats {
    pub algorithm_used: OptimalJoinAlgorithm,
    pub execution_time: std::time::Duration,
    pub input_cardinalities: (usize, usize),
    pub output_cardinality: usize,
    pub memory_used: usize,
    pub join_selectivity: f64,
}

impl JoinExecutionStats {
    /// Get performance metrics as a human-readable string
    pub fn performance_summary(&self) -> String {
        format!(
            "Algorithm: {:?}, Time: {:?}, Input: ({}, {}), Output: {}, Selectivity: {:.4}, Memory: {} bytes",
            self.algorithm_used,
            self.execution_time,
            self.input_cardinalities.0,
            self.input_cardinalities.1,
            self.output_cardinality,
            self.join_selectivity,
            self.memory_used
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Binding, Term, Variable};
    use crate::cost_model::{CostModel, CostModelConfig};
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_join_algorithm_selection() {
        let cost_model = CostModel::new(CostModelConfig::default());
        let mut selector = JoinAlgorithmSelector::new(cost_model, 1024 * 1024); // 1MB threshold

        // Create test data
        let var_x = Variable::new("x").unwrap();
        let _var_y = Variable::new("y").unwrap();

        let left_solutions = vec![create_test_solution(&var_x, "value1")];
        let right_solutions = vec![create_test_solution(&var_x, "value1")];
        let join_variables = vec![var_x];

        let result =
            selector.execute_optimal_join(left_solutions, right_solutions, &join_variables);
        assert!(result.is_ok());

        let (solutions, stats) = result.unwrap();
        assert!(!solutions.is_empty());
        println!("Join stats: {}", stats.performance_summary());
    }

    fn create_test_solution(variable: &Variable, value: &str) -> Solution {
        let mut binding = Binding::new();
        binding.insert(
            variable.clone(),
            Term::Iri(NamedNode::new_unchecked(format!(
                "http://example.org/{value}"
            ))),
        );
        vec![binding]
    }
}
