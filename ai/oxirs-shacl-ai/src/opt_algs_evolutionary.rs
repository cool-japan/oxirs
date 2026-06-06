//! Evolutionary optimization algorithms for SHACL constraint optimization.
//!
//! Contains:
//! - `GeneticOptimizer` with tournament selection, crossover, mutation
//! - `MultiObjectiveOptimizer` with NSGA-II non-dominated sorting
//! - Supporting types: chromosomes, objectives, Pareto front
//! - Additional algorithm support types: DEIndividual, TabuMove, RL/Adaptive types

use crate::{shape::PropertyConstraint, Result, ShaclAiError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::optimization_engine::{
    CacheConfiguration, ParallelExecutionStrategy, ParallelValidationConfig,
    PerformanceMetrics as OptimizationPerformanceMetrics,
};

// Re-export the optimization_engine PerformanceMetrics so callers can use it through this module.
pub use crate::optimization_engine::PerformanceMetrics;

/// Chromosome for genetic algorithm optimisation
#[derive(Debug, Clone)]
pub struct OptimizationChromosome {
    pub constraint_order: Vec<usize>,
    pub parallelization_config: ParallelValidationConfig,
    pub cache_config: CacheConfiguration,
    pub fitness: f64,
}

impl OptimizationChromosome {
    pub fn random(constraints: &[PropertyConstraint]) -> Result<Self> {
        let mut constraint_order: Vec<usize> = (0..constraints.len()).collect();
        for i in (1..constraint_order.len()).rev() {
            let j = fastrand::usize(0..=i);
            constraint_order.swap(i, j);
        }

        Ok(Self {
            constraint_order,
            parallelization_config: ParallelValidationConfig {
                enabled: true,
                max_parallel_constraints: fastrand::usize(1..=8),
                constraint_groups: Vec::new(),
                execution_strategy: ParallelExecutionStrategy::GroupBased,
                estimated_speedup: 1.0,
            },
            cache_config: CacheConfiguration {
                enabled: true,
                cacheable_constraints: Vec::new(),
                cache_strategies: Vec::new(),
                estimated_hit_rate: fastrand::f64(),
                memory_limit_mb: fastrand::f64() * 100.0 + 10.0,
            },
            fitness: 0.0,
        })
    }
}

/// Optimisation objectives
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    MinimizeTime,
    MinimizeMemory,
    MaximizeCacheEfficiency,
    Balanced,
}

/// Result of constraint-order optimisation
#[derive(Debug, Clone)]
pub struct OptimizedConstraintConfiguration {
    pub constraint_order: Vec<usize>,
    pub parallelization_config: ParallelValidationConfig,
    pub cache_configuration: CacheConfiguration,
    pub fitness_score: f64,
    pub optimization_metadata: GeneticOptimizationMetadata,
}

/// Metadata produced by the genetic algorithm run
#[derive(Debug, Clone)]
pub struct GeneticOptimizationMetadata {
    pub generations: usize,
    pub final_fitness: f64,
    pub convergence_generation: usize,
    pub population_size: usize,
}

/// A concrete optimisation solution (wraps constraint configuration + metrics)
#[derive(Debug, Clone)]
pub struct OptimizationSolution {
    pub constraint_configuration: OptimizedConstraintConfiguration,
    pub performance_metrics: PerformanceMetrics,
}

/// Advanced Genetic Algorithm Optimizer for SHACL Constraints
#[derive(Debug)]
pub struct GeneticOptimizer {
    population_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    elite_ratio: f64,
    max_generations: usize,
    population: Vec<OptimizationChromosome>,
    fitness_history: Vec<f64>,
}

impl GeneticOptimizer {
    pub fn new() -> Self {
        Self {
            population_size: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            elite_ratio: 0.2,
            max_generations: 50,
            population: Vec::new(),
            fitness_history: Vec::new(),
        }
    }

    /// Optimize constraint configuration using genetic algorithm
    pub async fn optimize_constraints(
        &mut self,
        constraints: &[PropertyConstraint],
        objective: OptimizationObjective,
    ) -> Result<OptimizedConstraintConfiguration> {
        self.initialize_population(constraints)?;

        let mut best_solution = None;
        let mut best_fitness = f64::NEG_INFINITY;

        for generation in 0..self.max_generations {
            let mut fitness_values = Vec::new();
            for chromosome in &self.population {
                fitness_values.push(self.evaluate_fitness(chromosome, &objective).await?);
            }
            for (chromosome, fitness) in self.population.iter_mut().zip(fitness_values) {
                chromosome.fitness = fitness;
            }

            if let Some(best_chromosome) = self.population.iter().max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .expect("fitness values should be comparable")
            }) {
                if best_chromosome.fitness > best_fitness {
                    best_fitness = best_chromosome.fitness;
                    best_solution = Some(best_chromosome.clone());
                }
            }

            self.fitness_history.push(best_fitness);
            self.evolve_population().await?;

            tracing::debug!(
                "Generation {}: Best fitness = {:.4}",
                generation,
                best_fitness
            );
        }

        let best_chromosome = best_solution
            .ok_or_else(|| ShaclAiError::ShapeManagement("No valid solution found".to_string()))?;

        Ok(OptimizedConstraintConfiguration {
            constraint_order: best_chromosome.constraint_order,
            parallelization_config: best_chromosome.parallelization_config,
            cache_configuration: best_chromosome.cache_config,
            fitness_score: best_chromosome.fitness,
            optimization_metadata: GeneticOptimizationMetadata {
                generations: self.max_generations,
                final_fitness: best_fitness,
                convergence_generation: self.fitness_history.len(),
                population_size: self.population_size,
            },
        })
    }

    fn initialize_population(&mut self, constraints: &[PropertyConstraint]) -> Result<()> {
        self.population.clear();
        for _ in 0..self.population_size {
            self.population
                .push(OptimizationChromosome::random(constraints)?);
        }
        Ok(())
    }

    async fn evaluate_fitness(
        &self,
        chromosome: &OptimizationChromosome,
        objective: &OptimizationObjective,
    ) -> Result<f64> {
        let execution_time = self.simulate_execution_time(&chromosome.constraint_order)?;
        let memory_usage = self.simulate_memory_usage(&chromosome.parallelization_config)?;
        let cache_efficiency = self.simulate_cache_efficiency(&chromosome.cache_config)?;

        let fitness = match objective {
            OptimizationObjective::MinimizeTime => 1000.0 / (execution_time + 1.0),
            OptimizationObjective::MinimizeMemory => 1000.0 / (memory_usage + 1.0),
            OptimizationObjective::MaximizeCacheEfficiency => cache_efficiency * 100.0,
            OptimizationObjective::Balanced => {
                let time_score = 1000.0 / (execution_time + 1.0);
                let memory_score = 1000.0 / (memory_usage + 1.0);
                let cache_score = cache_efficiency * 100.0;
                time_score * 0.5 + memory_score * 0.3 + cache_score * 0.2
            }
        };

        Ok(fitness)
    }

    async fn evolve_population(&mut self) -> Result<()> {
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .expect("fitness values should be comparable")
        });

        let elite_count = (self.population_size as f64 * self.elite_ratio) as usize;
        let mut new_population = Vec::new();
        new_population.extend_from_slice(&self.population[0..elite_count]);

        while new_population.len() < self.population_size {
            let parent1 = self.tournament_selection(3)?;
            let parent2 = self.tournament_selection(3)?;

            let mut offspring = if fastrand::f64() < self.crossover_rate {
                self.crossover(&parent1, &parent2)?
            } else {
                parent1.clone()
            };

            if fastrand::f64() < self.mutation_rate {
                self.mutate(&mut offspring)?;
            }

            new_population.push(offspring);
        }

        self.population = new_population;
        Ok(())
    }

    fn tournament_selection(&self, tournament_size: usize) -> Result<OptimizationChromosome> {
        let mut best_individual = None;
        let mut best_fitness = f64::NEG_INFINITY;

        for _ in 0..tournament_size {
            let idx = fastrand::usize(0..self.population.len());
            let individual = &self.population[idx];
            if individual.fitness > best_fitness {
                best_fitness = individual.fitness;
                best_individual = Some(individual.clone());
            }
        }

        best_individual
            .ok_or_else(|| ShaclAiError::ShapeManagement("Tournament selection failed".to_string()))
    }

    fn crossover(
        &self,
        parent1: &OptimizationChromosome,
        parent2: &OptimizationChromosome,
    ) -> Result<OptimizationChromosome> {
        let crossover_point = fastrand::usize(1..parent1.constraint_order.len());
        let mut child_order = parent1.constraint_order[0..crossover_point].to_vec();
        child_order.extend_from_slice(&parent2.constraint_order[crossover_point..]);

        Ok(OptimizationChromosome {
            constraint_order: child_order,
            parallelization_config: if fastrand::bool() {
                parent1.parallelization_config.clone()
            } else {
                parent2.parallelization_config.clone()
            },
            cache_config: if fastrand::bool() {
                parent1.cache_config.clone()
            } else {
                parent2.cache_config.clone()
            },
            fitness: 0.0,
        })
    }

    fn mutate(&self, chromosome: &mut OptimizationChromosome) -> Result<()> {
        if chromosome.constraint_order.len() > 1 {
            let idx1 = fastrand::usize(0..chromosome.constraint_order.len());
            let idx2 = fastrand::usize(0..chromosome.constraint_order.len());
            chromosome.constraint_order.swap(idx1, idx2);
        }

        if fastrand::f64() < 0.3 {
            chromosome.parallelization_config.max_parallel_constraints =
                (chromosome.parallelization_config.max_parallel_constraints
                    + fastrand::i8(-2..=2) as usize)
                    .max(1)
                    .min(16);
        }

        if fastrand::f64() < 0.3 {
            chromosome.cache_config.estimated_hit_rate =
                (chromosome.cache_config.estimated_hit_rate + fastrand::f64() * 0.2 - 0.1)
                    .clamp(0.0, 1.0);
        }

        Ok(())
    }

    fn simulate_execution_time(&self, constraint_order: &[usize]) -> Result<f64> {
        let mut total_time = 0.0;
        let mut failure_probability = 0.0;
        for &constraint_idx in constraint_order {
            let constraint_time = match constraint_idx % 5 {
                0 => 1.0,
                1 => 2.5,
                2 => 5.0,
                3 => 10.0,
                _ => 1.5,
            };
            total_time += constraint_time * (1.0 - failure_probability);
            failure_probability += 0.1;
        }
        Ok(total_time)
    }

    fn simulate_memory_usage(
        &self,
        parallelization_config: &ParallelValidationConfig,
    ) -> Result<f64> {
        let base_memory = 10.0;
        let parallel_overhead = parallelization_config.max_parallel_constraints as f64 * 2.0;
        Ok(base_memory + parallel_overhead)
    }

    fn simulate_cache_efficiency(&self, cache_config: &CacheConfiguration) -> Result<f64> {
        Ok(cache_config.estimated_hit_rate)
    }
}

impl Default for GeneticOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-objective solution for NSGA-II
#[derive(Debug, Clone)]
pub struct MultiObjectiveSolution {
    pub parameters: Vec<f64>,
    pub objective_values: Vec<f64>,
}

impl MultiObjectiveSolution {
    pub fn random() -> Result<Self> {
        let num_parameters = 5;
        let parameters: Vec<f64> = (0..num_parameters)
            .map(|_| fastrand::f64() * 2.0 - 1.0)
            .collect();
        Ok(Self {
            parameters,
            objective_values: Vec::new(),
        })
    }
}

/// Pareto front from multi-objective optimisation
#[derive(Debug, Clone)]
pub struct ParetoFront {
    pub solutions: Vec<MultiObjectiveSolution>,
    pub hypervolume: f64,
    pub generation: usize,
}

/// Multi-Objective Optimizer using NSGA-II
#[derive(Debug)]
pub struct MultiObjectiveOptimizer {
    population_size: usize,
    max_generations: usize,
    crossover_probability: f64,
    mutation_probability: f64,
    population: Vec<MultiObjectiveSolution>,
}

impl MultiObjectiveOptimizer {
    pub fn new() -> Self {
        Self {
            population_size: 100,
            max_generations: 50,
            crossover_probability: 0.9,
            mutation_probability: 0.1,
            population: Vec::new(),
        }
    }

    /// Optimize multiple objectives simultaneously using NSGA-II
    pub async fn optimize(
        &mut self,
        objectives: Vec<OptimizationObjective>,
    ) -> Result<ParetoFront> {
        self.initialize_population()?;

        for generation in 0..self.max_generations {
            let mut objective_values = Vec::new();
            for solution in &self.population {
                objective_values.push(self.evaluate_objectives(solution, &objectives).await?);
            }
            for (solution, values) in self.population.iter_mut().zip(objective_values) {
                solution.objective_values = values;
            }

            let fronts = self.non_dominated_sort(&self.population)?;
            self.population = self.select_next_generation(&fronts)?;

            tracing::debug!(
                "Multi-objective Generation {}: {} fronts, {} solutions",
                generation,
                fronts.len(),
                self.population.len()
            );
        }

        let fronts = self.non_dominated_sort(&self.population)?;
        let pareto_front = fronts.into_iter().next().unwrap_or_default();

        Ok(ParetoFront {
            solutions: pareto_front.clone(),
            hypervolume: self.calculate_hypervolume(&pareto_front),
            generation: self.max_generations,
        })
    }

    fn initialize_population(&mut self) -> Result<()> {
        self.population.clear();
        for _ in 0..self.population_size {
            self.population.push(MultiObjectiveSolution::random()?);
        }
        Ok(())
    }

    async fn evaluate_objectives(
        &self,
        solution: &MultiObjectiveSolution,
        objectives: &[OptimizationObjective],
    ) -> Result<Vec<f64>> {
        let mut objective_values = Vec::new();
        for objective in objectives {
            let value = match objective {
                OptimizationObjective::MinimizeTime => {
                    solution.parameters.iter().sum::<f64>() * 10.0
                }
                OptimizationObjective::MinimizeMemory => {
                    solution.parameters.iter().map(|x| x * x).sum::<f64>() * 5.0
                }
                OptimizationObjective::MaximizeCacheEfficiency => {
                    -(solution
                        .parameters
                        .iter()
                        .map(|x| 1.0 - x.abs())
                        .sum::<f64>()
                        / solution.parameters.len() as f64)
                }
                OptimizationObjective::Balanced => {
                    let time_component = solution.parameters.iter().sum::<f64>() * 10.0;
                    let memory_component =
                        solution.parameters.iter().map(|x| x * x).sum::<f64>() * 5.0;
                    time_component * 0.6 + memory_component * 0.4
                }
            };
            objective_values.push(value);
        }
        Ok(objective_values)
    }

    fn non_dominated_sort(
        &self,
        population: &[MultiObjectiveSolution],
    ) -> Result<Vec<Vec<MultiObjectiveSolution>>> {
        let mut fronts: Vec<Vec<MultiObjectiveSolution>> = Vec::new();
        let mut first_front = Vec::new();
        let mut domination_counts = vec![0; population.len()];
        let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); population.len()];

        for (i, solution_a) in population.iter().enumerate() {
            for (j, solution_b) in population.iter().enumerate() {
                if i != j {
                    if self.dominates(solution_a, solution_b) {
                        dominated_solutions[i].push(j);
                    } else if self.dominates(solution_b, solution_a) {
                        domination_counts[i] += 1;
                    }
                }
            }
            if domination_counts[i] == 0 {
                first_front.push(solution_a.clone());
            }
        }

        fronts.push(first_front);

        let mut current_front_idx = 0;
        while current_front_idx < fronts.len() && !fronts[current_front_idx].is_empty() {
            let mut next_front = Vec::new();

            for solution_idx in 0..population.len() {
                if domination_counts[solution_idx] == 0
                    && !fronts
                        .iter()
                        .flatten()
                        .any(|s| std::ptr::eq(s, &population[solution_idx]))
                {
                    continue;
                }

                for &dominated_idx in &dominated_solutions[solution_idx] {
                    domination_counts[dominated_idx] -= 1;
                    if domination_counts[dominated_idx] == 0 {
                        next_front.push(population[dominated_idx].clone());
                    }
                }
            }

            if !next_front.is_empty() {
                fronts.push(next_front);
            }
            current_front_idx += 1;
        }

        Ok(fronts)
    }

    fn dominates(
        &self,
        solution_a: &MultiObjectiveSolution,
        solution_b: &MultiObjectiveSolution,
    ) -> bool {
        let mut at_least_one_better = false;
        for (obj_a, obj_b) in solution_a
            .objective_values
            .iter()
            .zip(solution_b.objective_values.iter())
        {
            if obj_a > obj_b {
                return false;
            }
            if obj_a < obj_b {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }

    fn select_next_generation(
        &self,
        fronts: &[Vec<MultiObjectiveSolution>],
    ) -> Result<Vec<MultiObjectiveSolution>> {
        let mut next_population = Vec::new();
        for front in fronts {
            if next_population.len() + front.len() <= self.population_size {
                next_population.extend_from_slice(front);
            } else {
                let remaining_slots = self.population_size - next_population.len();
                let selected = self.select_by_crowding_distance(front, remaining_slots)?;
                next_population.extend(selected);
                break;
            }
        }
        Ok(next_population)
    }

    fn select_by_crowding_distance(
        &self,
        front: &[MultiObjectiveSolution],
        count: usize,
    ) -> Result<Vec<MultiObjectiveSolution>> {
        let mut solutions_with_distance: Vec<(MultiObjectiveSolution, f64)> = front
            .iter()
            .map(|s| (s.clone(), self.calculate_crowding_distance(s, front)))
            .collect();
        solutions_with_distance.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("distance values should be comparable")
        });
        Ok(solutions_with_distance
            .into_iter()
            .take(count)
            .map(|(solution, _)| solution)
            .collect())
    }

    fn calculate_crowding_distance(
        &self,
        solution: &MultiObjectiveSolution,
        front: &[MultiObjectiveSolution],
    ) -> f64 {
        if front.len() <= 2 {
            return f64::INFINITY;
        }

        let mut distance = 0.0;
        let num_objectives = solution.objective_values.len();

        for obj_idx in 0..num_objectives {
            let mut sorted_front: Vec<_> = front.iter().collect();
            sorted_front.sort_by(|a, b| {
                a.objective_values[obj_idx]
                    .partial_cmp(&b.objective_values[obj_idx])
                    .expect("objective values should be comparable")
            });

            let solution_idx = sorted_front
                .iter()
                .position(|s| std::ptr::eq(*s, solution))
                .expect("solution should be in sorted front");

            if solution_idx == 0 || solution_idx == sorted_front.len() - 1 {
                distance = f64::INFINITY;
                break;
            }

            let obj_range = sorted_front
                .last()
                .expect("sorted_front should not be empty")
                .objective_values[obj_idx]
                - sorted_front
                    .first()
                    .expect("sorted_front should not be empty")
                    .objective_values[obj_idx];

            if obj_range > 0.0 {
                distance += (sorted_front[solution_idx + 1].objective_values[obj_idx]
                    - sorted_front[solution_idx - 1].objective_values[obj_idx])
                    / obj_range;
            }
        }

        distance
    }

    fn calculate_hypervolume(&self, pareto_front: &[MultiObjectiveSolution]) -> f64 {
        if pareto_front.is_empty() {
            return 0.0;
        }
        let reference_point = vec![1000.0; pareto_front[0].objective_values.len()];
        let mut hypervolume = 0.0;
        for solution in pareto_front {
            let mut volume = 1.0;
            for (obj_val, ref_val) in solution.objective_values.iter().zip(reference_point.iter()) {
                volume *= (ref_val - obj_val).max(0.0);
            }
            hypervolume += volume;
        }
        hypervolume
    }
}

impl Default for MultiObjectiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Supporting types shared across algorithm implementations ─────────────────

/// Differential evolution individual
#[derive(Debug, Clone)]
pub struct DEIndividual {
    pub parameters: Vec<f64>,
    pub fitness: f64,
}

impl DEIndividual {
    pub fn random(search_space: &OptimizationSearchSpace) -> Result<Self> {
        Ok(Self {
            parameters: vec![
                fastrand::f64()
                    * (search_space.execution_time_range.1 - search_space.execution_time_range.0)
                    + search_space.execution_time_range.0,
                fastrand::f64()
                    * (search_space.memory_usage_range.1 - search_space.memory_usage_range.0)
                    + search_space.memory_usage_range.0,
                fastrand::f64()
                    * (search_space.cache_efficiency_range.1
                        - search_space.cache_efficiency_range.0)
                    + search_space.cache_efficiency_range.0,
            ],
            fitness: 0.0,
        })
    }

    pub fn to_optimization_point(&self) -> OptimizationPoint {
        OptimizationPoint {
            execution_time_weight: self.parameters[0],
            memory_usage_weight: self.parameters[1],
            cache_efficiency_weight: self.parameters[2],
        }
    }
}

/// Tabu search move
#[derive(Debug, Clone, PartialEq)]
pub struct TabuMove {
    pub description: String,
    pub move_type: TabuMoveType,
}

impl TabuMove {
    pub fn from_solutions(from: &OptimizationSolution, to: &OptimizationSolution) -> Self {
        Self {
            description: format!(
                "Move from solution {} to {}",
                from.constraint_configuration.fitness_score,
                to.constraint_configuration.fitness_score
            ),
            move_type: TabuMoveType::ConstraintReordering,
        }
    }
}

/// Types of tabu moves
#[derive(Debug, Clone, PartialEq)]
pub enum TabuMoveType {
    ConstraintReordering,
    ParallelizationChange,
    CacheConfigChange,
}

/// State-action pair for reinforcement learning
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StateActionPair {
    pub state: OptimizationState,
    pub action: OptimizationAction,
}

/// RL optimisation state
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptimizationState {
    pub parallel_threads: usize,
    pub cache_size_mb: u64,
    pub constraint_order_entropy: u64,
}

/// RL optimisation action
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OptimizationAction {
    IncreaseParallelism,
    DecreaseParallelism,
    IncreaseCacheSize,
    DecreaseCacheSize,
    ReorderConstraints,
    NoAction,
}

impl OptimizationAction {
    pub fn random() -> Self {
        match fastrand::u8(0..6) {
            0 => OptimizationAction::IncreaseParallelism,
            1 => OptimizationAction::DecreaseParallelism,
            2 => OptimizationAction::IncreaseCacheSize,
            3 => OptimizationAction::DecreaseCacheSize,
            4 => OptimizationAction::ReorderConstraints,
            _ => OptimizationAction::NoAction,
        }
    }

    pub fn from_id(id: usize) -> Self {
        match id % 6 {
            0 => OptimizationAction::IncreaseParallelism,
            1 => OptimizationAction::DecreaseParallelism,
            2 => OptimizationAction::IncreaseCacheSize,
            3 => OptimizationAction::DecreaseCacheSize,
            4 => OptimizationAction::ReorderConstraints,
            _ => OptimizationAction::NoAction,
        }
    }
}

/// Learned RL policy
#[derive(Debug, Clone)]
pub struct OptimizationPolicy {
    pub state_action_mapping: HashMap<OptimizationState, OptimizationAction>,
    pub confidence: f64,
}

/// Optimisation problem description
#[derive(Debug, Clone)]
pub struct OptimizationProblem {
    pub constraints: Vec<PropertyConstraint>,
    pub baseline_metrics: PerformanceMetrics,
}

impl OptimizationProblem {
    pub fn extract_features(&self) -> ProblemFeatures {
        ProblemFeatures {
            num_constraints: self.constraints.len(),
            avg_complexity: self
                .constraints
                .iter()
                .map(|c| match c.constraint_type().as_str() {
                    "sh:pattern" => 3.5,
                    "sh:sparql" => 4.0,
                    "sh:class" => 2.5,
                    _ => 1.5,
                })
                .sum::<f64>()
                / self.constraints.len().max(1) as f64,
            baseline_execution_time: self.baseline_metrics.validation_time_ms,
            baseline_memory_usage: self.baseline_metrics.memory_usage_mb,
        }
    }
}

/// Features extracted from a problem for ML-based strategy selection
#[derive(Debug, Clone)]
pub struct ProblemFeatures {
    pub num_constraints: usize,
    pub avg_complexity: f64,
    pub baseline_execution_time: f64,
    pub baseline_memory_usage: f64,
}

/// Optimisation strategy descriptor
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    pub use_parallel_execution: bool,
    pub cache_strategy: CacheStrategyType,
    pub constraint_ordering: ConstraintOrderingType,
    pub memory_optimization: bool,
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self {
            use_parallel_execution: true,
            cache_strategy: CacheStrategyType::ResultCaching,
            constraint_ordering: ConstraintOrderingType::CostBased,
            memory_optimization: true,
        }
    }
}

impl OptimizationStrategy {
    pub fn extract_features(&self) -> StrategyFeatures {
        StrategyFeatures {
            parallel_execution: if self.use_parallel_execution {
                1.0
            } else {
                0.0
            },
            cache_type: match self.cache_strategy {
                CacheStrategyType::ResultCaching => 1.0,
                CacheStrategyType::QueryCaching => 2.0,
                CacheStrategyType::DataCaching => 3.0,
                _ => 0.0,
            },
            ordering_type: match self.constraint_ordering {
                ConstraintOrderingType::CostBased => 1.0,
                ConstraintOrderingType::FailFast => 2.0,
                ConstraintOrderingType::DependencyBased => 3.0,
            },
            memory_optimization: if self.memory_optimization { 1.0 } else { 0.0 },
        }
    }
}

/// Cache strategy type (mirrors the engine's CacheStrategyType for extensibility)
#[derive(Debug, Clone)]
pub enum CacheStrategyType {
    ResultCaching,
    QueryCaching,
    DataCaching,
    NoCache,
}

/// Constraint ordering approach
#[derive(Debug, Clone)]
pub enum ConstraintOrderingType {
    CostBased,
    FailFast,
    DependencyBased,
}

/// Features of a strategy for ML prediction
#[derive(Debug, Clone)]
pub struct StrategyFeatures {
    pub parallel_execution: f64,
    pub cache_type: f64,
    pub ordering_type: f64,
    pub memory_optimization: f64,
}

/// Result from adaptive optimisation
#[derive(Debug, Clone)]
pub struct AdaptiveOptimizationResult {
    pub strategy_applied: OptimizationStrategy,
    pub baseline_performance: PerformanceMetrics,
    pub optimized_performance: PerformanceMetrics,
    pub performance_improvement: f64,
    pub optimization_duration: Duration,
    pub confidence: f64,
}

/// Historical optimisation record
#[derive(Debug, Clone)]
pub struct HistoricalOptimization {
    pub problem: OptimizationProblem,
    pub strategy: OptimizationStrategy,
    pub result: AdaptiveOptimizationResult,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Adaptive performance model (simple linear predictor)
#[derive(Debug)]
pub struct AdaptivePerformanceModel {
    model_parameters: Vec<f64>,
    confidence_score: f64,
}

impl AdaptivePerformanceModel {
    pub fn new() -> Self {
        Self {
            model_parameters: vec![0.5, 0.3, 0.2],
            confidence_score: 0.5,
        }
    }

    pub fn train(&mut self, examples: &[PerformanceTrainingExample]) -> Result<()> {
        if !examples.is_empty() {
            self.confidence_score = 0.8;
            self.model_parameters = vec![0.6, 0.25, 0.15];
        }
        Ok(())
    }

    pub fn predict(
        &self,
        _problem_features: &ProblemFeatures,
        strategy_features: &StrategyFeatures,
    ) -> Result<f64> {
        let prediction = strategy_features.parallel_execution * self.model_parameters[0]
            + strategy_features.cache_type * self.model_parameters[1]
            + strategy_features.memory_optimization * self.model_parameters[2];
        Ok(prediction.clamp(0.0, 1.0))
    }

    pub fn confidence(&self) -> f64 {
        self.confidence_score
    }
}

impl Default for AdaptivePerformanceModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Training example for the performance model
#[derive(Debug, Clone)]
pub struct PerformanceTrainingExample {
    pub problem_features: ProblemFeatures,
    pub strategy_features: StrategyFeatures,
    pub performance_outcome: f64,
}

// Re-export search-space and point types defined in opt_algs_swarm
// to keep a unified public API in the facade.
pub use crate::opt_algs_swarm::{OptimizationPoint, OptimizationSearchSpace};
