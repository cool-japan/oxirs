//! Advanced optimization algorithms for constraint ordering and performance tuning

use crate::{shape::PropertyConstraint, Result, ShaclAiError};
use std::collections::{HashMap, HashSet, VecDeque};

use super::types::*;

// Bring in the evolutionary/swarm types used by the public optimize methods below.
use crate::opt_algs_evolutionary::{
    AdaptiveOptimizationResult as EvoAdaptiveResult, AdaptivePerformanceModel, OptimizationPolicy,
    OptimizationProblem as EvoProblem, OptimizationState as EvoState,
    OptimizationStrategy as EvoStrategy,
};
use crate::opt_algs_swarm::{
    OptimizationObjectiveFunction, OptimizationPoint, OptimizationResult as SwarmResult,
    OptimizationSearchSpace,
};

/// Ant Colony Optimization for Constraint Ordering
#[derive(Debug)]
pub struct AntColonyOptimizer {
    num_ants: usize,
    max_iterations: usize,
    pheromone_evaporation_rate: f64,
    pheromone_deposit_strength: f64,
    alpha: f64, // Pheromone importance
    beta: f64,  // Heuristic importance
    pheromone_matrix: Vec<Vec<f64>>,
    distance_matrix: Vec<Vec<f64>>,
    best_solution: Option<Vec<usize>>,
    best_cost: f64,
}

impl AntColonyOptimizer {
    pub fn new(num_constraints: usize) -> Self {
        let pheromone_matrix = vec![vec![1.0; num_constraints]; num_constraints];
        let distance_matrix = vec![vec![1.0; num_constraints]; num_constraints];

        Self {
            num_ants: 20,
            max_iterations: 100,
            pheromone_evaporation_rate: 0.1,
            pheromone_deposit_strength: 1.0,
            alpha: 1.0,
            beta: 2.0,
            pheromone_matrix,
            distance_matrix,
            best_solution: None,
            best_cost: f64::INFINITY,
        }
    }

    /// Alias for [`Self::optimize`] that matches the test API name.
    pub async fn optimize_constraint_order(
        &mut self,
        constraints: &[PropertyConstraint],
    ) -> Result<Vec<usize>> {
        self.optimize(constraints)
    }

    pub fn optimize(&mut self, constraints: &[PropertyConstraint]) -> Result<Vec<usize>> {
        if constraints.is_empty() {
            return Ok(Vec::new());
        }

        // Initialize distance matrix based on constraint dependencies
        self.initialize_distance_matrix(constraints)?;

        for iteration in 0..self.max_iterations {
            let mut solutions = Vec::new();

            // Generate solutions with each ant
            for _ in 0..self.num_ants {
                let solution = self.construct_ant_solution(constraints.len())?;
                let cost = self.calculate_solution_cost(&solution, constraints)?;
                solutions.push((solution, cost));
            }

            // Update best solution
            for (solution, cost) in &solutions {
                if *cost < self.best_cost {
                    self.best_cost = *cost;
                    self.best_solution = Some(solution.clone());
                }
            }

            // Update pheromones
            self.update_pheromones(&solutions)?;

            if iteration % 20 == 0 {
                tracing::debug!(
                    "ACO Iteration {}: Best cost = {:.4}",
                    iteration,
                    self.best_cost
                );
            }
        }

        self.best_solution
            .clone()
            .ok_or_else(|| ShaclAiError::ShapeManagement("ACO failed to find solution".to_string()))
    }

    fn initialize_distance_matrix(&mut self, constraints: &[PropertyConstraint]) -> Result<()> {
        let n = constraints.len();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let distance =
                        self.calculate_constraint_distance(&constraints[i], &constraints[j]);
                    self.distance_matrix[i][j] = distance;
                }
            }
        }
        Ok(())
    }

    fn calculate_constraint_distance(
        &self,
        c1: &PropertyConstraint,
        c2: &PropertyConstraint,
    ) -> f64 {
        let property_similarity = if c1.property() == c2.property() {
            0.1
        } else {
            1.0
        };
        let type_compatibility =
            match (c1.constraint_type().as_str(), c2.constraint_type().as_str()) {
                ("sh:datatype", "sh:pattern") => 0.3, // Compatible
                ("sh:class", "sh:nodeKind") => 0.4,   // Somewhat compatible
                (a, b) if a == b => 0.2,              // Same type
                _ => 1.0,                             // Default distance
            };

        property_similarity + type_compatibility
    }

    fn construct_ant_solution(&self, num_constraints: usize) -> Result<Vec<usize>> {
        let mut solution = Vec::new();
        let mut unvisited: HashSet<usize> = (0..num_constraints).collect();

        // Start from random constraint
        let mut current = fastrand::usize(0..num_constraints);
        solution.push(current);
        unvisited.remove(&current);

        while !unvisited.is_empty() {
            let next = self.select_next_constraint(current, &unvisited)?;
            solution.push(next);
            unvisited.remove(&next);
            current = next;
        }

        Ok(solution)
    }

    fn select_next_constraint(&self, current: usize, unvisited: &HashSet<usize>) -> Result<usize> {
        let mut probabilities = Vec::new();
        let mut total_probability = 0.0;

        for &next in unvisited {
            let pheromone = self.pheromone_matrix[current][next];
            let heuristic = 1.0 / self.distance_matrix[current][next];
            let probability = pheromone.powf(self.alpha) * heuristic.powf(self.beta);

            probabilities.push((next, probability));
            total_probability += probability;
        }

        // Roulette wheel selection
        let random_value = fastrand::f64() * total_probability;
        let mut cumulative = 0.0;

        for (constraint, probability) in probabilities {
            cumulative += probability;
            if cumulative >= random_value {
                return Ok(constraint);
            }
        }

        // Fallback to random selection
        let unvisited_vec: Vec<_> = unvisited.iter().cloned().collect();
        Ok(unvisited_vec[fastrand::usize(0..unvisited_vec.len())])
    }

    fn calculate_solution_cost(
        &self,
        solution: &[usize],
        constraints: &[PropertyConstraint],
    ) -> Result<f64> {
        let mut total_cost = 0.0;

        for &constraint_idx in solution {
            let constraint_cost = match constraints[constraint_idx].constraint_type().as_str() {
                "sh:pattern" => 3.5,
                "sh:sparql" => 4.0,
                "sh:class" => 2.5,
                _ => 1.0,
            };
            total_cost += constraint_cost;
        }

        Ok(total_cost)
    }

    fn update_pheromones(&mut self, solutions: &[(Vec<usize>, f64)]) -> Result<()> {
        // Evaporation
        for i in 0..self.pheromone_matrix.len() {
            for j in 0..self.pheromone_matrix[i].len() {
                self.pheromone_matrix[i][j] *= 1.0 - self.pheromone_evaporation_rate;
            }
        }

        // Deposit pheromones for good solutions
        for (solution, cost) in solutions {
            let pheromone_deposit = self.pheromone_deposit_strength / cost;

            for window in solution.windows(2) {
                let from = window[0];
                let to = window[1];
                self.pheromone_matrix[from][to] += pheromone_deposit;
            }
        }

        Ok(())
    }
}

/// Simplified Differential Evolution Optimizer
#[derive(Debug)]
pub struct DifferentialEvolutionOptimizer {
    population_size: usize,
    max_generations: usize,
    differential_weight: f64,
    crossover_probability: f64,
    population: Vec<DEIndividual>,
    best_individual: Option<DEIndividual>,
}

#[derive(Debug, Clone)]
pub struct DEIndividual {
    parameters: Vec<f64>,
    fitness: f64,
}

impl Default for DifferentialEvolutionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl DifferentialEvolutionOptimizer {
    pub fn new() -> Self {
        Self {
            population_size: 50,
            max_generations: 200,
            differential_weight: 0.8,
            crossover_probability: 0.9,
            population: Vec::new(),
            best_individual: None,
        }
    }

    /// Optimize an objective function over the given search space using Differential Evolution.
    pub async fn optimize(
        &mut self,
        objective: &impl OptimizationObjectiveFunction,
        search_space: &OptimizationSearchSpace,
    ) -> Result<SwarmResult> {
        // Initialise population
        self.population.clear();
        for _ in 0..self.population_size {
            let point = OptimizationPoint::random(search_space)?;
            let fitness = objective.evaluate(&point).await?;
            self.population.push(DEIndividual {
                parameters: vec![
                    point.execution_time_weight,
                    point.memory_usage_weight,
                    point.cache_efficiency_weight,
                ],
                fitness,
            });
        }

        for _generation in 0..self.max_generations {
            let pop_snapshot = self.population.clone();
            let n = pop_snapshot.len();
            for i in 0..n {
                // Pick three distinct random indices different from i
                let mut indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                let r1 = indices.remove(fastrand::usize(0..indices.len()));
                let r2 = indices.remove(fastrand::usize(0..indices.len()));
                let r3 = indices.remove(fastrand::usize(0..indices.len()));

                // Mutant vector: a + F*(b - c)
                let mutant: Vec<f64> = (0..3)
                    .map(|k| {
                        pop_snapshot[r1].parameters[k]
                            + self.differential_weight
                                * (pop_snapshot[r2].parameters[k] - pop_snapshot[r3].parameters[k])
                    })
                    .collect();

                // Crossover to produce trial vector
                let trial: Vec<f64> = mutant
                    .iter()
                    .enumerate()
                    .map(|(k, &m)| {
                        if fastrand::f64() < self.crossover_probability {
                            m
                        } else {
                            pop_snapshot[i].parameters[k]
                        }
                    })
                    .collect();

                // Clamp trial to search space
                let ranges = [
                    search_space.execution_time_range,
                    search_space.memory_usage_range,
                    search_space.cache_efficiency_range,
                ];
                let clamped: Vec<f64> = trial
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| v.clamp(ranges[k].0, ranges[k].1))
                    .collect();

                let trial_point = OptimizationPoint {
                    execution_time_weight: clamped[0],
                    memory_usage_weight: clamped[1],
                    cache_efficiency_weight: clamped[2],
                };
                let trial_fitness = objective.evaluate(&trial_point).await?;

                // Selection
                if trial_fitness >= self.population[i].fitness {
                    self.population[i] = DEIndividual {
                        parameters: clamped,
                        fitness: trial_fitness,
                    };
                }
            }
        }

        // Find best individual
        let best = self
            .population
            .iter()
            .max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| ShaclAiError::ShapeManagement("DE: empty population".to_string()))?
            .clone();

        self.best_individual = Some(best.clone());

        Ok(SwarmResult {
            optimization_id: uuid::Uuid::new_v4().to_string(),
            strategy_applied: "DifferentialEvolution".to_string(),
            before_metrics: PerformanceMetrics::default(),
            after_metrics: PerformanceMetrics::default(),
            improvement_percentage: best.fitness,
            optimization_time_ms: 0.0,
            applied_at: chrono::Utc::now(),
        })
    }
}

/// Simplified Tabu Search Optimizer
#[derive(Debug)]
pub struct TabuSearchOptimizer {
    max_iterations: usize,
    tabu_list_size: usize,
    neighborhood_size: usize,
    tabu_list: VecDeque<TabuMove>,
    current_solution: Option<OptimizationSolution>,
    best_solution: Option<OptimizationSolution>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabuMove {
    move_type: String,
    parameters: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct OptimizationSolution {
    pub constraint_configuration: ConstraintConfiguration,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct ConstraintConfiguration {
    pub constraint_order: Vec<usize>,
    pub parallelization_config: ParallelValidationConfig,
    pub cache_configuration: CacheConfiguration,
}

impl Default for TabuSearchOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl TabuSearchOptimizer {
    pub fn new() -> Self {
        Self {
            max_iterations: 1000,
            tabu_list_size: 50,
            neighborhood_size: 20,
            tabu_list: VecDeque::new(),
            current_solution: None,
            best_solution: None,
        }
    }
}

/// Simplified Reinforcement Learning Optimizer
#[derive(Debug)]
pub struct ReinforcementLearningOptimizer {
    q_table: HashMap<StateActionPair, f64>,
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64, // Exploration rate
    episode_count: usize,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StateActionPair {
    state: OptimizationState,
    action: OptimizationAction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationState {
    parallel_threads: usize,
    cache_size_mb: f64,
    constraint_order_entropy: f64,
}

impl Eq for OptimizationState {}

impl std::hash::Hash for OptimizationState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.parallel_threads.hash(state);
        // Hash f64 values by converting to bits
        self.cache_size_mb.to_bits().hash(state);
        self.constraint_order_entropy.to_bits().hash(state);
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum OptimizationAction {
    IncreaseParallelism,
    DecreaseParallelism,
    IncreaseCacheSize,
    DecreaseCacheSize,
    ReorderConstraints,
    NoAction,
}

impl Default for ReinforcementLearningOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ReinforcementLearningOptimizer {
    pub fn new() -> Self {
        Self {
            q_table: HashMap::new(),
            learning_rate: 0.1,
            discount_factor: 0.95,
            epsilon: 0.1,
            episode_count: 0,
        }
    }

    /// Optimize a policy starting from `initial_state` over `episodes` episodes.
    ///
    /// Uses the `EvoState` / `OptimizationAction` types from `opt_algs_evolutionary`
    /// and returns an `OptimizationPolicy` from the same module.
    pub async fn optimize(
        &mut self,
        initial_state: EvoState,
        episodes: usize,
    ) -> Result<OptimizationPolicy> {
        use crate::opt_algs_evolutionary::OptimizationAction;

        let mut state_action_mapping = HashMap::new();

        for _episode in 0..episodes {
            let mut current_state = initial_state.clone();

            for _step in 0..20 {
                // Epsilon-greedy action selection
                let action = if fastrand::f64() < self.epsilon {
                    OptimizationAction::random()
                } else {
                    OptimizationAction::from_id(
                        (current_state.parallel_threads + current_state.cache_size_mb as usize) % 6,
                    )
                };

                // Simulate state transition
                let next_state = EvoState {
                    parallel_threads: match action {
                        OptimizationAction::IncreaseParallelism => {
                            current_state.parallel_threads.saturating_add(1).min(16)
                        }
                        OptimizationAction::DecreaseParallelism => {
                            current_state.parallel_threads.saturating_sub(1).max(1)
                        }
                        _ => current_state.parallel_threads,
                    },
                    cache_size_mb: match action {
                        OptimizationAction::IncreaseCacheSize => {
                            current_state.cache_size_mb.saturating_add(16)
                        }
                        OptimizationAction::DecreaseCacheSize => {
                            current_state.cache_size_mb.saturating_sub(16).max(8)
                        }
                        _ => current_state.cache_size_mb,
                    },
                    constraint_order_entropy: current_state.constraint_order_entropy,
                };

                // Simple reward: increasing parallelism and cache is good
                let reward: f64 = match action {
                    OptimizationAction::IncreaseParallelism => 1.0,
                    OptimizationAction::IncreaseCacheSize => 0.8,
                    OptimizationAction::ReorderConstraints => 0.5,
                    OptimizationAction::NoAction => 0.0,
                    _ => -0.1,
                };

                // Record best action for this state (epsilon-greedy policy improvement)
                state_action_mapping
                    .entry(current_state.clone())
                    .and_modify(|existing: &mut OptimizationAction| {
                        if reward > 0.5 {
                            *existing = action.clone();
                        }
                    })
                    .or_insert_with(|| action.clone());

                self.episode_count += 1;
                current_state = next_state;
            }
        }

        // Confidence grows with more episodes, saturating at 1.0
        let confidence = (episodes as f64 / 100.0).min(1.0);

        Ok(OptimizationPolicy {
            state_action_mapping,
            confidence,
        })
    }
}

/// Simplified Adaptive Optimizer
#[derive(Debug)]
pub struct AdaptiveOptimizer {
    historical_optimizations: Vec<HistoricalOptimization>,
    performance_model: PerformanceModel,
    min_history_size: usize,
}

#[derive(Debug, Clone)]
pub struct HistoricalOptimization {
    problem: OptimizationProblem,
    strategy: OptimizationStrategy,
    result: AdaptiveOptimizationResult,
}

#[derive(Debug, Clone)]
pub struct OptimizationProblem {
    constraints: Vec<PropertyConstraint>,
    baseline_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    use_parallel_execution: bool,
    cache_strategy: CacheStrategyType,
    constraint_ordering: OrderingStrategyType,
    memory_optimization: bool,
}

#[derive(Debug, Clone)]
pub struct AdaptiveOptimizationResult {
    strategy_applied: OptimizationStrategy,
    baseline_performance: PerformanceMetrics,
    optimized_performance: PerformanceMetrics,
    performance_improvement: f64,
    optimization_duration: std::time::Duration,
}

#[derive(Debug)]
pub struct PerformanceModel {
    // Simplified model
    weights: Vec<f64>,
}

impl Default for AdaptiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveOptimizer {
    pub fn new() -> Self {
        Self {
            historical_optimizations: Vec::new(),
            performance_model: PerformanceModel {
                weights: vec![1.0; 10],
            },
            min_history_size: 10,
        }
    }

    /// Optimize the given problem using adaptive strategy selection.
    ///
    /// Takes an `OptimizationProblem` from `opt_algs_evolutionary` and returns an
    /// `AdaptiveOptimizationResult` from the same module.
    pub async fn optimize(&mut self, problem: &EvoProblem) -> Result<EvoAdaptiveResult> {
        use std::time::Instant;

        let start = Instant::now();
        let problem_features = problem.extract_features();

        // Build an adaptive model using a fresh predictor (historical data not wired here
        // because the local HistoricalOptimization type is incompatible with evo types).
        let model = AdaptivePerformanceModel::new();
        let strategy = EvoStrategy::default();
        let strategy_features = strategy.extract_features();
        let predicted_improvement = model.predict(&problem_features, &strategy_features)?;
        let confidence = model.confidence();

        // Simulate optimization using the chosen strategy
        let optimized_metrics = PerformanceMetrics {
            validation_time_ms: problem.baseline_metrics.validation_time_ms
                * (1.0 - predicted_improvement.min(0.5)),
            memory_usage_mb: problem.baseline_metrics.memory_usage_mb * 0.9,
            cpu_usage_percent: problem.baseline_metrics.cpu_usage_percent * 0.85,
            cache_hit_rate: (problem.baseline_metrics.cache_hit_rate + 0.1).min(1.0),
            parallelization_factor: problem.baseline_metrics.parallelization_factor
                * if strategy.use_parallel_execution {
                    1.5
                } else {
                    1.0
                },
            constraint_execution_times: HashMap::new(),
        };

        let performance_improvement = if problem.baseline_metrics.validation_time_ms > 0.0 {
            (problem.baseline_metrics.validation_time_ms - optimized_metrics.validation_time_ms)
                / problem.baseline_metrics.validation_time_ms
        } else {
            predicted_improvement
        };

        Ok(EvoAdaptiveResult {
            strategy_applied: strategy,
            baseline_performance: problem.baseline_metrics.clone(),
            optimized_performance: optimized_metrics,
            performance_improvement: performance_improvement.max(0.0),
            optimization_duration: start.elapsed(),
            confidence,
        })
    }
}

// Additional supporting types
use super::constraint_optimizer::OrderingStrategyType;

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self {
            use_parallel_execution: true,
            cache_strategy: CacheStrategyType::ResultCaching,
            constraint_ordering: OrderingStrategyType::CostBased,
            memory_optimization: false,
        }
    }
}

impl OptimizationProblem {
    pub fn extract_features(&self) -> Vec<f64> {
        vec![
            self.constraints.len() as f64,
            self.baseline_metrics.validation_time_ms,
            self.baseline_metrics.memory_usage_mb,
        ]
    }
}

impl OptimizationStrategy {
    pub fn extract_features(&self) -> Vec<f64> {
        vec![
            if self.use_parallel_execution {
                1.0
            } else {
                0.0
            },
            if self.memory_optimization { 1.0 } else { 0.0 },
        ]
    }
}

impl PerformanceModel {
    pub fn train(&mut self, _examples: &[PerformanceTrainingExample]) -> Result<()> {
        // Simplified training - just placeholder
        Ok(())
    }

    pub fn predict(&self, _problem_features: &[f64], _strategy_features: &[f64]) -> Result<f64> {
        // Simplified prediction
        Ok(0.15) // 15% improvement prediction
    }
}

#[derive(Debug)]
pub struct PerformanceTrainingExample {
    pub problem_features: Vec<f64>,
    pub strategy_features: Vec<f64>,
    pub performance_outcome: f64,
}

/// Bayesian optimization for hyperparameter tuning
#[derive(Debug)]
pub struct BayesianOptimizer {
    pub acquisition_function: String,
    pub surrogate_model: String,
    pub n_initial_points: usize,
    pub n_calls: usize,
    pub noise_level: f64,
}

impl Default for BayesianOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BayesianOptimizer {
    pub fn new() -> Self {
        Self {
            acquisition_function: "expected_improvement".to_string(),
            surrogate_model: "gaussian_process".to_string(),
            n_initial_points: 10,
            n_calls: 100,
            noise_level: 0.01,
        }
    }
}

/// Genetic algorithm optimizer for constraint evolution
#[derive(Debug)]
pub struct GeneticOptimizer {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub max_generations: usize,
    pub selection_method: String,
}

impl Default for GeneticOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneticOptimizer {
    pub fn new() -> Self {
        Self {
            population_size: 100,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            max_generations: 500,
            selection_method: "tournament".to_string(),
        }
    }
}

/// Multi-objective optimization for competing goals
#[derive(Debug)]
pub struct MultiObjectiveOptimizer {
    pub objectives: Vec<String>,
    pub weights: Vec<f64>,
    pub method: String,
    pub pareto_front_size: usize,
}

impl Default for MultiObjectiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiObjectiveOptimizer {
    pub fn new() -> Self {
        Self {
            objectives: vec!["performance".to_string(), "accuracy".to_string()],
            weights: vec![0.5, 0.5],
            method: "nsga2".to_string(),
            pareto_front_size: 50,
        }
    }
}

/// Particle swarm optimization for parameter tuning
#[derive(Debug)]
pub struct ParticleSwarmOptimizer {
    pub swarm_size: usize,
    pub max_iterations: usize,
    pub inertia_weight: f64,
    pub cognitive_coefficient: f64,
    pub social_coefficient: f64,
}

impl Default for ParticleSwarmOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ParticleSwarmOptimizer {
    pub fn new() -> Self {
        Self {
            swarm_size: 50,
            max_iterations: 1000,
            inertia_weight: 0.9,
            cognitive_coefficient: 2.0,
            social_coefficient: 2.0,
        }
    }
}

/// Simulated annealing for global optimization
#[derive(Debug)]
pub struct SimulatedAnnealingOptimizer {
    pub initial_temperature: f64,
    pub final_temperature: f64,
    pub cooling_rate: f64,
    pub max_iterations: usize,
    pub acceptance_probability: String,
}

impl Default for SimulatedAnnealingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulatedAnnealingOptimizer {
    pub fn new() -> Self {
        Self {
            initial_temperature: 1000.0,
            final_temperature: 0.1,
            cooling_rate: 0.95,
            max_iterations: 10000,
            acceptance_probability: "boltzmann".to_string(),
        }
    }
}
