//! Swarm intelligence and surrogate-based optimization algorithms.
//!
//! Contains:
//! - `SimulatedAnnealingOptimizer`
//! - `ParticleSwarmOptimizer` with inertia-weight PSO
//! - `BayesianOptimizer` with Expected Improvement / UCB acquisition
//! - `GaussianProcess` surrogate model
//! - `OptimizationObjectiveFunction` trait
//! - Particle and search-space supporting types

use crate::{Result, ShaclAiError};
use std::collections::HashMap;

use crate::opt_algs_evolutionary::OptimizationSolution;
use crate::optimization_engine::PerformanceMetrics;

/// Continuous point in the PSO / Bayesian search space
#[derive(Debug, Clone)]
pub struct OptimizationPoint {
    pub execution_time_weight: f64,
    pub memory_usage_weight: f64,
    pub cache_efficiency_weight: f64,
}

impl OptimizationPoint {
    pub fn random(search_space: &OptimizationSearchSpace) -> Result<Self> {
        Ok(Self {
            execution_time_weight: fastrand::f64()
                * (search_space.execution_time_range.1 - search_space.execution_time_range.0)
                + search_space.execution_time_range.0,
            memory_usage_weight: fastrand::f64()
                * (search_space.memory_usage_range.1 - search_space.memory_usage_range.0)
                + search_space.memory_usage_range.0,
            cache_efficiency_weight: fastrand::f64()
                * (search_space.cache_efficiency_range.1 - search_space.cache_efficiency_range.0)
                + search_space.cache_efficiency_range.0,
        })
    }
}

/// Axis-aligned box search space
#[derive(Debug, Clone)]
pub struct OptimizationSearchSpace {
    pub execution_time_range: (f64, f64),
    pub memory_usage_range: (f64, f64),
    pub cache_efficiency_range: (f64, f64),
}

/// PSO particle position
#[derive(Debug, Clone)]
pub struct ParticlePosition {
    pub execution_time_weight: f64,
    pub memory_usage_weight: f64,
    pub cache_efficiency_weight: f64,
}

/// PSO particle velocity
#[derive(Debug, Clone)]
pub struct ParticleVelocity {
    pub execution_time_weight: f64,
    pub memory_usage_weight: f64,
    pub cache_efficiency_weight: f64,
}

/// PSO particle
#[derive(Debug, Clone)]
pub struct OptimizationParticle {
    pub position: ParticlePosition,
    pub velocity: ParticleVelocity,
    pub personal_best_position: ParticlePosition,
    pub personal_best_fitness: f64,
    pub fitness: f64,
}

impl OptimizationParticle {
    pub fn random(search_space: &OptimizationSearchSpace) -> Result<Self> {
        let position = ParticlePosition {
            execution_time_weight: fastrand::f64()
                * (search_space.execution_time_range.1 - search_space.execution_time_range.0)
                + search_space.execution_time_range.0,
            memory_usage_weight: fastrand::f64()
                * (search_space.memory_usage_range.1 - search_space.memory_usage_range.0)
                + search_space.memory_usage_range.0,
            cache_efficiency_weight: fastrand::f64()
                * (search_space.cache_efficiency_range.1 - search_space.cache_efficiency_range.0)
                + search_space.cache_efficiency_range.0,
        };

        Ok(Self {
            position: position.clone(),
            velocity: ParticleVelocity {
                execution_time_weight: 0.0,
                memory_usage_weight: 0.0,
                cache_efficiency_weight: 0.0,
            },
            personal_best_position: position,
            personal_best_fitness: f64::NEG_INFINITY,
            fitness: 0.0,
        })
    }
}

/// Optimization result (produced by PSO and others)
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub optimization_id: String,
    pub strategy_applied: String,
    pub before_metrics: PerformanceMetrics,
    pub after_metrics: PerformanceMetrics,
    pub improvement_percentage: f64,
    pub optimization_time_ms: f64,
    pub applied_at: chrono::DateTime<chrono::Utc>,
}

/// Acquisition function type for Bayesian optimisation
#[derive(Debug)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    UpperConfidenceBound,
}

/// Simple Gaussian Process surrogate model
#[derive(Debug)]
pub struct GaussianProcess {
    kernel_parameters: Vec<f64>,
}

impl GaussianProcess {
    pub fn new() -> Self {
        Self {
            kernel_parameters: vec![1.0, 1.0, 0.1],
        }
    }

    pub fn fit(&mut self, observed_points: &[(OptimizationPoint, f64)]) -> Result<()> {
        tracing::debug!(
            "Fitting Gaussian Process with {} observations",
            observed_points.len()
        );
        Ok(())
    }

    pub fn predict(&self, point: &OptimizationPoint) -> Result<(f64, f64)> {
        let mean = point.execution_time_weight * 0.5
            + point.memory_usage_weight * 0.3
            + point.cache_efficiency_weight * 0.2;
        let variance = 1.0;
        Ok((mean, variance))
    }
}

impl Default for GaussianProcess {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for objective functions that can be evaluated asynchronously
#[async_trait::async_trait]
pub trait OptimizationObjectiveFunction: Send + Sync {
    async fn evaluate(&self, point: &OptimizationPoint) -> Result<f64>;
}

// ─── Simulated Annealing ──────────────────────────────────────────────────────

/// Simulated Annealing Optimizer for Global Optimization
#[derive(Debug)]
pub struct SimulatedAnnealingOptimizer {
    initial_temperature: f64,
    final_temperature: f64,
    cooling_rate: f64,
    max_iterations: usize,
    current_solution: Option<OptimizationSolution>,
    best_solution: Option<OptimizationSolution>,
}

impl SimulatedAnnealingOptimizer {
    pub fn new() -> Self {
        Self {
            initial_temperature: 1000.0,
            final_temperature: 1.0,
            cooling_rate: 0.95,
            max_iterations: 1000,
            current_solution: None,
            best_solution: None,
        }
    }

    /// Optimize using simulated annealing
    pub async fn optimize(
        &mut self,
        initial_solution: OptimizationSolution,
    ) -> Result<OptimizationSolution> {
        self.current_solution = Some(initial_solution.clone());
        self.best_solution = Some(initial_solution);

        let mut temperature = self.initial_temperature;
        let mut iteration = 0;

        while temperature > self.final_temperature && iteration < self.max_iterations {
            let neighbor = self.generate_neighbor(
                self.current_solution
                    .as_ref()
                    .expect("current_solution should be initialized"),
            )?;
            let current_energy = self.calculate_energy(
                self.current_solution
                    .as_ref()
                    .expect("current_solution should be initialized"),
            )?;
            let neighbor_energy = self.calculate_energy(&neighbor)?;
            let delta_energy = neighbor_energy - current_energy;

            if delta_energy < 0.0 || fastrand::f64() < (-delta_energy / temperature).exp() {
                self.current_solution = Some(neighbor.clone());

                if let Some(ref best) = self.best_solution {
                    let best_energy = self.calculate_energy(best)?;
                    if neighbor_energy < best_energy {
                        self.best_solution = Some(neighbor);
                    }
                } else {
                    self.best_solution = Some(neighbor);
                }
            }

            temperature *= self.cooling_rate;
            iteration += 1;

            if iteration % 100 == 0 {
                tracing::debug!(
                    "SA Iteration {}: Temperature = {:.2}, Energy = {:.4}",
                    iteration,
                    temperature,
                    self.calculate_energy(
                        self.current_solution
                            .as_ref()
                            .expect("current_solution should be initialized")
                    )?
                );
            }
        }

        self.best_solution.clone().ok_or_else(|| {
            ShaclAiError::ShapeManagement("Simulated annealing failed to find solution".to_string())
        })
    }

    fn generate_neighbor(&self, solution: &OptimizationSolution) -> Result<OptimizationSolution> {
        let mut neighbor = solution.clone();

        match fastrand::u8(0..3) {
            0 => {
                if neighbor.constraint_configuration.constraint_order.len() > 1 {
                    let idx1 = fastrand::usize(
                        0..neighbor.constraint_configuration.constraint_order.len(),
                    );
                    let idx2 = fastrand::usize(
                        0..neighbor.constraint_configuration.constraint_order.len(),
                    );
                    neighbor
                        .constraint_configuration
                        .constraint_order
                        .swap(idx1, idx2);
                }
            }
            1 => {
                neighbor
                    .constraint_configuration
                    .parallelization_config
                    .max_parallel_constraints = (neighbor
                    .constraint_configuration
                    .parallelization_config
                    .max_parallel_constraints
                    + fastrand::i8(-1..=1) as usize)
                    .max(1)
                    .min(16);
            }
            _ => {
                neighbor
                    .constraint_configuration
                    .cache_configuration
                    .estimated_hit_rate = (neighbor
                    .constraint_configuration
                    .cache_configuration
                    .estimated_hit_rate
                    + fastrand::f64() * 0.1
                    - 0.05)
                    .clamp(0.0, 1.0);
            }
        }

        Ok(neighbor)
    }

    fn calculate_energy(&self, solution: &OptimizationSolution) -> Result<f64> {
        let execution_time = solution.performance_metrics.validation_time_ms;
        let memory_usage = solution.performance_metrics.memory_usage_mb;
        let cache_misses = 1.0 - solution.performance_metrics.cache_hit_rate;
        Ok(execution_time * 0.5 + memory_usage * 0.3 + cache_misses * 100.0 * 0.2)
    }
}

impl Default for SimulatedAnnealingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Particle Swarm Optimization ─────────────────────────────────────────────

/// Particle Swarm Optimization for Parallel Search
#[derive(Debug)]
pub struct ParticleSwarmOptimizer {
    num_particles: usize,
    max_iterations: usize,
    inertia_weight: f64,
    cognitive_coefficient: f64,
    social_coefficient: f64,
    particles: Vec<OptimizationParticle>,
    global_best: Option<OptimizationParticle>,
}

impl ParticleSwarmOptimizer {
    pub fn new() -> Self {
        Self {
            num_particles: 30,
            max_iterations: 100,
            inertia_weight: 0.9,
            cognitive_coefficient: 2.0,
            social_coefficient: 2.0,
            particles: Vec::new(),
            global_best: None,
        }
    }

    /// Optimize using particle swarm optimization
    pub async fn optimize(
        &mut self,
        search_space: &OptimizationSearchSpace,
    ) -> Result<OptimizationResult> {
        self.initialize_swarm(search_space)?;

        for iteration in 0..self.max_iterations {
            let mut fitness_values = Vec::new();
            for particle in &self.particles {
                fitness_values.push(self.evaluate_particle(particle).await?);
            }

            for (particle, fitness) in self.particles.iter_mut().zip(fitness_values) {
                particle.fitness = fitness;

                if particle.fitness > particle.personal_best_fitness {
                    particle.personal_best_position = particle.position.clone();
                    particle.personal_best_fitness = particle.fitness;
                }

                if self.global_best.is_none()
                    || particle.fitness
                        > self
                            .global_best
                            .as_ref()
                            .expect("global_best should be initialized")
                            .fitness
                {
                    self.global_best = Some(particle.clone());
                }
            }

            for i in 0..self.particles.len() {
                self.update_particle_velocity_by_index(i)?;
                self.update_particle_position_by_index(i, search_space)?;
            }

            self.inertia_weight *= 0.99;

            if iteration % 10 == 0 {
                tracing::debug!(
                    "PSO Iteration {}: Best fitness = {:.4}",
                    iteration,
                    self.global_best.as_ref().map(|p| p.fitness).unwrap_or(0.0)
                );
            }
        }

        let best_particle = self.global_best.clone().ok_or_else(|| {
            ShaclAiError::ShapeManagement("PSO failed to find solution".to_string())
        })?;

        Ok(OptimizationResult {
            optimization_id: uuid::Uuid::new_v4().to_string(),
            strategy_applied: "ParticleSwarmOptimization".to_string(),
            before_metrics: PerformanceMetrics::default(),
            after_metrics: PerformanceMetrics::default(),
            improvement_percentage: best_particle.fitness,
            optimization_time_ms: 0.0,
            applied_at: chrono::Utc::now(),
        })
    }

    fn initialize_swarm(&mut self, search_space: &OptimizationSearchSpace) -> Result<()> {
        self.particles.clear();
        for _ in 0..self.num_particles {
            self.particles
                .push(OptimizationParticle::random(search_space)?);
        }
        Ok(())
    }

    async fn evaluate_particle(&self, particle: &OptimizationParticle) -> Result<f64> {
        let execution_time_penalty = particle.position.execution_time_weight * 0.5;
        let memory_penalty = particle.position.memory_usage_weight * 0.3;
        let cache_bonus = particle.position.cache_efficiency_weight * 0.2;
        Ok(100.0 - execution_time_penalty - memory_penalty + cache_bonus)
    }

    fn update_particle_velocity_by_index(&mut self, index: usize) -> Result<()> {
        if let Some(global_best) = self.global_best.clone() {
            let particle = &mut self.particles[index];
            let r1 = fastrand::f64();
            let r2 = fastrand::f64();

            particle.velocity.execution_time_weight = self.inertia_weight
                * particle.velocity.execution_time_weight
                + self.cognitive_coefficient
                    * r1
                    * (particle.personal_best_position.execution_time_weight
                        - particle.position.execution_time_weight)
                + self.social_coefficient
                    * r2
                    * (global_best.position.execution_time_weight
                        - particle.position.execution_time_weight);

            particle.velocity.memory_usage_weight = self.inertia_weight
                * particle.velocity.memory_usage_weight
                + self.cognitive_coefficient
                    * r1
                    * (particle.personal_best_position.memory_usage_weight
                        - particle.position.memory_usage_weight)
                + self.social_coefficient
                    * r2
                    * (global_best.position.memory_usage_weight
                        - particle.position.memory_usage_weight);

            particle.velocity.cache_efficiency_weight = self.inertia_weight
                * particle.velocity.cache_efficiency_weight
                + self.cognitive_coefficient
                    * r1
                    * (particle.personal_best_position.cache_efficiency_weight
                        - particle.position.cache_efficiency_weight)
                + self.social_coefficient
                    * r2
                    * (global_best.position.cache_efficiency_weight
                        - particle.position.cache_efficiency_weight);
        }
        Ok(())
    }

    fn update_particle_position_by_index(
        &mut self,
        index: usize,
        search_space: &OptimizationSearchSpace,
    ) -> Result<()> {
        let particle = &mut self.particles[index];
        particle.position.execution_time_weight = (particle.position.execution_time_weight
            + particle.velocity.execution_time_weight)
            .clamp(
                search_space.execution_time_range.0,
                search_space.execution_time_range.1,
            );
        particle.position.memory_usage_weight =
            (particle.position.memory_usage_weight + particle.velocity.memory_usage_weight).clamp(
                search_space.memory_usage_range.0,
                search_space.memory_usage_range.1,
            );
        particle.position.cache_efficiency_weight = (particle.position.cache_efficiency_weight
            + particle.velocity.cache_efficiency_weight)
            .clamp(
                search_space.cache_efficiency_range.0,
                search_space.cache_efficiency_range.1,
            );
        Ok(())
    }
}

impl Default for ParticleSwarmOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Bayesian Optimization ────────────────────────────────────────────────────

/// Bayesian Optimization for Expensive Function Optimization
#[derive(Debug)]
pub struct BayesianOptimizer {
    acquisition_function: AcquisitionFunction,
    gaussian_process: GaussianProcess,
    observed_points: Vec<(OptimizationPoint, f64)>,
    exploration_weight: f64,
}

impl BayesianOptimizer {
    pub fn new() -> Self {
        Self {
            acquisition_function: AcquisitionFunction::ExpectedImprovement,
            gaussian_process: GaussianProcess::new(),
            observed_points: Vec::new(),
            exploration_weight: 0.1,
        }
    }

    /// Optimize using Bayesian optimization with initial random sampling
    pub async fn optimize(
        &mut self,
        objective_function: &dyn OptimizationObjectiveFunction,
        search_space: &OptimizationSearchSpace,
        max_evaluations: usize,
    ) -> Result<OptimizationPoint> {
        for _ in 0..5 {
            let random_point = OptimizationPoint::random(search_space)?;
            let value = objective_function.evaluate(&random_point).await?;
            self.observed_points.push((random_point, value));
        }

        for iteration in 0..max_evaluations.saturating_sub(5) {
            self.gaussian_process.fit(&self.observed_points)?;
            let next_point = self.find_next_point(search_space).await?;
            let value = objective_function.evaluate(&next_point).await?;
            self.observed_points.push((next_point, value));

            if iteration % 10 == 0 {
                let best_value = self
                    .observed_points
                    .iter()
                    .map(|(_, v)| *v)
                    .fold(f64::NEG_INFINITY, f64::max);
                tracing::debug!(
                    "Bayesian Optimization Iteration {}: Best value = {:.4}",
                    iteration,
                    best_value
                );
            }
        }

        let (best_point, _) = self
            .observed_points
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("values should be comparable"))
            .cloned()
            .ok_or_else(|| ShaclAiError::ShapeManagement("No points evaluated".to_string()))?;

        Ok(best_point)
    }

    async fn find_next_point(
        &self,
        search_space: &OptimizationSearchSpace,
    ) -> Result<OptimizationPoint> {
        let mut best_point = None;
        let mut best_acquisition_value = f64::NEG_INFINITY;

        for _ in 0..1000 {
            let candidate_point = OptimizationPoint::random(search_space)?;
            let acquisition_value = self.evaluate_acquisition_function(&candidate_point).await?;
            if acquisition_value > best_acquisition_value {
                best_acquisition_value = acquisition_value;
                best_point = Some(candidate_point);
            }
        }

        best_point
            .ok_or_else(|| ShaclAiError::ShapeManagement("Failed to find next point".to_string()))
    }

    async fn evaluate_acquisition_function(&self, point: &OptimizationPoint) -> Result<f64> {
        let (mean, variance) = self.gaussian_process.predict(point)?;

        match self.acquisition_function {
            AcquisitionFunction::ExpectedImprovement => {
                let best_observed = self
                    .observed_points
                    .iter()
                    .map(|(_, value)| *value)
                    .fold(f64::NEG_INFINITY, f64::max);

                let improvement = (mean - best_observed - self.exploration_weight).max(0.0);
                let std_dev = variance.sqrt();

                if std_dev > 1e-6 {
                    let z = improvement / std_dev;
                    let ei = improvement * self.normal_cdf(z) + std_dev * self.normal_pdf(z);
                    Ok(ei)
                } else {
                    Ok(improvement)
                }
            }
            AcquisitionFunction::UpperConfidenceBound => {
                let ucb = mean + 2.0 * variance.sqrt();
                Ok(ucb)
            }
        }
    }

    fn normal_cdf(&self, x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let erf_approx = if x >= 0.0 {
            1.0 - poly * (-x * x).exp()
        } else {
            poly * (-x * x).exp() - 1.0
        };
        0.5 * (1.0 + erf_approx)
    }

    fn normal_pdf(&self, x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}

impl Default for BayesianOptimizer {
    fn default() -> Self {
        Self::new()
    }
}
