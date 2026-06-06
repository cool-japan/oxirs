//! Evolutionary NAS — Evolution
//!
//! Evolutionary loop: selection (tournament/roulette), crossover, mutation, and population
//! management.

use crate::evolutionary_nas_types::{
    ArchitectureCandidate, ArchitectureGenome, ConnectionGene, ConvergenceMetrics,
    DiversityMetrics, EvolutionaryConfig, FitnessScores, GenerationStatistics, GlobalParameters,
    HardwareMetrics, HardwareTarget, InnovationTracker, NodeGene, OperationType,
    PerformanceMetrics,
};
use anyhow::{anyhow, Result};
use scirs2_core::random::{Random, RngExt};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

// ── Traits ────────────────────────────────────────────────────────────────────

/// Trait for crossover operators
pub trait CrossoverOperator: Send + Sync {
    fn crossover(
        &self,
        parent1: &ArchitectureCandidate,
        parent2: &ArchitectureCandidate,
        innovation_tracker: &mut InnovationTracker,
    ) -> Result<(ArchitectureCandidate, ArchitectureCandidate)>;
}

/// Trait for mutation operators
pub trait MutationOperator: Send + Sync {
    fn mutate(
        &self,
        candidate: &mut ArchitectureCandidate,
        innovation_tracker: &mut InnovationTracker,
        mutation_rate: f32,
    ) -> Result<()>;
}

/// Trait for selection operators
pub trait SelectionOperator: Send + Sync {
    fn select(&self, population: &[ArchitectureCandidate], selection_size: usize) -> Vec<usize>;
}

/// Trait for diversity maintenance strategies
pub trait DiversityStrategy: Send + Sync {
    fn maintain_diversity(
        &self,
        population: &mut Vec<ArchitectureCandidate>,
        diversity_target: f32,
    ) -> Result<()>;
}

/// Trait for hardware optimization strategies
pub trait HardwareOptimizationStrategy: Send + Sync {
    fn optimize_for_hardware(
        &self,
        genome: &mut ArchitectureGenome,
        target_hardware: &HardwareTarget,
    ) -> Result<crate::evolutionary_nas_types::OptimizationResult>;
}

/// Trait for hardware performance models
pub trait PerformanceModel: Send + Sync {
    fn predict_performance(
        &self,
        genome: &ArchitectureGenome,
        hardware: &HardwareTarget,
    ) -> Result<PerformanceMetrics>;
}

// ── Internal sub-structures ───────────────────────────────────────────────────

/// Evolution history tracking
pub(crate) struct EvolutionHistory {
    pub generation_stats: Vec<GenerationStatistics>,
    pub hall_of_fame: VecDeque<ArchitectureCandidate>,
    pub innovation_tracker: InnovationTracker,
    pub convergence_metrics: ConvergenceMetrics,
}

/// Genetic operators container
pub(crate) struct GeneticOperators {
    pub crossover_ops: Vec<Box<dyn CrossoverOperator>>,
    pub mutation_ops: Vec<Box<dyn MutationOperator>>,
    pub selection_ops: Vec<Box<dyn SelectionOperator>>,
}

/// Population diversity manager
pub(crate) struct DiversityManager {
    pub diversity_metrics: DiversityMetrics,
    pub novelty_archive: Vec<ArchitectureCandidate>,
    pub strategies: Vec<Box<dyn DiversityStrategy>>,
}

/// Hardware-aware optimizer
pub(crate) struct HardwareOptimizer {
    pub target_hardware: HardwareTarget,
    pub optimization_strategies: Vec<Box<dyn HardwareOptimizationStrategy>>,
    pub performance_models: HashMap<String, Box<dyn PerformanceModel>>,
}

// ── EvolutionaryNAS ───────────────────────────────────────────────────────────

/// Evolutionary Neural Architecture Search Engine
pub struct EvolutionaryNAS {
    pub(crate) config: EvolutionaryConfig,
    pub population: Arc<RwLock<Vec<ArchitectureCandidate>>>,
    pub(crate) evolution_history: EvolutionHistory,
    pub(crate) fitness_evaluator: crate::evolutionary_nas_eval::FitnessEvaluator,
    pub(crate) _genetic_operators: GeneticOperators,
    pub(crate) _diversity_manager: DiversityManager,
    pub(crate) _hardware_optimizer: HardwareOptimizer,
    pub(crate) performance_cache: Arc<RwLock<HashMap<String, PerformanceMetrics>>>,
}

impl EvolutionaryNAS {
    /// Create a new evolutionary NAS engine
    pub fn new(config: EvolutionaryConfig) -> Result<Self> {
        let population = Arc::new(RwLock::new(Vec::new()));
        let performance_cache = Arc::new(RwLock::new(HashMap::new()));

        let evolution_history = EvolutionHistory {
            generation_stats: Vec::new(),
            hall_of_fame: VecDeque::new(),
            innovation_tracker: InnovationTracker::new(),
            convergence_metrics: ConvergenceMetrics {
                improvement_rate: 0.0,
                stagnation_count: 0,
                diversity_trend: Vec::new(),
                convergence_probability: 0.0,
            },
        };

        let fitness_evaluator = crate::evolutionary_nas_eval::FitnessEvaluator {
            datasets: HashMap::new(),
            hardware_profiler: crate::evolutionary_nas_eval::HardwareProfiler {
                target_hardware: config.target_hardware.clone(),
                profiling_history: Vec::new(),
            },
            evaluation_cache: performance_cache.clone(),
        };

        let _genetic_operators = GeneticOperators {
            crossover_ops: Vec::new(),
            mutation_ops: Vec::new(),
            selection_ops: Vec::new(),
        };

        let _diversity_manager = DiversityManager {
            diversity_metrics: DiversityMetrics {
                genotypic_diversity: 0.0,
                phenotypic_diversity: 0.0,
                novelty_distribution: Vec::new(),
                population_entropy: 0.0,
            },
            novelty_archive: Vec::new(),
            strategies: Vec::new(),
        };

        let _hardware_optimizer = HardwareOptimizer {
            target_hardware: config.target_hardware.clone(),
            optimization_strategies: Vec::new(),
            performance_models: HashMap::new(),
        };

        Ok(Self {
            config,
            population,
            evolution_history,
            fitness_evaluator,
            _genetic_operators,
            _diversity_manager,
            _hardware_optimizer,
            performance_cache,
        })
    }

    /// Initialize the population with random architectures
    pub async fn initialize_population(&mut self) -> Result<()> {
        info!(
            "Initializing population with {} candidates",
            self.config.population_size
        );
        let mut candidates = Vec::with_capacity(self.config.population_size);
        for i in 0..self.config.population_size {
            candidates.push(self.generate_random_candidate(i)?);
        }
        let mut population = self.population.write().await;
        population.clear();
        population.extend(candidates);
        info!("Population initialized successfully");
        Ok(())
    }

    /// Generate a random architecture candidate
    pub(crate) fn generate_random_candidate(
        &mut self,
        _index: usize,
    ) -> Result<ArchitectureCandidate> {
        let mut random = Random::default();

        let base_complexity = self.config.progressive_config.start_complexity;
        let num_nodes = base_complexity + random.random_range(0..2usize);

        let mut nodes = Vec::new();
        let mut connections = Vec::new();

        for i in 0..num_nodes {
            let operation = self.generate_random_operation(&mut random)?;
            let node = NodeGene {
                id: i,
                operation,
                parameters: self.generate_random_parameters(&mut random),
                active: true,
                innovation_number: self
                    .evolution_history
                    .innovation_tracker
                    .get_innovation_number(&format!("node_{i}")),
            };
            nodes.push(node);
        }

        let num_connections = random.random_range(num_nodes..num_nodes * 2);
        for _ in 0..num_connections {
            if nodes.len() >= 2 {
                let from_node = random.random_range(0..nodes.len() - 1);
                let to_node = random.random_range(from_node + 1..nodes.len());
                let connection = ConnectionGene {
                    from_node,
                    to_node,
                    weight: random.random_range(-1.0f32..1.0f32),
                    active: true,
                    innovation_number: self
                        .evolution_history
                        .innovation_tracker
                        .get_innovation_number(&format!("conn_{from_node}_{to_node}")),
                };
                connections.push(connection);
            }
        }

        let genome = ArchitectureGenome {
            nodes,
            connections,
            global_params: GlobalParameters::default(),
            modules: Vec::new(),
        };

        Ok(ArchitectureCandidate {
            id: Uuid::new_v4(),
            genome,
            fitness: FitnessScores::default(),
            performance: None,
            generation: 0,
            parents: Vec::new(),
            novelty_score: 0.0,
            hardware_metrics: HardwareMetrics::default(),
        })
    }

    /// Generate a random operation type
    pub(crate) fn generate_random_operation(&self, random: &mut Random) -> Result<OperationType> {
        let operation_count: usize = 7;
        Ok(match random.random_range(0..operation_count) {
            0 => OperationType::Linear {
                input_dim: 128,
                output_dim: 128,
            },
            1 => OperationType::GraphConv {
                channels: 64,
                aggregation: "mean".to_string(),
            },
            2 => OperationType::Attention {
                heads: 8,
                embed_dim: 128,
            },
            3 => OperationType::Activation {
                function: "relu".to_string(),
            },
            4 => OperationType::Normalization {
                method: "batch_norm".to_string(),
            },
            5 => OperationType::Dropout { rate: 0.1 },
            _ => OperationType::SkipConnection,
        })
    }

    /// Generate random parameters for an operation
    pub(crate) fn generate_random_parameters(&self, random: &mut Random) -> HashMap<String, f32> {
        let mut params = HashMap::new();
        params.insert(
            "learning_rate".to_string(),
            random.random_range(0.0001f32..0.01f32),
        );
        params.insert(
            "dropout_rate".to_string(),
            random.random_range(0.0f32..0.5f32),
        );
        params.insert(
            "weight_decay".to_string(),
            random.random_range(0.0f32..0.01f32),
        );
        params
    }

    /// Run the evolutionary optimization process
    pub async fn evolve(&mut self) -> Result<ArchitectureCandidate> {
        info!(
            "Starting evolutionary optimization for {} generations",
            self.config.max_generations
        );

        if self.population.read().await.is_empty() {
            self.initialize_population().await?;
        }

        let mut best_candidate: Option<ArchitectureCandidate> = None;

        for generation in 0..self.config.max_generations {
            self.evaluate_population().await?;

            let gen_stats = self.calculate_generation_statistics(generation).await?;
            self.evolution_history.generation_stats.push(gen_stats);

            let current_best = self.get_best_candidate().await?;
            if best_candidate.is_none()
                || current_best.fitness.overall_fitness
                    > best_candidate
                        .as_ref()
                        .expect("best_candidate should be set")
                        .fitness
                        .overall_fitness
            {
                best_candidate = Some(current_best);
            }

            if self.check_convergence(generation).await? {
                info!("Convergence detected at generation {}", generation);
                break;
            }

            self.evolve_next_generation().await?;
            self.maintain_population_diversity().await?;

            if self.config.progressive_config.enable_modular_building {
                self.apply_progressive_complexification(generation).await?;
            }
        }

        info!("Evolution completed");
        best_candidate.ok_or_else(|| anyhow!("No best candidate found"))
    }

    async fn evaluate_population(&mut self) -> Result<()> {
        let mut population = self.population.write().await;
        for candidate in population.iter_mut() {
            let genome_hash = Self::calculate_genome_hash_static(&candidate.genome);
            if let Some(cached_performance) = self.performance_cache.read().await.get(&genome_hash)
            {
                candidate.performance = Some(cached_performance.clone());
            } else {
                let performance = self
                    .fitness_evaluator
                    .evaluate_candidate_performance(candidate)
                    .await?;
                candidate.performance = Some(performance.clone());
                self.performance_cache
                    .write()
                    .await
                    .insert(genome_hash, performance);
            }
            candidate.fitness = self.calculate_fitness_scores(candidate)?;
        }
        self.calculate_pareto_ranking(&mut population)?;
        Ok(())
    }

    fn calculate_genome_hash_static(genome: &ArchitectureGenome) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        genome.nodes.len().hash(&mut hasher);
        genome.connections.len().hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    pub(crate) fn calculate_fitness_scores(
        &self,
        candidate: &ArchitectureCandidate,
    ) -> Result<FitnessScores> {
        let performance = candidate
            .performance
            .as_ref()
            .ok_or_else(|| anyhow!("No performance metrics available"))?;

        let weights = &self.config.objective_weights;
        let accuracy = performance.validation_accuracy;
        let efficiency = 1.0 / (performance.inference_time_ms + 1.0);
        let memory_efficiency = 1.0 / (performance.memory_usage_mb / 1000.0 + 1.0);
        let simplicity = 1.0 / (candidate.genome.nodes.len() as f32 / 10.0 + 1.0);
        let novelty = candidate.novelty_score;
        let hardware_compatibility = candidate.hardware_metrics.efficiency_score;

        let overall_fitness = weights.accuracy_weight * accuracy
            + weights.efficiency_weight * efficiency
            + weights.memory_weight * memory_efficiency
            + weights.simplicity_weight * simplicity
            + weights.novelty_weight * novelty;

        Ok(FitnessScores {
            overall_fitness,
            accuracy,
            efficiency,
            memory_efficiency,
            simplicity,
            novelty,
            hardware_compatibility,
            pareto_rank: 0,
            crowding_distance: 0.0,
        })
    }

    fn calculate_pareto_ranking(&self, population: &mut [ArchitectureCandidate]) -> Result<()> {
        let n = population.len();
        let mut domination_count = vec![0usize; n];
        let mut dominated_solutions: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut current_front: Vec<usize> = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    if self.dominates(&population[i], &population[j]) {
                        dominated_solutions[i].push(j);
                    } else if self.dominates(&population[j], &population[i]) {
                        domination_count[i] += 1;
                    }
                }
            }
            if domination_count[i] == 0 {
                population[i].fitness.pareto_rank = 0;
                current_front.push(i);
            }
        }

        let mut front_number = 0usize;
        while !current_front.is_empty() {
            fronts.push(current_front.clone());
            let mut next_front = Vec::new();
            for &i in &current_front {
                for &j in &dominated_solutions[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        population[j].fitness.pareto_rank = front_number + 1;
                        next_front.push(j);
                    }
                }
            }
            front_number += 1;
            current_front = next_front;
        }

        for front in fronts {
            self.calculate_crowding_distance(population, &front)?;
        }
        Ok(())
    }

    fn dominates(&self, a: &ArchitectureCandidate, b: &ArchitectureCandidate) -> bool {
        let a_better = a.fitness.accuracy >= b.fitness.accuracy
            && a.fitness.efficiency >= b.fitness.efficiency
            && a.fitness.memory_efficiency >= b.fitness.memory_efficiency
            && a.fitness.simplicity >= b.fitness.simplicity;
        let a_strictly_better = a.fitness.accuracy > b.fitness.accuracy
            || a.fitness.efficiency > b.fitness.efficiency
            || a.fitness.memory_efficiency > b.fitness.memory_efficiency
            || a.fitness.simplicity > b.fitness.simplicity;
        a_better && a_strictly_better
    }

    fn calculate_crowding_distance(
        &self,
        population: &mut [ArchitectureCandidate],
        front: &[usize],
    ) -> Result<()> {
        let front_size = front.len();
        if front_size <= 2 {
            for &i in front {
                population[i].fitness.crowding_distance = f32::INFINITY;
            }
            return Ok(());
        }
        for &i in front {
            population[i].fitness.crowding_distance = 0.0;
        }
        let objectives = ["accuracy", "efficiency", "memory_efficiency", "simplicity"];
        for objective in objectives {
            let mut sorted_indices = front.to_vec();
            sorted_indices.sort_by(|&a, &b| {
                let va = self.get_objective_value(&population[a], objective);
                let vb = self.get_objective_value(&population[b], objective);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            });
            population[sorted_indices[0]].fitness.crowding_distance = f32::INFINITY;
            population[sorted_indices[front_size - 1]]
                .fitness
                .crowding_distance = f32::INFINITY;
            let obj_min = self.get_objective_value(&population[sorted_indices[0]], objective);
            let obj_max =
                self.get_objective_value(&population[sorted_indices[front_size - 1]], objective);
            let obj_range = obj_max - obj_min;
            if obj_range > 0.0 {
                for i in 1..front_size - 1 {
                    let next_obj =
                        self.get_objective_value(&population[sorted_indices[i + 1]], objective);
                    let prev_obj =
                        self.get_objective_value(&population[sorted_indices[i - 1]], objective);
                    population[sorted_indices[i]].fitness.crowding_distance +=
                        (next_obj - prev_obj) / obj_range;
                }
            }
        }
        Ok(())
    }

    fn get_objective_value(&self, candidate: &ArchitectureCandidate, objective: &str) -> f32 {
        match objective {
            "accuracy" => candidate.fitness.accuracy,
            "efficiency" => candidate.fitness.efficiency,
            "memory_efficiency" => candidate.fitness.memory_efficiency,
            "simplicity" => candidate.fitness.simplicity,
            _ => 0.0,
        }
    }

    async fn calculate_generation_statistics(
        &self,
        generation: usize,
    ) -> Result<GenerationStatistics> {
        let population = self.population.read().await;
        let fitness_values: Vec<f32> = population
            .iter()
            .map(|c| c.fitness.overall_fitness)
            .collect();
        let best_fitness = fitness_values.iter().fold(0.0f32, |a, &b| a.max(b));
        let average_fitness = fitness_values.iter().sum::<f32>() / fitness_values.len() as f32;
        let variance = fitness_values
            .iter()
            .map(|&f| (f - average_fitness).powi(2))
            .sum::<f32>()
            / fitness_values.len() as f32;
        let fitness_std = variance.sqrt();
        let diversity_score = self.calculate_population_diversity(&population)?;
        Ok(GenerationStatistics {
            generation,
            best_fitness,
            average_fitness,
            fitness_std,
            diversity_score,
            new_innovations: 0,
            timestamp: chrono::Utc::now(),
        })
    }

    pub(crate) fn calculate_population_diversity(
        &self,
        population: &[ArchitectureCandidate],
    ) -> Result<f32> {
        if population.len() < 2 {
            return Ok(0.0);
        }
        let mut total_distance = 0.0f32;
        let mut comparisons = 0usize;
        for i in 0..population.len() {
            for j in i + 1..population.len() {
                let distance =
                    self.calculate_genome_distance(&population[i].genome, &population[j].genome)?;
                total_distance += distance;
                comparisons += 1;
            }
        }
        Ok(total_distance / comparisons as f32)
    }

    pub(crate) fn calculate_genome_distance(
        &self,
        genome1: &ArchitectureGenome,
        genome2: &ArchitectureGenome,
    ) -> Result<f32> {
        let node_diff = (genome1.nodes.len() as f32 - genome2.nodes.len() as f32).abs();
        let conn_diff = (genome1.connections.len() as f32 - genome2.connections.len() as f32).abs();
        Ok((node_diff + conn_diff) / 10.0)
    }

    async fn get_best_candidate(&self) -> Result<ArchitectureCandidate> {
        let population = self.population.read().await;
        population
            .iter()
            .max_by(|a, b| {
                a.fitness
                    .overall_fitness
                    .partial_cmp(&b.fitness.overall_fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .ok_or_else(|| anyhow!("Empty population"))
    }

    async fn check_convergence(&self, generation: usize) -> Result<bool> {
        if generation < 10 {
            return Ok(false);
        }
        let recent_stats = &self.evolution_history.generation_stats;
        if recent_stats.len() < 10 {
            return Ok(false);
        }
        let recent_best: Vec<f32> = recent_stats
            .iter()
            .rev()
            .take(10)
            .map(|s| s.best_fitness)
            .collect();
        let improvement = recent_best[0] - recent_best[9];
        Ok(improvement < 0.001)
    }

    async fn evolve_next_generation(&mut self) -> Result<()> {
        // Clone the current population to release the Arc lock before calling &mut self methods
        let mut current_population: Vec<ArchitectureCandidate> =
            self.population.read().await.clone();
        let mut new_population = Vec::new();

        let elite_count = (current_population.len() as f32 * self.config.elite_percentage) as usize;
        current_population.sort_by(|a, b| {
            b.fitness
                .overall_fitness
                .partial_cmp(&a.fitness.overall_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        new_population.extend(current_population.iter().take(elite_count).cloned());

        while new_population.len() < self.config.population_size {
            let parent1_idx = self.tournament_selection(&current_population)?;
            let parent2_idx = self.tournament_selection(&current_population)?;
            let parent1 = current_population[parent1_idx].clone();
            let parent2 = current_population[parent2_idx].clone();

            let mut random = Random::default();
            if random.random::<f32>() < self.config.crossover_probability {
                let (mut child1, mut child2) = self.crossover(&parent1, &parent2)?;
                if random.random::<f32>() < self.config.mutation_probability {
                    self.mutate(&mut child1)?;
                }
                if random.random::<f32>() < self.config.mutation_probability {
                    self.mutate(&mut child2)?;
                }
                new_population.push(child1);
                if new_population.len() < self.config.population_size {
                    new_population.push(child2);
                }
            } else {
                let mut child = parent1.clone();
                child.id = Uuid::new_v4();
                child.parents = vec![parent1.id];
                if random.random::<f32>() < self.config.mutation_probability {
                    self.mutate(&mut child)?;
                }
                new_population.push(child);
            }
        }
        *self.population.write().await = new_population;
        Ok(())
    }

    /// Tournament selection
    pub(crate) fn tournament_selection(
        &self,
        population: &[ArchitectureCandidate],
    ) -> Result<usize> {
        let mut random = Random::default();
        let mut best_idx = random.random_range(0..population.len());
        let mut best_fitness = population[best_idx].fitness.overall_fitness;
        for _ in 1..self.config.tournament_size {
            let idx = random.random_range(0..population.len());
            if population[idx].fitness.overall_fitness > best_fitness {
                best_idx = idx;
                best_fitness = population[idx].fitness.overall_fitness;
            }
        }
        Ok(best_idx)
    }

    /// Crossover operation
    fn crossover(
        &mut self,
        parent1: &ArchitectureCandidate,
        parent2: &ArchitectureCandidate,
    ) -> Result<(ArchitectureCandidate, ArchitectureCandidate)> {
        let mut child1 = parent1.clone();
        let mut child2 = parent2.clone();
        child1.id = Uuid::new_v4();
        child2.id = Uuid::new_v4();
        child1.parents = vec![parent1.id, parent2.id];
        child2.parents = vec![parent1.id, parent2.id];

        let mut random = Random::default();
        let crossover_point =
            random.random_range(1..parent1.genome.nodes.len().min(parent2.genome.nodes.len()));
        for i in crossover_point..child1.genome.nodes.len().min(child2.genome.nodes.len()) {
            std::mem::swap(&mut child1.genome.nodes[i], &mut child2.genome.nodes[i]);
        }
        child1.fitness = FitnessScores::default();
        child2.fitness = FitnessScores::default();
        child1.performance = None;
        child2.performance = None;
        Ok((child1, child2))
    }

    /// Mutation operation
    fn mutate(&mut self, candidate: &mut ArchitectureCandidate) -> Result<()> {
        let mut random = Random::default();
        for node in &mut candidate.genome.nodes {
            if random.random::<f32>() < self.config.mutation_probability {
                for (_, value) in node.parameters.iter_mut() {
                    *value *= random.random_range(0.8f32..1.2f32);
                }
            }
        }
        for connection in &mut candidate.genome.connections {
            if random.random::<f32>() < self.config.mutation_probability {
                connection.weight += random.random_range(-0.1f32..0.1f32);
                connection.weight = connection.weight.clamp(-2.0, 2.0);
            }
        }
        if random.random::<f32>() < 0.05 {
            self.structural_mutation(candidate)?;
        }
        candidate.fitness = FitnessScores::default();
        candidate.performance = None;
        Ok(())
    }

    fn structural_mutation(&mut self, candidate: &mut ArchitectureCandidate) -> Result<()> {
        let mut random = Random::default();
        match random.random_range(0..4usize) {
            0 => self.add_node_mutation(candidate)?,
            1 => self.add_connection_mutation(candidate)?,
            2 => self.remove_node_mutation(candidate)?,
            3 => self.remove_connection_mutation(candidate)?,
            _ => {}
        }
        Ok(())
    }

    fn add_node_mutation(&mut self, candidate: &mut ArchitectureCandidate) -> Result<()> {
        let mut random = Random::default();
        let new_id = candidate.genome.nodes.len();
        let operation = self.generate_random_operation(&mut random)?;
        let node = NodeGene {
            id: new_id,
            operation,
            parameters: self.generate_random_parameters(&mut random),
            active: true,
            innovation_number: self
                .evolution_history
                .innovation_tracker
                .get_innovation_number(&format!("node_{new_id}")),
        };
        candidate.genome.nodes.push(node);
        Ok(())
    }

    fn add_connection_mutation(&mut self, candidate: &mut ArchitectureCandidate) -> Result<()> {
        let mut random = Random::default();
        let num_nodes = candidate.genome.nodes.len();
        if num_nodes >= 2 {
            let from_node = random.random_range(0..num_nodes);
            let to_node = random.random_range(0..num_nodes);
            if from_node != to_node {
                let connection = ConnectionGene {
                    from_node,
                    to_node,
                    weight: random.random_range(-1.0f32..1.0f32),
                    active: true,
                    innovation_number: self
                        .evolution_history
                        .innovation_tracker
                        .get_innovation_number(&format!("conn_{from_node}_{to_node}")),
                };
                candidate.genome.connections.push(connection);
            }
        }
        Ok(())
    }

    fn remove_node_mutation(&mut self, candidate: &mut ArchitectureCandidate) -> Result<()> {
        if candidate.genome.nodes.len() > 3 {
            let mut random = Random::default();
            let remove_idx = random.random_range(0..candidate.genome.nodes.len());
            candidate.genome.nodes.remove(remove_idx);
            candidate
                .genome
                .connections
                .retain(|conn| conn.from_node != remove_idx && conn.to_node != remove_idx);
        }
        Ok(())
    }

    fn remove_connection_mutation(&mut self, candidate: &mut ArchitectureCandidate) -> Result<()> {
        if !candidate.genome.connections.is_empty() {
            let mut random = Random::default();
            let remove_idx = random.random_range(0..candidate.genome.connections.len());
            candidate.genome.connections.remove(remove_idx);
        }
        Ok(())
    }

    async fn maintain_population_diversity(&mut self) -> Result<()> {
        let mut population = self.population.write().await;
        let novelty_scores: Vec<f32> = population
            .iter()
            .map(|candidate| {
                self.calculate_novelty_score(candidate, &population)
                    .unwrap_or(0.0)
            })
            .collect();
        for (candidate, score) in population.iter_mut().zip(novelty_scores) {
            candidate.novelty_score = score;
        }

        let mut to_remove = Vec::new();
        for i in 0..population.len() {
            for j in i + 1..population.len() {
                let distance =
                    self.calculate_genome_distance(&population[i].genome, &population[j].genome)?;
                if distance < 0.1 {
                    if population[i].fitness.overall_fitness < population[j].fitness.overall_fitness
                    {
                        to_remove.push(i);
                    } else {
                        to_remove.push(j);
                    }
                }
            }
        }
        to_remove.sort();
        to_remove.dedup();
        to_remove.reverse();
        for idx in to_remove {
            if population.len() > self.config.population_size / 2 {
                population.remove(idx);
            }
        }
        Ok(())
    }

    fn calculate_novelty_score(
        &self,
        candidate: &ArchitectureCandidate,
        population: &[ArchitectureCandidate],
    ) -> Result<f32> {
        let k = 15;
        let mut distances = Vec::new();
        for other in population {
            if other.id != candidate.id {
                let distance = self.calculate_genome_distance(&candidate.genome, &other.genome)?;
                distances.push(distance);
            }
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let novelty = if distances.len() >= k {
            distances.iter().take(k).sum::<f32>() / k as f32
        } else {
            distances.iter().sum::<f32>() / distances.len().max(1) as f32
        };
        Ok(novelty)
    }

    async fn apply_progressive_complexification(&mut self, generation: usize) -> Result<()> {
        let complexity_increase =
            self.config.progressive_config.complexity_increase_rate * generation as f32;
        let max_nodes =
            (self.config.progressive_config.start_complexity as f32 + complexity_increase) as usize;
        let max_nodes = max_nodes.min(self.config.progressive_config.max_complexity);

        // Clone population so the Arc<RwLock> is not held while we call &mut self methods
        let mut snapshot = self.population.read().await.clone();
        for candidate in snapshot.iter_mut() {
            let mut random = Random::default();
            if candidate.genome.nodes.len() < max_nodes && random.random::<f32>() < 0.1 {
                self.add_node_mutation(candidate)?;
            }
        }
        *self.population.write().await = snapshot;
        Ok(())
    }

    /// Get evolution statistics
    pub fn get_evolution_statistics(&self) -> &[GenerationStatistics] {
        &self.evolution_history.generation_stats
    }

    /// Export the best architectures
    pub async fn export_best_architectures(
        &self,
        count: usize,
    ) -> Result<Vec<ArchitectureCandidate>> {
        let mut population = self.population.read().await.clone();
        population.sort_by(|a, b| {
            b.fitness
                .overall_fitness
                .partial_cmp(&a.fitness.overall_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(population.into_iter().take(count).collect())
    }
}
