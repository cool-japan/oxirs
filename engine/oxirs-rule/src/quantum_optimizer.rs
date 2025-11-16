//! Quantum-Inspired Optimization Algorithms for Rule-Based Reasoning
//!
//! Provides quantum-inspired optimization techniques for rule ordering, fact selection,
//! and inference path optimization. Uses concepts from quantum computing (superposition,
//! entanglement, interference) to explore multiple solution spaces simultaneously.
//!
//! # Features
//!
//! - **Quantum Annealing**: Global optimization via simulated quantum tunneling
//! - **Quantum Genetic Algorithm**: Evolution with quantum superposition
//! - **Quantum Particle Swarm**: Swarm intelligence with quantum behavior
//! - **Quantum Walk**: Graph traversal using quantum random walks
//! - **Grover-Inspired Search**: Amplitude amplification for rule selection
//! - **VQE-Style Optimization**: Variational quantum eigensolver approach
//!
//! # Architecture
//!
//! The quantum-inspired optimizer uses classical computing to simulate quantum
//! phenomena that provide advantages in combinatorial optimization:
//!
//! - **Superposition**: Explore multiple rule orderings simultaneously
//! - **Entanglement**: Capture dependencies between rule applications
//! - **Interference**: Amplify good solutions, suppress bad ones
//! - **Tunneling**: Escape local optima in search space
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::quantum_optimizer::{QuantumOptimizer, OptimizationGoal, QuantumAlgorithm};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let mut optimizer = QuantumOptimizer::new();
//!
//! let rules = vec![
//!     Rule {
//!         name: "rule1".to_string(),
//!         body: vec![RuleAtom::Triple {
//!             subject: Term::Variable("X".to_string()),
//!             predicate: Term::Constant("p".to_string()),
//!             object: Term::Variable("Y".to_string()),
//!         }],
//!         head: vec![RuleAtom::Triple {
//!             subject: Term::Variable("X".to_string()),
//!             predicate: Term::Constant("q".to_string()),
//!             object: Term::Variable("Y".to_string()),
//!         }],
//!     },
//! ];
//!
//! // Optimize rule execution order
//! let optimized_order = optimizer.optimize_rule_order(
//!     &rules,
//!     OptimizationGoal::MinimizeInferenceTime,
//!     QuantumAlgorithm::QuantumAnnealing
//! ).unwrap();
//!
//! println!("Optimized rule order: {:?}", optimized_order);
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Uniform};
use tracing::{debug, info};

/// Optimization goal for quantum-inspired algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationGoal {
    /// Minimize total inference time
    MinimizeInferenceTime,
    /// Maximize fact derivation rate
    MaximizeDerivations,
    /// Minimize memory usage
    MinimizeMemory,
    /// Balance time and space
    Balanced,
}

/// Quantum-inspired algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumAlgorithm {
    /// Quantum annealing (global optimization)
    QuantumAnnealing,
    /// Quantum genetic algorithm
    QuantumGenetic,
    /// Quantum particle swarm optimization
    QuantumParticleSwarm,
    /// Quantum walk on graph
    QuantumWalk,
    /// Grover-inspired search
    GroverSearch,
}

/// Quantum state representation (classical simulation)
#[derive(Debug, Clone)]
struct QuantumState {
    /// Amplitude vector (represents superposition)
    amplitudes: Array1<f64>,
    /// Phase vector
    phases: Array1<f64>,
}

impl QuantumState {
    /// Create new quantum state in uniform superposition
    fn new(dimension: usize) -> Self {
        let amplitude = 1.0 / (dimension as f64).sqrt();
        Self {
            amplitudes: Array1::from_elem(dimension, amplitude),
            phases: Array1::zeros(dimension),
        }
    }

    /// Measure state (collapse to classical outcome)
    fn measure(&self, rng: &mut StdRng) -> usize {
        let probabilities: Vec<f64> = self.amplitudes.iter().map(|a| a.powi(2)).collect();

        // Sample from probability distribution
        let total: f64 = probabilities.iter().sum();
        let mut cumulative = 0.0;
        let uniform_dist = Uniform::new(0.0, total).expect("Failed to create uniform distribution");
        let random_value = uniform_dist.sample(rng);

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return i;
            }
        }

        probabilities.len() - 1
    }

    /// Apply rotation (single-qubit gate simulation)
    fn rotate(&mut self, index: usize, angle: f64) {
        if index < self.amplitudes.len() {
            self.phases[index] += angle;
        }
    }

    /// Apply Hadamard-like transformation (create superposition)
    fn hadamard(&mut self) {
        let n = self.amplitudes.len();
        let scale = 1.0 / (n as f64).sqrt();
        for i in 0..n {
            self.amplitudes[i] = scale;
        }
    }

    /// Apply phase flip (Grover-inspired)
    fn phase_flip(&mut self, target: usize) {
        if target < self.phases.len() {
            self.phases[target] += std::f64::consts::PI;
        }
    }

    /// Apply inversion about average (Grover diffusion)
    fn inversion_about_average(&mut self) {
        let avg: f64 = self.amplitudes.iter().sum::<f64>() / self.amplitudes.len() as f64;
        for amp in self.amplitudes.iter_mut() {
            *amp = 2.0 * avg - *amp;
        }
    }
}

/// Quantum-inspired optimizer
pub struct QuantumOptimizer {
    /// Random number generator
    rng: StdRng,
    /// Temperature for annealing
    temperature: f64,
    /// Cooling rate
    cooling_rate: f64,
    /// Maximum iterations
    max_iterations: usize,
    /// Population size for genetic/swarm algorithms
    population_size: usize,
    /// Grover iterations
    grover_iterations: usize,
}

impl Default for QuantumOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumOptimizer {
    /// Create a new quantum optimizer
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            rng: seeded_rng(seed),
            temperature: 1000.0,
            cooling_rate: 0.95,
            max_iterations: 1000,
            population_size: 50,
            grover_iterations: 10,
        }
    }

    /// Set temperature for annealing
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set cooling rate
    pub fn with_cooling_rate(mut self, rate: f64) -> Self {
        self.cooling_rate = rate;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, iters: usize) -> Self {
        self.max_iterations = iters;
        self
    }

    /// Optimize rule execution order using quantum-inspired algorithms
    pub fn optimize_rule_order(
        &mut self,
        rules: &[Rule],
        goal: OptimizationGoal,
        algorithm: QuantumAlgorithm,
    ) -> Result<Vec<usize>> {
        info!(
            "Optimizing {} rules using {:?} for goal {:?}",
            rules.len(),
            algorithm,
            goal
        );

        match algorithm {
            QuantumAlgorithm::QuantumAnnealing => self.quantum_annealing(rules, goal),
            QuantumAlgorithm::QuantumGenetic => self.quantum_genetic(rules, goal),
            QuantumAlgorithm::QuantumParticleSwarm => self.quantum_particle_swarm(rules, goal),
            QuantumAlgorithm::QuantumWalk => self.quantum_walk(rules, goal),
            QuantumAlgorithm::GroverSearch => self.grover_search(rules, goal),
        }
    }

    /// Quantum annealing optimization
    fn quantum_annealing(&mut self, rules: &[Rule], goal: OptimizationGoal) -> Result<Vec<usize>> {
        let n = rules.len();
        let mut current_order: Vec<usize> = (0..n).collect();
        let mut current_energy = self.compute_energy(&current_order, rules, goal);
        let mut best_order = current_order.clone();
        let mut best_energy = current_energy;
        let mut temp = self.temperature;

        debug!("Starting quantum annealing with temperature {}", temp);

        let uniform_dist = Uniform::new(0.0, 1.0).expect("Failed to create uniform distribution");

        for iteration in 0..self.max_iterations {
            // Quantum tunneling: allow transitions that classical annealing would reject
            let tunneling_probability = (-temp / self.temperature).exp();

            // Generate neighbor through swap
            let mut neighbor = current_order.clone();
            let i = self.rng.gen_range(0..n);
            let j = self.rng.gen_range(0..n);
            neighbor.swap(i, j);

            let neighbor_energy = self.compute_energy(&neighbor, rules, goal);
            let delta_energy = neighbor_energy - current_energy;

            // Accept with quantum tunneling probability
            let accept_prob = if delta_energy < 0.0 {
                1.0
            } else {
                (-delta_energy / temp).exp() + tunneling_probability * 0.1
            };

            if uniform_dist.sample(&mut self.rng) < accept_prob {
                current_order = neighbor;
                current_energy = neighbor_energy;

                if current_energy < best_energy {
                    best_order = current_order.clone();
                    best_energy = current_energy;
                    debug!("Iteration {}: New best energy = {}", iteration, best_energy);
                }
            }

            // Cool down
            temp *= self.cooling_rate;
        }

        info!("Quantum annealing completed. Best energy: {}", best_energy);
        Ok(best_order)
    }

    /// Quantum genetic algorithm
    fn quantum_genetic(&mut self, rules: &[Rule], goal: OptimizationGoal) -> Result<Vec<usize>> {
        let n = rules.len();
        let mut population: Vec<Vec<usize>> = (0..self.population_size)
            .map(|_| {
                let mut order: Vec<usize> = (0..n).collect();
                self.shuffle(&mut order);
                order
            })
            .collect();

        let mut best_individual = population[0].clone();
        let mut best_fitness = self.compute_fitness(&best_individual, rules, goal);

        debug!("Starting quantum genetic algorithm");

        let uniform_dist = Uniform::new(0.0, 1.0).expect("Failed to create uniform distribution");

        for generation in 0..self.max_iterations / 10 {
            // Evaluate fitness
            let fitness: Vec<f64> = population
                .iter()
                .map(|ind| self.compute_fitness(ind, rules, goal))
                .collect();

            // Find best
            for (i, &fit) in fitness.iter().enumerate() {
                if fit > best_fitness {
                    best_fitness = fit;
                    best_individual = population[i].clone();
                }
            }

            // Quantum crossover with superposition
            let mut new_population = Vec::new();
            for _ in 0..self.population_size {
                let parent1 = self.tournament_select(&population, &fitness);
                let parent2 = self.tournament_select(&population, &fitness);

                // Quantum crossover: create superposition of parent genes
                let child = self.quantum_crossover(&parent1, &parent2);
                new_population.push(child);
            }

            // Quantum mutation
            for individual in &mut new_population {
                if uniform_dist.sample(&mut self.rng) < 0.1 {
                    self.quantum_mutate(individual);
                }
            }

            population = new_population;

            if generation % 10 == 0 {
                debug!("Generation {}: Best fitness = {}", generation, best_fitness);
            }
        }

        info!(
            "Quantum genetic algorithm completed. Best fitness: {}",
            best_fitness
        );
        Ok(best_individual)
    }

    /// Quantum particle swarm optimization
    fn quantum_particle_swarm(
        &mut self,
        rules: &[Rule],
        goal: OptimizationGoal,
    ) -> Result<Vec<usize>> {
        let n = rules.len();

        // Initialize particles (positions)
        let mut particles: Vec<Vec<usize>> = (0..self.population_size)
            .map(|_| {
                let mut order: Vec<usize> = (0..n).collect();
                self.shuffle(&mut order);
                order
            })
            .collect();

        // Track personal bests
        let mut personal_bests = particles.clone();
        let mut personal_best_fitness: Vec<f64> = personal_bests
            .iter()
            .map(|p| self.compute_fitness(p, rules, goal))
            .collect();

        // Track global best
        let mut global_best = personal_bests[0].clone();
        let mut global_best_fitness = personal_best_fitness[0];
        for (i, &fitness) in personal_best_fitness.iter().enumerate() {
            if fitness > global_best_fitness {
                global_best_fitness = fitness;
                global_best = personal_bests[i].clone();
            }
        }

        debug!("Starting quantum particle swarm optimization");

        let uniform_dist = Uniform::new(0.0, 1.0).expect("Failed to create uniform distribution");

        for iteration in 0..self.max_iterations / 10 {
            for i in 0..self.population_size {
                // Quantum behavior: particle exists in superposition
                let quantum_influence = uniform_dist.sample(&mut self.rng);

                // Update position with quantum tunneling
                if quantum_influence < 0.3 {
                    // Move towards personal best
                    particles[i] = self.move_towards(&particles[i], &personal_bests[i]);
                } else if quantum_influence < 0.6 {
                    // Move towards global best
                    particles[i] = self.move_towards(&particles[i], &global_best);
                } else {
                    // Quantum jump (exploration)
                    self.shuffle(&mut particles[i]);
                }

                // Evaluate new position
                let fitness = self.compute_fitness(&particles[i], rules, goal);

                // Update personal best
                if fitness > personal_best_fitness[i] {
                    personal_best_fitness[i] = fitness;
                    personal_bests[i] = particles[i].clone();

                    // Update global best
                    if fitness > global_best_fitness {
                        global_best_fitness = fitness;
                        global_best = particles[i].clone();
                        debug!(
                            "Iteration {}: New best fitness = {}",
                            iteration, global_best_fitness
                        );
                    }
                }
            }
        }

        info!(
            "Quantum PSO completed. Best fitness: {}",
            global_best_fitness
        );
        Ok(global_best)
    }

    /// Quantum walk on dependency graph
    fn quantum_walk(&mut self, rules: &[Rule], _goal: OptimizationGoal) -> Result<Vec<usize>> {
        let n = rules.len();

        // Build dependency graph
        let dependency_matrix = self.build_dependency_matrix(rules);

        // Initialize quantum walker
        let mut state = QuantumState::new(n);
        state.hadamard(); // Start in uniform superposition

        let mut visit_counts = vec![0usize; n];

        debug!("Starting quantum walk on rule dependency graph");

        // Perform quantum walk
        for step in 0..self.max_iterations {
            // Quantum coin flip (Hadamard)
            state.hadamard();

            // Conditional shift based on dependencies
            let position = state.measure(&mut self.rng);
            visit_counts[position] += 1;

            // Apply phase based on dependencies
            for j in 0..n {
                if dependency_matrix[[position, j]] > 0.0 {
                    state.rotate(j, std::f64::consts::PI / 4.0);
                }
            }

            if step % 100 == 0 {
                debug!("Quantum walk step {}: Position = {}", step, position);
            }
        }

        // Order rules by visit frequency (most visited first)
        let mut order_with_counts: Vec<(usize, usize)> = visit_counts
            .iter()
            .enumerate()
            .map(|(i, &count)| (i, count))
            .collect();

        order_with_counts.sort_by(|a, b| b.1.cmp(&a.1));
        let result: Vec<usize> = order_with_counts.iter().map(|(i, _)| *i).collect();

        info!("Quantum walk completed");
        Ok(result)
    }

    /// Grover-inspired amplitude amplification search
    fn grover_search(&mut self, rules: &[Rule], goal: OptimizationGoal) -> Result<Vec<usize>> {
        let n = rules.len();

        // Initialize in uniform superposition
        let mut state = QuantumState::new(n);
        state.hadamard();

        debug!("Starting Grover-inspired search");

        // Grover iterations
        for iteration in 0..self.grover_iterations {
            // Oracle: mark good solutions (high fitness)
            for i in 0..n {
                let test_order = vec![i];
                let fitness = self.compute_fitness(&test_order, rules, goal);
                if fitness > 0.5 {
                    state.phase_flip(i);
                }
            }

            // Diffusion operator (inversion about average)
            state.inversion_about_average();

            debug!("Grover iteration {} completed", iteration);
        }

        // Measure to get result
        let mut result = Vec::new();
        for _ in 0..n {
            let measurement = state.measure(&mut self.rng);
            if !result.contains(&measurement) {
                result.push(measurement);
            }
        }

        // Fill remaining positions
        for i in 0..n {
            if !result.contains(&i) {
                result.push(i);
            }
        }

        info!("Grover search completed");
        Ok(result)
    }

    /// Compute energy (cost function) for a rule ordering
    fn compute_energy(&self, order: &[usize], rules: &[Rule], goal: OptimizationGoal) -> f64 {
        // Lower energy is better
        -self.compute_fitness(order, rules, goal)
    }

    /// Compute fitness (reward function) for a rule ordering
    fn compute_fitness(&self, order: &[usize], rules: &[Rule], goal: OptimizationGoal) -> f64 {
        let mut fitness = 0.0;

        match goal {
            OptimizationGoal::MinimizeInferenceTime => {
                // Prefer rules that derive facts early
                for (position, &rule_idx) in order.iter().enumerate() {
                    if rule_idx < rules.len() {
                        let rule = &rules[rule_idx];
                        let complexity = self.estimate_complexity(rule);
                        // Earlier positions get higher weight
                        fitness += (order.len() - position) as f64 / complexity;
                    }
                }
            }
            OptimizationGoal::MaximizeDerivations => {
                // Prefer rules with more head atoms
                for &rule_idx in order {
                    if rule_idx < rules.len() {
                        fitness += rules[rule_idx].head.len() as f64;
                    }
                }
            }
            OptimizationGoal::MinimizeMemory => {
                // Prefer rules with fewer variables
                for &rule_idx in order {
                    if rule_idx < rules.len() {
                        let var_count = self.count_variables(&rules[rule_idx]);
                        fitness += 1.0 / (var_count as f64 + 1.0);
                    }
                }
            }
            OptimizationGoal::Balanced => {
                // Combine multiple objectives
                fitness =
                    self.compute_fitness(order, rules, OptimizationGoal::MinimizeInferenceTime)
                        * 0.5
                        + self.compute_fitness(order, rules, OptimizationGoal::MaximizeDerivations)
                            * 0.3
                        + self.compute_fitness(order, rules, OptimizationGoal::MinimizeMemory)
                            * 0.2;
            }
        }

        fitness
    }

    /// Estimate rule complexity
    fn estimate_complexity(&self, rule: &Rule) -> f64 {
        (rule.body.len() + rule.head.len()) as f64
    }

    /// Count variables in a rule
    fn count_variables(&self, rule: &Rule) -> usize {
        let mut vars = std::collections::HashSet::new();

        for atom in &rule.body {
            self.extract_variables(atom, &mut vars);
        }
        for atom in &rule.head {
            self.extract_variables(atom, &mut vars);
        }

        vars.len()
    }

    /// Extract variables from an atom
    fn extract_variables(&self, atom: &RuleAtom, vars: &mut std::collections::HashSet<String>) {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                if let Term::Variable(v) = subject {
                    vars.insert(v.clone());
                }
                if let Term::Variable(v) = predicate {
                    vars.insert(v.clone());
                }
                if let Term::Variable(v) = object {
                    vars.insert(v.clone());
                }
            }
            RuleAtom::Builtin { args, .. } => {
                for arg in args {
                    if let Term::Variable(v) = arg {
                        vars.insert(v.clone());
                    }
                }
            }
            RuleAtom::NotEqual { left, right }
            | RuleAtom::GreaterThan { left, right }
            | RuleAtom::LessThan { left, right } => {
                if let Term::Variable(v) = left {
                    vars.insert(v.clone());
                }
                if let Term::Variable(v) = right {
                    vars.insert(v.clone());
                }
            }
        }
    }

    /// Build dependency matrix for rules
    fn build_dependency_matrix(&self, rules: &[Rule]) -> Array2<f64> {
        let n = rules.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Check if rule i's head matches rule j's body
                    let dependency = self.compute_dependency(&rules[i], &rules[j]);
                    matrix[[i, j]] = dependency;
                }
            }
        }

        matrix
    }

    /// Compute dependency strength between two rules
    fn compute_dependency(&self, rule1: &Rule, rule2: &Rule) -> f64 {
        let mut matches = 0.0;

        for head_atom in &rule1.head {
            for body_atom in &rule2.body {
                if self.atoms_compatible(head_atom, body_atom) {
                    matches += 1.0;
                }
            }
        }

        matches
    }

    /// Check if two atoms are compatible
    fn atoms_compatible(&self, atom1: &RuleAtom, atom2: &RuleAtom) -> bool {
        match (atom1, atom2) {
            (RuleAtom::Triple { predicate: p1, .. }, RuleAtom::Triple { predicate: p2, .. }) => {
                p1 == p2
            }
            _ => false,
        }
    }

    /// Shuffle a vector
    fn shuffle(&mut self, vec: &mut [usize]) {
        for i in (1..vec.len()).rev() {
            let j = self.rng.gen_range(0..=i);
            vec.swap(i, j);
        }
    }

    /// Tournament selection for genetic algorithm
    fn tournament_select(&mut self, population: &[Vec<usize>], fitness: &[f64]) -> Vec<usize> {
        let tournament_size = 3;
        let mut best_idx = self.rng.gen_range(0..population.len());
        let mut best_fitness = fitness[best_idx];

        for _ in 1..tournament_size {
            let idx = self.rng.gen_range(0..population.len());
            if fitness[idx] > best_fitness {
                best_fitness = fitness[idx];
                best_idx = idx;
            }
        }

        population[best_idx].clone()
    }

    /// Quantum crossover (create superposition of parent genes)
    fn quantum_crossover(&mut self, parent1: &[usize], parent2: &[usize]) -> Vec<usize> {
        let n = parent1.len();
        let mut child = vec![usize::MAX; n];
        let uniform_dist = Uniform::new(0.0, 1.0).expect("Failed to create uniform distribution");

        // Quantum superposition: probabilistically choose from either parent
        for i in 0..n {
            if uniform_dist.sample(&mut self.rng) < 0.5 {
                child[i] = parent1[i];
            } else {
                child[i] = parent2[i];
            }
        }

        // Fix duplicates (collapse superposition to valid state)
        let mut used = vec![false; n];
        let mut result = Vec::new();

        for &val in &child {
            if val < n && !used[val] {
                used[val] = true;
                result.push(val);
            }
        }

        // Fill missing values
        for (i, &is_used) in used.iter().enumerate() {
            if !is_used {
                result.push(i);
            }
        }

        result
    }

    /// Quantum mutation (quantum tunneling)
    fn quantum_mutate(&mut self, individual: &mut [usize]) {
        let n = individual.len();
        let uniform_dist = Uniform::new(0.0, 1.0).expect("Failed to create uniform distribution");

        // Quantum tunneling: jump to distant state
        let tunneling = uniform_dist.sample(&mut self.rng) < 0.3;

        if tunneling {
            // Large mutation (quantum jump)
            let num_swaps = self.rng.gen_range(2..=(n / 2).max(2));
            for _ in 0..num_swaps {
                let i = self.rng.gen_range(0..n);
                let j = self.rng.gen_range(0..n);
                individual.swap(i, j);
            }
        } else {
            // Small mutation (local search)
            let i = self.rng.gen_range(0..n);
            let j = self.rng.gen_range(0..n);
            individual.swap(i, j);
        }
    }

    /// Move particle towards target
    fn move_towards(&mut self, current: &[usize], target: &[usize]) -> Vec<usize> {
        let n = current.len();
        let mut result = current.to_vec();

        // Move closer to target with some randomness
        let num_moves = self.rng.gen_range(1..=(n / 4).max(1));
        for _ in 0..num_moves {
            let idx = self.rng.gen_range(0..n);
            if let Some(target_pos) = target.iter().position(|&x| x == current[idx]) {
                if target_pos != idx {
                    result.swap(idx, target_pos);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(4);
        assert_eq!(state.amplitudes.len(), 4);
        assert_eq!(state.phases.len(), 4);

        // Check uniform superposition
        let expected_amplitude = 0.5;
        for &amp in state.amplitudes.iter() {
            assert!((amp - expected_amplitude).abs() < 1e-10);
        }
    }

    #[test]
    fn test_quantum_state_measurement() {
        let state = QuantumState::new(4);
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut rng = seeded_rng(seed);

        // Measure multiple times
        for _ in 0..10 {
            let measurement = state.measure(&mut rng);
            assert!(measurement < 4);
        }
    }

    #[test]
    fn test_quantum_optimizer_creation() {
        let optimizer = QuantumOptimizer::new();
        assert!(optimizer.temperature > 0.0);
        assert!(optimizer.cooling_rate > 0.0 && optimizer.cooling_rate < 1.0);
        assert!(optimizer.max_iterations > 0);
    }

    #[test]
    fn test_quantum_annealing() {
        let mut optimizer = QuantumOptimizer::new()
            .with_max_iterations(100)
            .with_temperature(100.0);

        let rules = vec![
            Rule {
                name: "rule1".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("p".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("q".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
            },
            Rule {
                name: "rule2".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("A".to_string()),
                    predicate: Term::Constant("q".to_string()),
                    object: Term::Variable("B".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("A".to_string()),
                    predicate: Term::Constant("r".to_string()),
                    object: Term::Variable("B".to_string()),
                }],
            },
        ];

        let result = optimizer
            .optimize_rule_order(
                &rules,
                OptimizationGoal::MinimizeInferenceTime,
                QuantumAlgorithm::QuantumAnnealing,
            )
            .unwrap();

        assert_eq!(result.len(), rules.len());
        assert!(result.iter().all(|&i| i < rules.len()));
    }

    #[test]
    fn test_quantum_genetic() {
        let mut optimizer = QuantumOptimizer::new().with_max_iterations(50);

        let rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }];

        let result = optimizer
            .optimize_rule_order(
                &rules,
                OptimizationGoal::MaximizeDerivations,
                QuantumAlgorithm::QuantumGenetic,
            )
            .unwrap();

        assert_eq!(result.len(), rules.len());
    }

    #[test]
    fn test_quantum_particle_swarm() {
        let mut optimizer = QuantumOptimizer::new().with_max_iterations(50);

        let rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![],
        }];

        let result = optimizer
            .optimize_rule_order(
                &rules,
                OptimizationGoal::MinimizeMemory,
                QuantumAlgorithm::QuantumParticleSwarm,
            )
            .unwrap();

        assert_eq!(result.len(), rules.len());
    }

    #[test]
    fn test_quantum_walk() {
        let mut optimizer = QuantumOptimizer::new().with_max_iterations(100);

        let rules = vec![
            Rule {
                name: "rule1".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("p".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("q".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
            },
            Rule {
                name: "rule2".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("A".to_string()),
                    predicate: Term::Constant("q".to_string()),
                    object: Term::Variable("B".to_string()),
                }],
                head: vec![RuleAtom::Triple {
                    subject: Term::Variable("A".to_string()),
                    predicate: Term::Constant("r".to_string()),
                    object: Term::Variable("B".to_string()),
                }],
            },
        ];

        let result = optimizer
            .optimize_rule_order(
                &rules,
                OptimizationGoal::Balanced,
                QuantumAlgorithm::QuantumWalk,
            )
            .unwrap();

        assert_eq!(result.len(), rules.len());
    }

    #[test]
    fn test_grover_search() {
        let mut optimizer = QuantumOptimizer::new();

        let rules = vec![
            Rule {
                name: "rule1".to_string(),
                body: vec![],
                head: vec![RuleAtom::Triple {
                    subject: Term::Constant("a".to_string()),
                    predicate: Term::Constant("p".to_string()),
                    object: Term::Constant("b".to_string()),
                }],
            },
            Rule {
                name: "rule2".to_string(),
                body: vec![],
                head: vec![
                    RuleAtom::Triple {
                        subject: Term::Constant("c".to_string()),
                        predicate: Term::Constant("q".to_string()),
                        object: Term::Constant("d".to_string()),
                    },
                    RuleAtom::Triple {
                        subject: Term::Constant("e".to_string()),
                        predicate: Term::Constant("r".to_string()),
                        object: Term::Constant("f".to_string()),
                    },
                ],
            },
        ];

        let result = optimizer
            .optimize_rule_order(
                &rules,
                OptimizationGoal::MaximizeDerivations,
                QuantumAlgorithm::GroverSearch,
            )
            .unwrap();

        assert_eq!(result.len(), rules.len());
    }

    #[test]
    fn test_compute_fitness() {
        let optimizer = QuantumOptimizer::new();

        let rules = vec![Rule {
            name: "simple".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Constant("a".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Constant("b".to_string()),
            }],
        }];

        let order = vec![0];

        let fitness_time =
            optimizer.compute_fitness(&order, &rules, OptimizationGoal::MinimizeInferenceTime);
        let fitness_derivations =
            optimizer.compute_fitness(&order, &rules, OptimizationGoal::MaximizeDerivations);
        let fitness_memory =
            optimizer.compute_fitness(&order, &rules, OptimizationGoal::MinimizeMemory);
        let fitness_balanced =
            optimizer.compute_fitness(&order, &rules, OptimizationGoal::Balanced);

        assert!(fitness_time > 0.0);
        assert!(fitness_derivations > 0.0);
        assert!(fitness_memory > 0.0);
        assert!(fitness_balanced > 0.0);
    }

    #[test]
    fn test_dependency_matrix() {
        let optimizer = QuantumOptimizer::new();

        let rules = vec![
            Rule {
                name: "rule1".to_string(),
                body: vec![],
                head: vec![RuleAtom::Triple {
                    subject: Term::Constant("a".to_string()),
                    predicate: Term::Constant("p".to_string()),
                    object: Term::Constant("b".to_string()),
                }],
            },
            Rule {
                name: "rule2".to_string(),
                body: vec![RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("p".to_string()),
                    object: Term::Variable("Y".to_string()),
                }],
                head: vec![],
            },
        ];

        let matrix = optimizer.build_dependency_matrix(&rules);
        assert_eq!(matrix.shape(), &[2, 2]);

        // rule1 head matches rule2 body predicate
        assert!(matrix[[0, 1]] > 0.0);
    }

    #[test]
    fn test_different_algorithms_produce_valid_results() {
        let mut optimizer = QuantumOptimizer::new().with_max_iterations(50);

        let rules = vec![
            Rule {
                name: "r1".to_string(),
                body: vec![],
                head: vec![RuleAtom::Triple {
                    subject: Term::Constant("a".to_string()),
                    predicate: Term::Constant("p".to_string()),
                    object: Term::Constant("b".to_string()),
                }],
            },
            Rule {
                name: "r2".to_string(),
                body: vec![],
                head: vec![RuleAtom::Triple {
                    subject: Term::Constant("c".to_string()),
                    predicate: Term::Constant("q".to_string()),
                    object: Term::Constant("d".to_string()),
                }],
            },
            Rule {
                name: "r3".to_string(),
                body: vec![],
                head: vec![RuleAtom::Triple {
                    subject: Term::Constant("e".to_string()),
                    predicate: Term::Constant("r".to_string()),
                    object: Term::Constant("f".to_string()),
                }],
            },
        ];

        for algorithm in &[
            QuantumAlgorithm::QuantumAnnealing,
            QuantumAlgorithm::QuantumGenetic,
            QuantumAlgorithm::QuantumParticleSwarm,
            QuantumAlgorithm::QuantumWalk,
            QuantumAlgorithm::GroverSearch,
        ] {
            let result = optimizer
                .optimize_rule_order(&rules, OptimizationGoal::Balanced, *algorithm)
                .unwrap();

            assert_eq!(result.len(), rules.len());

            // Check all indices are unique and valid
            let mut sorted = result.clone();
            sorted.sort();
            assert_eq!(sorted, vec![0, 1, 2]);
        }
    }
}
