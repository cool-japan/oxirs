//! Genetic algorithm implementation for SHACL shape constraint-order optimization.
//!
//! Uses tournament-selection, single-point crossover, and rotation-based mutation
//! to discover a constraint ordering that minimises total estimated evaluation cost.

use indexmap::IndexMap;
use oxirs_shacl::{Constraint, ConstraintComponentId, Shape};

/// Parameters that control the genetic optimisation run.
pub(super) struct GeneticParams {
    pub population_size: usize,
    pub num_generations: usize,
    pub tournament_k: usize,
    /// Mutation rate expressed as a fraction in [0, 1].
    pub mutation_rate: f64,
}

impl Default for GeneticParams {
    fn default() -> Self {
        Self {
            population_size: 20,
            num_generations: 50,
            tournament_k: 3,
            mutation_rate: 0.1,
        }
    }
}

/// A fitness callable that scores a population member.
///
/// Higher score is better (costs are negated before returning).
pub(super) fn fitness_of<F>(candidate: &[Shape], cost_fn: &F) -> f64
where
    F: Fn(&Constraint) -> f64,
{
    let total_cost: f64 = candidate
        .iter()
        .map(|s| s.constraints.values().map(cost_fn).sum::<f64>())
        .sum();
    -total_cost
}

/// Rotate the constraint map of a shape by `rotate_by` positions (deterministic).
pub(super) fn rotate_constraints(shape: &mut Shape, rotate_by: usize) {
    let len = shape.constraints.len();
    if len < 2 {
        return;
    }
    let rotate_by = rotate_by % len;
    let keys: Vec<ConstraintComponentId> = shape.constraints.keys().cloned().collect();
    let values: Vec<Constraint> = shape.constraints.values().cloned().collect();
    let mut rotated = IndexMap::with_capacity(len);
    for (i, key) in keys.iter().enumerate().take(len) {
        let src = (i + rotate_by) % len;
        rotated.insert(key.clone(), values[src].clone());
    }
    shape.constraints = rotated;
}

/// Run the genetic algorithm and return the best discovered ordering.
///
/// `cost_fn` maps a constraint to its estimated evaluation cost in [0, 1].
pub(super) fn run<F>(shapes: Vec<Shape>, params: &GeneticParams, cost_fn: &F) -> Vec<Shape>
where
    F: Fn(&Constraint) -> f64,
{
    if shapes.len() < 2 {
        return shapes;
    }

    // Seed population: identity first, remainder are rotations.
    let mut population: Vec<Vec<Shape>> = Vec::with_capacity(params.population_size);
    population.push(shapes.clone());

    for pop_idx in 1..params.population_size {
        let mut individual: Vec<Shape> = shapes.clone();
        for shape in &mut individual {
            rotate_constraints(shape, pop_idx);
        }
        population.push(individual);
    }

    // Evolve for `num_generations` generations.
    for generation in 0..params.num_generations {
        let fitnesses: Vec<f64> = population
            .iter()
            .map(|ind| fitness_of(ind, cost_fn))
            .collect();

        // Tournament selection: deterministic, index-arithmetic based.
        let select = |seed: usize| -> usize {
            let mut best_idx = seed % population.len();
            let mut best_fit = fitnesses[best_idx];
            for k in 1..params.tournament_k {
                let candidate = (seed + k * 7 + generation) % population.len();
                if fitnesses[candidate] > best_fit {
                    best_fit = fitnesses[candidate];
                    best_idx = candidate;
                }
            }
            best_idx
        };

        let mut new_population: Vec<Vec<Shape>> = Vec::with_capacity(params.population_size);

        // Elitism: keep the current best unchanged.
        let elite_idx = fitnesses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        new_population.push(population[elite_idx].clone());

        while new_population.len() < params.population_size {
            let p1_idx = select(new_population.len() * 3 + generation);
            let p2_idx = select(new_population.len() * 5 + generation + 1);
            let parent1 = &population[p1_idx];
            let parent2 = &population[p2_idx];

            // Single-point crossover at a shape boundary.
            let crossover_point = (new_population.len() + generation) % shapes.len().max(1);
            let mut child: Vec<Shape> = parent1[..crossover_point].to_vec();
            child.extend_from_slice(&parent2[crossover_point..]);

            // Mutation: rotate constraints in one shape by 1 position.
            let mutate_idx = (new_population.len() + generation * 3) % child.len();
            let mutate_threshold = (params.mutation_rate * 100.0) as usize;
            if (generation + new_population.len()) % 100 < mutate_threshold {
                rotate_constraints(&mut child[mutate_idx], 1);
            }

            new_population.push(child);
        }

        population = new_population;
    }

    // Return the fittest individual.
    let final_fitnesses: Vec<f64> = population
        .iter()
        .map(|ind| fitness_of(ind, cost_fn))
        .collect();
    let best_idx = final_fitnesses
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    population.remove(best_idx)
}
