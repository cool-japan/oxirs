//! Genetic Algorithm Optimization for Graph Structures
//!
//! This module implements evolutionary optimization for RDF graph storage and query
//! execution using genetic algorithms inspired by biological evolution.

use super::dna_structures::DnaDataStructure;
use crate::model::Triple;
use crate::OxirsError;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Genetic algorithm for optimizing graph structures
pub struct GeneticGraphOptimizer {
    /// Population of graph structures
    population: Vec<GraphStructure>,
    /// Fitness function for evaluating structures
    fitness_function: Box<dyn Fn(&GraphStructure) -> f64 + Send + Sync>,
    /// Mutation rate (0.0 to 1.0)
    mutation_rate: f64,
    /// Crossover rate (0.0 to 1.0)
    crossover_rate: f64,
    /// Number of generations to evolve
    generations: usize,
    /// Population size
    population_size: usize,
    /// Elite size (top performers to preserve)
    elite_size: usize,
    /// Current generation number
    current_generation: usize,
    /// Best fitness achieved so far
    best_fitness: f64,
    /// Evolution history
    evolution_history: Vec<GenerationStats>,
    /// Best individual found so far
    best_individual: Option<GraphStructure>,
}

/// Graph structure chromosome representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStructure {
    /// DNA representation of the graph
    pub dna: DnaDataStructure,
    /// Indexing strategy genes
    pub indexing_genes: IndexingGenes,
    /// Storage layout genes
    pub storage_genes: StorageGenes,
    /// Access pattern genes
    pub access_genes: AccessGenes,
    /// Fitness score
    pub fitness: f64,
    /// Age in generations
    pub age: usize,
    /// Mutation history
    pub mutations: Vec<MutationType>,
}

/// Indexing strategy encoded as genetic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingGenes {
    /// Primary index type
    pub primary_index: IndexGene,
    /// Secondary indexes
    pub secondary_indexes: Vec<IndexGene>,
    /// Index compression settings
    pub compression: CompressionGene,
    /// Adaptive index triggers
    pub adaptive_triggers: Vec<AdaptiveTrigger>,
}

/// Single index gene
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexGene {
    /// Index type identifier
    pub index_type: String,
    /// Index parameters
    pub parameters: Vec<f64>,
    /// Enabled status
    pub enabled: bool,
    /// Priority level
    pub priority: u8,
}

/// Storage layout genetic encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageGenes {
    /// Block size for storage
    pub block_size: u32,
    /// Clustering strategy
    pub clustering: ClusteringGene,
    /// Partitioning strategy
    pub partitioning: PartitioningGene,
    /// Caching configuration
    pub caching: CachingGene,
}

/// Clustering genetic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringGene {
    /// Clustering algorithm
    pub algorithm: String,
    /// Cluster size target
    pub target_size: u32,
    /// Similarity threshold
    pub similarity_threshold: f64,
    /// Rebalancing frequency
    pub rebalance_frequency: u32,
}

/// Partitioning strategy gene
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitioningGene {
    /// Partitioning method
    pub method: String,
    /// Number of partitions
    pub partition_count: u32,
    /// Load balancing factor
    pub load_balance_factor: f64,
    /// Hot data threshold
    pub hot_data_threshold: f64,
}

/// Caching configuration gene
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingGene {
    /// Cache size in MB
    pub cache_size_mb: u32,
    /// Eviction policy
    pub eviction_policy: String,
    /// Prefetch strategy
    pub prefetch_strategy: String,
    /// Write policy
    pub write_policy: String,
}

/// Compression gene for storage optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionGene {
    /// Compression algorithm
    pub algorithm: String,
    /// Compression level (1-9)
    pub level: u8,
    /// Block size for compression
    pub block_size: u32,
    /// Dictionary size
    pub dictionary_size: u32,
}

/// Access pattern genetic encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessGenes {
    /// Read pattern preferences
    pub read_patterns: Vec<AccessPattern>,
    /// Write pattern preferences
    pub write_patterns: Vec<AccessPattern>,
    /// Query optimization preferences
    pub query_preferences: QueryPreferences,
    /// Concurrent access settings
    pub concurrency: ConcurrencyGene,
}

/// Individual access pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Frequency weight
    pub frequency: f64,
    /// Optimization hint
    pub optimization: String,
    /// Buffer size preference
    pub buffer_size: u32,
}

/// Query preferences genetic encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPreferences {
    /// Join algorithm preference
    pub join_algorithm: String,
    /// Index selection strategy
    pub index_selection: String,
    /// Result caching strategy
    pub result_caching: String,
    /// Parallel execution preference
    pub parallel_execution: bool,
}

/// Concurrency genetic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyGene {
    /// Maximum concurrent readers
    pub max_readers: u32,
    /// Maximum concurrent writers
    pub max_writers: u32,
    /// Lock timeout in milliseconds
    pub lock_timeout_ms: u32,
    /// Thread pool size
    pub thread_pool_size: u32,
}

/// Adaptive trigger for dynamic behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveTrigger {
    /// Trigger condition
    pub condition: String,
    /// Threshold value
    pub threshold: f64,
    /// Action to take
    pub action: String,
    /// Cooldown period in seconds
    pub cooldown_seconds: u32,
}

/// Types of mutations that can occur
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationType {
    /// Index configuration change
    IndexMutation {
        index_id: String,
        old_value: String,
        new_value: String,
    },
    /// Storage parameter change
    StorageMutation {
        parameter: String,
        old_value: f64,
        new_value: f64,
    },
    /// Access pattern modification
    AccessMutation {
        pattern_id: String,
        modification: String,
    },
    /// Gene deletion
    GeneDeletion { gene_type: String, gene_id: String },
    /// Gene duplication
    GeneDuplication { gene_type: String, gene_id: String },
    /// Chromosomal rearrangement
    ChromosomalRearrangement {
        old_order: Vec<String>,
        new_order: Vec<String>,
    },
}

/// Statistics for a generation
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Generation number
    pub generation: usize,
    /// Best fitness in generation
    pub best_fitness: f64,
    /// Average fitness
    pub average_fitness: f64,
    /// Worst fitness
    pub worst_fitness: f64,
    /// Genetic diversity measure
    pub diversity: f64,
    /// Number of mutations
    pub mutations: usize,
    /// Number of crossovers
    pub crossovers: usize,
    /// Evolution time for this generation
    pub evolution_time: Duration,
}

impl GeneticGraphOptimizer {
    /// Get the best fitness achieved so far
    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }

    /// Get the best individual found so far
    pub fn best_individual(&self) -> Option<&GraphStructure> {
        self.best_individual.as_ref()
    }

    /// Update best individual if fitness improved
    fn update_best_individual(&mut self, individual: &GraphStructure) {
        if individual.fitness > self.best_fitness {
            self.best_fitness = individual.fitness;
            self.best_individual = Some(individual.clone());
        }
    }
    /// Create a new genetic optimizer
    pub fn new(
        population_size: usize,
        fitness_function: Box<dyn Fn(&GraphStructure) -> f64 + Send + Sync>,
    ) -> Self {
        Self {
            population: Vec::new(),
            fitness_function,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            generations: 100,
            population_size,
            elite_size: population_size / 10,
            current_generation: 0,
            best_fitness: 0.0,
            evolution_history: Vec::new(),
            best_individual: None,
        }
    }

    /// Set evolution parameters
    pub fn set_parameters(&mut self, mutation_rate: f64, crossover_rate: f64, generations: usize) {
        self.mutation_rate = mutation_rate;
        self.crossover_rate = crossover_rate;
        self.generations = generations;
    }

    /// Initialize random population
    pub fn initialize_population(&mut self, base_triples: &[Triple]) -> Result<(), OxirsError> {
        self.population.clear();

        for _ in 0..self.population_size {
            let mut structure = self.create_random_structure(base_triples)?;
            structure.fitness = (self.fitness_function)(&structure);
            self.population.push(structure);
        }

        // Sort by fitness (highest first)
        self.population
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        self.best_fitness = self.population[0].fitness;

        Ok(())
    }

    /// Create a random graph structure
    fn create_random_structure(
        &self,
        base_triples: &[Triple],
    ) -> Result<GraphStructure, OxirsError> {
        let mut dna = DnaDataStructure::new();

        // Encode base triples into DNA
        for triple in base_triples {
            dna.encode_triple(triple)?;
        }

        // Generate random genes
        let indexing_genes = self.generate_random_indexing_genes();
        let storage_genes = self.generate_random_storage_genes();
        let access_genes = self.generate_random_access_genes();

        Ok(GraphStructure {
            dna,
            indexing_genes,
            storage_genes,
            access_genes,
            fitness: 0.0,
            age: 0,
            mutations: Vec::new(),
        })
    }

    /// Generate random indexing genes
    fn generate_random_indexing_genes(&self) -> IndexingGenes {
        IndexingGenes {
            primary_index: IndexGene {
                index_type: "SPO".to_string(),
                parameters: vec![fastrand::f64(), fastrand::f64(), fastrand::f64()],
                enabled: true,
                priority: 10,
            },
            secondary_indexes: vec![
                IndexGene {
                    index_type: "POS".to_string(),
                    parameters: vec![fastrand::f64(), fastrand::f64()],
                    enabled: fastrand::bool(),
                    priority: fastrand::u8(..10),
                },
                IndexGene {
                    index_type: "OSP".to_string(),
                    parameters: vec![fastrand::f64(), fastrand::f64()],
                    enabled: fastrand::bool(),
                    priority: fastrand::u8(..10),
                },
            ],
            compression: CompressionGene {
                algorithm: "LZ4".to_string(),
                level: fastrand::u8(1..10),
                block_size: 4096 * fastrand::u32(1..17),
                dictionary_size: 1024 * fastrand::u32(1..65),
            },
            adaptive_triggers: vec![AdaptiveTrigger {
                condition: "high_load".to_string(),
                threshold: 0.8 + fastrand::f64() * 0.2,
                action: "create_index".to_string(),
                cooldown_seconds: 60 + fastrand::u32(..300),
            }],
        }
    }

    /// Generate random storage genes
    fn generate_random_storage_genes(&self) -> StorageGenes {
        StorageGenes {
            block_size: 4096 * fastrand::u32(1..33),
            clustering: ClusteringGene {
                algorithm: "k-means".to_string(),
                target_size: 100 + fastrand::u32(..900),
                similarity_threshold: 0.1 + fastrand::f64() * 0.8,
                rebalance_frequency: 1000 + fastrand::u32(..9000),
            },
            partitioning: PartitioningGene {
                method: "hash".to_string(),
                partition_count: 4 + fastrand::u32(..28),
                load_balance_factor: 0.1 + fastrand::f64() * 0.4,
                hot_data_threshold: 0.8 + fastrand::f64() * 0.2,
            },
            caching: CachingGene {
                cache_size_mb: 64 + fastrand::u32(..1936),
                eviction_policy: "LRU".to_string(),
                prefetch_strategy: "sequential".to_string(),
                write_policy: "write-back".to_string(),
            },
        }
    }

    /// Generate random access genes
    fn generate_random_access_genes(&self) -> AccessGenes {
        AccessGenes {
            read_patterns: vec![
                AccessPattern {
                    pattern_id: "sequential".to_string(),
                    frequency: fastrand::f64(),
                    optimization: "prefetch".to_string(),
                    buffer_size: 1024 + fastrand::u32(..7168),
                },
                AccessPattern {
                    pattern_id: "random".to_string(),
                    frequency: fastrand::f64(),
                    optimization: "cache".to_string(),
                    buffer_size: 512 + fastrand::u32(..3584),
                },
            ],
            write_patterns: vec![AccessPattern {
                pattern_id: "batch".to_string(),
                frequency: fastrand::f64(),
                optimization: "buffer".to_string(),
                buffer_size: 2048 + fastrand::u32(..14336),
            }],
            query_preferences: QueryPreferences {
                join_algorithm: "hash_join".to_string(),
                index_selection: "cost_based".to_string(),
                result_caching: "enabled".to_string(),
                parallel_execution: fastrand::bool(),
            },
            concurrency: ConcurrencyGene {
                max_readers: 10 + fastrand::u32(..90),
                max_writers: 1 + fastrand::u32(..9),
                lock_timeout_ms: 100 + fastrand::u32(..1900),
                thread_pool_size: 4 + fastrand::u32(..28),
            },
        }
    }

    /// Evolve the population for one generation
    pub fn evolve_generation(&mut self) -> Result<GenerationStats, OxirsError> {
        let start_time = Instant::now();
        let mut mutations = 0;
        let mut crossovers = 0;

        // Create new generation
        let mut new_population = Vec::new();

        // Keep elite individuals
        for i in 0..self.elite_size {
            let mut elite = self.population[i].clone();
            elite.age += 1;
            new_population.push(elite);
        }

        // Generate offspring through crossover and mutation
        while new_population.len() < self.population_size {
            // Selection
            let parent1 = self.tournament_selection();
            let parent2 = self.tournament_selection();

            // Crossover
            let mut offspring = if fastrand::f64() < self.crossover_rate {
                crossovers += 1;
                self.crossover(&parent1, &parent2)?
            } else {
                parent1.clone()
            };

            // Mutation
            if fastrand::f64() < self.mutation_rate {
                mutations += 1;
                self.mutate(&mut offspring)?;
            }

            // Evaluate fitness
            offspring.fitness = (self.fitness_function)(&offspring);
            offspring.age = 0;

            new_population.push(offspring);
        }

        // Replace population
        self.population = new_population;

        // Sort by fitness
        self.population
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Update best fitness
        if self.population[0].fitness > self.best_fitness {
            self.best_fitness = self.population[0].fitness;
        }

        // Calculate statistics
        let stats = GenerationStats {
            generation: self.current_generation,
            best_fitness: self.population[0].fitness,
            average_fitness: self.population.iter().map(|s| s.fitness).sum::<f64>()
                / self.population.len() as f64,
            worst_fitness: self.population.last().unwrap().fitness,
            diversity: self.calculate_diversity(),
            mutations,
            crossovers,
            evolution_time: start_time.elapsed(),
        };

        self.evolution_history.push(stats.clone());
        self.current_generation += 1;

        Ok(stats)
    }

    /// Tournament selection for parent selection
    fn tournament_selection(&self) -> GraphStructure {
        let tournament_size = 3;
        let mut best_fitness = -1.0;
        let mut best_individual = self.population[0].clone();

        for _ in 0..tournament_size {
            let candidate = &self.population[fastrand::usize(..self.population.len())];
            if candidate.fitness > best_fitness {
                best_fitness = candidate.fitness;
                best_individual = candidate.clone();
            }
        }

        best_individual
    }

    /// Crossover two parent structures
    fn crossover(
        &self,
        parent1: &GraphStructure,
        parent2: &GraphStructure,
    ) -> Result<GraphStructure, OxirsError> {
        let mut offspring = parent1.clone();

        // Crossover indexing genes
        if fastrand::bool() {
            offspring.indexing_genes.compression = parent2.indexing_genes.compression.clone();
        }

        if fastrand::bool() {
            offspring.indexing_genes.secondary_indexes =
                parent2.indexing_genes.secondary_indexes.clone();
        }

        // Crossover storage genes
        if fastrand::bool() {
            offspring.storage_genes.clustering = parent2.storage_genes.clustering.clone();
        }

        if fastrand::bool() {
            offspring.storage_genes.caching = parent2.storage_genes.caching.clone();
        }

        // Crossover access genes
        if fastrand::bool() {
            offspring.access_genes.query_preferences =
                parent2.access_genes.query_preferences.clone();
        }

        Ok(offspring)
    }

    /// Mutate a structure
    fn mutate(&self, structure: &mut GraphStructure) -> Result<(), OxirsError> {
        let mutation_types = [
            "index_parameter",
            "storage_parameter",
            "access_pattern",
            "compression_level",
            "cache_size",
        ];

        let mutation_type = &mutation_types[fastrand::usize(..mutation_types.len())];

        match *mutation_type {
            "index_parameter" => {
                if !structure.indexing_genes.secondary_indexes.is_empty() {
                    let index_idx =
                        fastrand::usize(..structure.indexing_genes.secondary_indexes.len());
                    let param_idx = fastrand::usize(
                        ..structure.indexing_genes.secondary_indexes[index_idx]
                            .parameters
                            .len(),
                    );
                    let old_value =
                        structure.indexing_genes.secondary_indexes[index_idx].parameters[param_idx];
                    structure.indexing_genes.secondary_indexes[index_idx].parameters[param_idx] =
                        fastrand::f64();

                    structure.mutations.push(MutationType::IndexMutation {
                        index_id: format!("secondary_{}", index_idx),
                        old_value: old_value.to_string(),
                        new_value: structure.indexing_genes.secondary_indexes[index_idx].parameters
                            [param_idx]
                            .to_string(),
                    });
                }
            }
            "storage_parameter" => {
                let old_value = structure.storage_genes.block_size as f64;
                structure.storage_genes.block_size = 4096 * fastrand::u32(1..33);

                structure.mutations.push(MutationType::StorageMutation {
                    parameter: "block_size".to_string(),
                    old_value,
                    new_value: structure.storage_genes.block_size as f64,
                });
            }
            "compression_level" => {
                let old_level = structure.indexing_genes.compression.level;
                structure.indexing_genes.compression.level = fastrand::u8(1..10);

                structure.mutations.push(MutationType::StorageMutation {
                    parameter: "compression_level".to_string(),
                    old_value: old_level as f64,
                    new_value: structure.indexing_genes.compression.level as f64,
                });
            }
            "cache_size" => {
                let old_size = structure.storage_genes.caching.cache_size_mb;
                structure.storage_genes.caching.cache_size_mb = 64 + fastrand::u32(..1936);

                structure.mutations.push(MutationType::StorageMutation {
                    parameter: "cache_size_mb".to_string(),
                    old_value: old_size as f64,
                    new_value: structure.storage_genes.caching.cache_size_mb as f64,
                });
            }
            _ => {}
        }

        Ok(())
    }

    /// Calculate genetic diversity of population
    fn calculate_diversity(&self) -> f64 {
        // Simple diversity measure based on fitness variance
        let avg_fitness =
            self.population.iter().map(|s| s.fitness).sum::<f64>() / self.population.len() as f64;
        let variance = self
            .population
            .iter()
            .map(|s| (s.fitness - avg_fitness).powi(2))
            .sum::<f64>()
            / self.population.len() as f64;

        variance.sqrt()
    }

    /// Run complete evolution process
    pub fn evolve(&mut self) -> Result<GraphStructure, OxirsError> {
        for generation in 0..self.generations {
            let stats = self.evolve_generation()?;

            // Print progress every 10 generations
            if generation % 10 == 0 {
                println!(
                    "Generation {}: Best={:.4}, Avg={:.4}, Diversity={:.4}",
                    stats.generation, stats.best_fitness, stats.average_fitness, stats.diversity
                );
            }

            // Early termination if no improvement for many generations
            if generation > 50 && self.evolution_history.len() >= 20 {
                let recent_best = self
                    .evolution_history
                    .iter()
                    .rev()
                    .take(20)
                    .map(|s| s.best_fitness)
                    .fold(0.0, f64::max);

                if (recent_best - self.best_fitness).abs() < 0.001 {
                    println!("Early termination: No improvement in 20 generations");
                    break;
                }
            }
        }

        Ok(self.population[0].clone())
    }

    /// Get the best structure from current population
    pub fn get_best_structure(&self) -> Option<&GraphStructure> {
        self.population.first()
    }

    /// Get evolution history
    pub fn get_evolution_history(&self) -> &[GenerationStats] {
        &self.evolution_history
    }
}

/// Default fitness function for graph structures
pub fn default_fitness_function(structure: &GraphStructure) -> f64 {
    let mut fitness = 0.0;

    // Reward efficient indexing
    fitness += structure.indexing_genes.secondary_indexes.len() as f64 * 0.1;

    // Reward optimal cache size (not too small, not too large)
    let cache_mb = structure.storage_genes.caching.cache_size_mb as f64;
    fitness += if cache_mb >= 128.0 && cache_mb <= 512.0 {
        0.2
    } else {
        0.0
    };

    // Reward balanced concurrency settings
    let readers = structure.access_genes.concurrency.max_readers as f64;
    let writers = structure.access_genes.concurrency.max_writers as f64;
    fitness += if readers >= 10.0 && writers >= 2.0 && readers / writers <= 50.0 {
        0.2
    } else {
        0.0
    };

    // Reward good compression settings
    let compression_level = structure.indexing_genes.compression.level as f64;
    fitness += if compression_level >= 3.0 && compression_level <= 7.0 {
        0.1
    } else {
        0.0
    };

    // Reward reasonable block sizes
    let block_size = structure.storage_genes.block_size as f64;
    fitness += if block_size >= 8192.0 && block_size <= 65536.0 {
        0.1
    } else {
        0.0
    };

    // Add some randomness to encourage exploration
    fitness += fastrand::f64() * 0.1;

    fitness
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[test]
    fn test_genetic_optimizer_creation() {
        let fitness_fn = Box::new(default_fitness_function);
        let optimizer = GeneticGraphOptimizer::new(20, fitness_fn);

        assert_eq!(optimizer.population_size, 20);
        assert_eq!(optimizer.elite_size, 2);
    }

    #[test]
    fn test_random_structure_generation() {
        let fitness_fn = Box::new(default_fitness_function);
        let optimizer = GeneticGraphOptimizer::new(10, fitness_fn);

        let triples = vec![Triple::new(
            NamedNode::new("http://example.org/subject").unwrap(),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Literal::new("object"),
        )];

        let structure = optimizer.create_random_structure(&triples);
        assert!(structure.is_ok());

        let structure = structure.unwrap();
        assert!(structure.fitness >= 0.0);
        assert!(!structure.indexing_genes.secondary_indexes.is_empty());
    }

    #[test]
    fn test_mutation() {
        let fitness_fn = Box::new(default_fitness_function);
        let optimizer = GeneticGraphOptimizer::new(10, fitness_fn);

        let triples = vec![Triple::new(
            NamedNode::new("http://example.org/subject").unwrap(),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Literal::new("object"),
        )];

        let mut structure = optimizer.create_random_structure(&triples).unwrap();
        let _old_block_size = structure.storage_genes.block_size;

        optimizer.mutate(&mut structure).unwrap();

        // Structure should have some changes
        assert!(!structure.mutations.is_empty());
    }

    #[test]
    fn test_fitness_function() {
        let fitness_fn = Box::new(default_fitness_function);
        let optimizer = GeneticGraphOptimizer::new(10, fitness_fn);

        let triples = vec![Triple::new(
            NamedNode::new("http://example.org/subject").unwrap(),
            NamedNode::new("http://example.org/predicate").unwrap(),
            Literal::new("object"),
        )];

        let structure = optimizer.create_random_structure(&triples).unwrap();
        let fitness = default_fitness_function(&structure);

        assert!(fitness >= 0.0);
        assert!(fitness <= 1.0); // Should be roughly in this range for default function
    }
}
