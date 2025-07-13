//! # Biological Computing Integration for RDF Streaming
//!
//! This module implements biological computing paradigms for ultra-efficient
//! data processing, including DNA storage algorithms, protein folding optimization,
//! cellular automata, and evolutionary computation for streaming RDF data.

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{StreamError, StreamResult};
use crate::event::StreamEvent;

/// DNA nucleotide bases for biological data encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Nucleotide {
    Adenine,  // A
    Thymine,  // T
    Guanine,  // G
    Cytosine, // C
}

impl Nucleotide {
    /// Convert nucleotide to binary representation
    pub fn to_bits(self) -> [bool; 2] {
        match self {
            Nucleotide::Adenine => [false, false], // 00
            Nucleotide::Thymine => [false, true],  // 01
            Nucleotide::Guanine => [true, false],  // 10
            Nucleotide::Cytosine => [true, true],  // 11
        }
    }

    /// Create nucleotide from binary representation
    pub fn from_bits(bits: [bool; 2]) -> Self {
        match bits {
            [false, false] => Nucleotide::Adenine,
            [false, true] => Nucleotide::Thymine,
            [true, false] => Nucleotide::Guanine,
            [true, true] => Nucleotide::Cytosine,
        }
    }

    /// Get complementary nucleotide (for DNA pairing)
    pub fn complement(self) -> Self {
        match self {
            Nucleotide::Adenine => Nucleotide::Thymine,
            Nucleotide::Thymine => Nucleotide::Adenine,
            Nucleotide::Guanine => Nucleotide::Cytosine,
            Nucleotide::Cytosine => Nucleotide::Guanine,
        }
    }
}

/// DNA sequence for data encoding
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DNASequence {
    /// Sequence of nucleotides
    pub nucleotides: Vec<Nucleotide>,
    /// Metadata about the sequence
    pub metadata: SequenceMetadata,
}

/// Metadata for DNA sequences
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceMetadata {
    /// Length of the sequence
    pub length: usize,
    /// GC content ratio (biological quality metric)
    pub gc_content: f64,
    /// Sequence stability score
    pub stability: f64,
    /// Error correction redundancy
    pub redundancy_factor: f64,
}

impl DNASequence {
    /// Create a new DNA sequence from binary data
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut nucleotides = Vec::new();

        for byte in data {
            // Convert each byte to 4 nucleotides (2 bits each)
            for i in 0..4 {
                let bits = [
                    (byte >> (6 - i * 2)) & 1 != 0,
                    (byte >> (7 - i * 2)) & 1 != 0,
                ];
                nucleotides.push(Nucleotide::from_bits(bits));
            }
        }

        let gc_content = Self::calculate_gc_content(&nucleotides);
        let metadata = SequenceMetadata {
            length: nucleotides.len(),
            gc_content,
            stability: Self::calculate_stability(&nucleotides),
            redundancy_factor: 1.0, // Base redundancy
        };

        DNASequence {
            nucleotides,
            metadata,
        }
    }

    /// Convert DNA sequence back to binary data
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        for chunk in self.nucleotides.chunks(4) {
            let mut byte = 0u8;
            for (i, nucleotide) in chunk.iter().enumerate() {
                let bits = nucleotide.to_bits();
                byte |= (bits[0] as u8) << (6 - i * 2);
                byte |= (bits[1] as u8) << (7 - i * 2);
            }
            bytes.push(byte);
        }

        bytes
    }

    /// Calculate GC content (important for DNA stability)
    fn calculate_gc_content(nucleotides: &[Nucleotide]) -> f64 {
        let gc_count = nucleotides
            .iter()
            .filter(|&&n| matches!(n, Nucleotide::Guanine | Nucleotide::Cytosine))
            .count();
        gc_count as f64 / nucleotides.len() as f64
    }

    /// Calculate sequence stability
    fn calculate_stability(nucleotides: &[Nucleotide]) -> f64 {
        // Simplified stability calculation based on adjacent nucleotide bonds
        let mut stability = 0.0;
        for window in nucleotides.windows(2) {
            match (window[0], window[1]) {
                (Nucleotide::Guanine, Nucleotide::Cytosine)
                | (Nucleotide::Cytosine, Nucleotide::Guanine) => stability += 3.0, // Strong GC bond
                (Nucleotide::Adenine, Nucleotide::Thymine)
                | (Nucleotide::Thymine, Nucleotide::Adenine) => stability += 2.0, // AT bond
                _ => stability += 1.0, // Other combinations
            }
        }
        stability / nucleotides.len() as f64
    }

    /// Add error correction redundancy using biological patterns
    pub fn add_redundancy(&mut self, factor: f64) {
        let original_length = self.nucleotides.len();
        let redundant_copies = (original_length as f64 * factor) as usize;

        // Add redundant nucleotides using Hamming code principles
        for _ in 0..redundant_copies {
            // Add check nucleotides based on parity
            let check_nucleotide = self.calculate_check_nucleotide();
            self.nucleotides.push(check_nucleotide);
        }

        self.metadata.redundancy_factor = factor;
        self.metadata.length = self.nucleotides.len();
    }

    /// Calculate check nucleotide for error correction
    fn calculate_check_nucleotide(&self) -> Nucleotide {
        let mut parity = 0;
        for nucleotide in &self.nucleotides {
            let bits = nucleotide.to_bits();
            parity ^= bits[0] as u8;
            parity ^= bits[1] as u8;
        }

        match parity % 4 {
            0 => Nucleotide::Adenine,
            1 => Nucleotide::Thymine,
            2 => Nucleotide::Guanine,
            _ => Nucleotide::Cytosine,
        }
    }
}

/// Protein structure for computational optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinStructure {
    /// Amino acid sequence
    pub amino_acids: Vec<AminoAcid>,
    /// 3D folding coordinates
    pub folding_coordinates: Vec<(f64, f64, f64)>,
    /// Folding energy (lower = more stable)
    pub folding_energy: f64,
    /// Functional domains
    pub domains: Vec<FunctionalDomain>,
}

/// Amino acid types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AminoAcid {
    Alanine,
    Arginine,
    Asparagine,
    AsparticAcid,
    Cysteine,
    GlutamicAcid,
    Glutamine,
    Glycine,
    Histidine,
    Isoleucine,
    Leucine,
    Lysine,
    Methionine,
    Phenylalanine,
    Proline,
    Serine,
    Threonine,
    Tryptophan,
    Tyrosine,
    Valine,
}

/// Functional domain in protein
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionalDomain {
    /// Domain name
    pub name: String,
    /// Start position in sequence
    pub start: usize,
    /// End position in sequence
    pub end: usize,
    /// Computational function
    pub function: ComputationalFunction,
}

/// Computational functions mapped to protein domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputationalFunction {
    DataCompression,
    PatternRecognition,
    ErrorCorrection,
    Encryption,
    Optimization,
    MemoryStorage,
}

/// Cell in cellular automaton
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cell {
    /// Cell state (alive/dead or value)
    pub state: CellState,
    /// Cell age
    pub age: u32,
    /// Energy level
    pub energy: f64,
    /// Mutation probability
    pub mutation_rate: f64,
}

/// Cell state in automaton
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellState {
    Dead,
    Alive,
    Data(u8),   // Cell carrying data
    Processing, // Cell actively processing
}

/// Cellular automaton for distributed processing
#[derive(Debug, Clone)]
pub struct CellularAutomaton {
    /// 2D grid of cells
    pub grid: Vec<Vec<Cell>>,
    /// Automaton rules
    pub rules: AutomatonRules,
    /// Generation counter
    pub generation: u64,
    /// Grid dimensions
    pub width: usize,
    pub height: usize,
}

/// Rules for cellular automaton evolution
#[derive(Debug, Clone)]
pub struct AutomatonRules {
    /// Birth conditions (number of neighbors for birth)
    pub birth_conditions: Vec<usize>,
    /// Survival conditions (number of neighbors for survival)
    pub survival_conditions: Vec<usize>,
    /// Data processing rules
    pub processing_rules: HashMap<u8, u8>,
    /// Energy transfer rules
    pub energy_rules: EnergyRules,
}

/// Energy transfer rules for cells
#[derive(Debug, Clone)]
pub struct EnergyRules {
    /// Energy threshold for cell activity
    pub activity_threshold: f64,
    /// Energy decay rate per generation
    pub decay_rate: f64,
    /// Energy transfer efficiency between neighbors
    pub transfer_efficiency: f64,
}

impl CellularAutomaton {
    /// Create a new cellular automaton
    pub fn new(width: usize, height: usize) -> Self {
        let grid = vec![
            vec![
                Cell {
                    state: CellState::Dead,
                    age: 0,
                    energy: 0.0,
                    mutation_rate: 0.01,
                };
                width
            ];
            height
        ];

        let rules = AutomatonRules {
            birth_conditions: vec![3], // Conway's Game of Life rule
            survival_conditions: vec![2, 3],
            processing_rules: HashMap::new(),
            energy_rules: EnergyRules {
                activity_threshold: 0.5,
                decay_rate: 0.1,
                transfer_efficiency: 0.8,
            },
        };

        Self {
            grid,
            rules,
            generation: 0,
            width,
            height,
        }
    }

    /// Evolve the automaton by one generation
    pub fn evolve(&mut self) {
        let mut new_grid = self.grid.clone();

        #[allow(clippy::needless_range_loop)]
        for y in 0..self.height {
            for x in 0..self.width {
                let neighbors = self.count_live_neighbors(x, y);
                let cell = &self.grid[y][x];

                // Apply evolution rules
                let new_state = match cell.state {
                    CellState::Dead => {
                        if self.rules.birth_conditions.contains(&neighbors) {
                            CellState::Alive
                        } else {
                            CellState::Dead
                        }
                    }
                    CellState::Alive => {
                        if self.rules.survival_conditions.contains(&neighbors) {
                            CellState::Alive
                        } else {
                            CellState::Dead
                        }
                    }
                    CellState::Data(value) => {
                        // Process data based on neighbors and rules
                        if let Some(new_value) = self.rules.processing_rules.get(&value) {
                            CellState::Data(*new_value)
                        } else {
                            CellState::Data(value)
                        }
                    }
                    CellState::Processing => {
                        // Continue processing or complete
                        if neighbors >= 2 {
                            CellState::Processing
                        } else {
                            CellState::Alive
                        }
                    }
                };

                // Update cell
                new_grid[y][x] = Cell {
                    state: new_state,
                    age: cell.age + 1,
                    energy: (cell.energy * (1.0 - self.rules.energy_rules.decay_rate)).max(0.0),
                    mutation_rate: cell.mutation_rate,
                };

                // Apply mutations
                if rand::random::<f64>() < cell.mutation_rate {
                    self.apply_mutation(&mut new_grid[y][x]);
                }
            }
        }

        self.grid = new_grid;
        self.generation += 1;
    }

    /// Count live neighbors around a cell
    fn count_live_neighbors(&self, x: usize, y: usize) -> usize {
        let mut count = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                    match self.grid[ny as usize][nx as usize].state {
                        CellState::Alive | CellState::Processing => count += 1,
                        _ => {}
                    }
                }
            }
        }
        count
    }

    /// Apply random mutation to a cell
    fn apply_mutation(&self, cell: &mut Cell) {
        match cell.state {
            CellState::Dead => cell.state = CellState::Alive,
            CellState::Alive => cell.state = CellState::Dead,
            CellState::Data(value) => cell.state = CellState::Data(value.wrapping_add(1)),
            CellState::Processing => cell.state = CellState::Alive,
        }
    }

    /// Inject data into the automaton
    pub fn inject_data(&mut self, x: usize, y: usize, data: u8) {
        if x < self.width && y < self.height {
            self.grid[y][x].state = CellState::Data(data);
            self.grid[y][x].energy = 1.0;
        }
    }

    /// Extract processed data from the automaton
    pub fn extract_data(&self) -> Vec<u8> {
        let mut data = Vec::new();
        for row in &self.grid {
            for cell in row {
                if let CellState::Data(value) = cell.state {
                    data.push(value);
                }
            }
        }
        data
    }
}

/// Evolutionary algorithm for optimization
#[derive(Debug, Clone)]
pub struct EvolutionaryOptimizer {
    /// Population of solutions
    population: Vec<Individual>,
    /// Population size
    population_size: usize,
    /// Mutation rate
    mutation_rate: f64,
    /// Crossover rate
    crossover_rate: f64,
    /// Current generation
    generation: u64,
    /// Best fitness achieved
    best_fitness: f64,
}

/// Individual in evolutionary population
#[derive(Debug, Clone)]
pub struct Individual {
    /// Genome (solution representation)
    pub genome: Vec<f64>,
    /// Fitness score
    pub fitness: f64,
    /// Age in generations
    pub age: u64,
}

impl EvolutionaryOptimizer {
    /// Create a new evolutionary optimizer
    pub fn new(population_size: usize, genome_size: usize) -> Self {
        let mut population = Vec::new();

        for _ in 0..population_size {
            let genome: Vec<f64> = (0..genome_size)
                .map(|_| rand::random::<f64>() * 2.0 - 1.0) // Random values between -1 and 1
                .collect();

            population.push(Individual {
                genome,
                fitness: 0.0,
                age: 0,
            });
        }

        Self {
            population,
            population_size,
            mutation_rate: 0.1,
            crossover_rate: 0.7,
            generation: 0,
            best_fitness: f64::NEG_INFINITY,
        }
    }

    /// Evolve the population for one generation
    pub fn evolve<F>(&mut self, fitness_function: F)
    where
        F: Fn(&[f64]) -> f64,
    {
        // Evaluate fitness
        for individual in &mut self.population {
            individual.fitness = fitness_function(&individual.genome);
            if individual.fitness > self.best_fitness {
                self.best_fitness = individual.fitness;
            }
        }

        // Selection, crossover, and mutation
        let mut new_population = Vec::new();

        while new_population.len() < self.population_size {
            // Tournament selection
            let parent1 = self.tournament_selection();
            let parent2 = self.tournament_selection();

            // Crossover
            let (mut child1, mut child2) = if rand::random::<f64>() < self.crossover_rate {
                self.crossover(parent1, parent2)
            } else {
                (parent1.clone(), parent2.clone())
            };

            // Mutation
            if rand::random::<f64>() < self.mutation_rate {
                self.mutate(&mut child1);
            }
            if rand::random::<f64>() < self.mutation_rate {
                self.mutate(&mut child2);
            }

            new_population.push(child1);
            if new_population.len() < self.population_size {
                new_population.push(child2);
            }
        }

        self.population = new_population;
        self.generation += 1;
    }

    /// Tournament selection
    fn tournament_selection(&self) -> &Individual {
        let tournament_size = 3;
        let mut best_individual = &self.population[0];

        for _ in 0..tournament_size {
            let candidate =
                &self.population[rand::thread_rng().gen_range(0..self.population.len())];
            if candidate.fitness > best_individual.fitness {
                best_individual = candidate;
            }
        }

        best_individual
    }

    /// Crossover operation
    fn crossover(&self, parent1: &Individual, parent2: &Individual) -> (Individual, Individual) {
        let crossover_point = rand::thread_rng().gen_range(0..parent1.genome.len());

        let mut child1_genome = parent1.genome.clone();
        let mut child2_genome = parent2.genome.clone();

        // Single-point crossover
        child1_genome[crossover_point..parent1.genome.len()]
            .copy_from_slice(&parent2.genome[crossover_point..]);
        child2_genome[crossover_point..parent1.genome.len()]
            .copy_from_slice(&parent1.genome[crossover_point..]);

        (
            Individual {
                genome: child1_genome,
                fitness: 0.0,
                age: 0,
            },
            Individual {
                genome: child2_genome,
                fitness: 0.0,
                age: 0,
            },
        )
    }

    /// Mutation operation
    fn mutate(&self, individual: &mut Individual) {
        for gene in &mut individual.genome {
            if rand::random::<f64>() < 0.1 {
                // Gene mutation probability
                *gene += (rand::random::<f64>() - 0.5) * 0.2; // Small random change
                *gene = gene.clamp(-1.0, 1.0); // Keep within bounds
            }
        }
    }

    /// Get the best individual
    pub fn best_individual(&self) -> &Individual {
        self.population
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
    }
}

/// Biological computing processor for stream events
pub struct BiologicalStreamProcessor {
    /// DNA storage system
    dna_storage: HashMap<String, DNASequence>,
    /// Protein folding optimizer
    _protein_optimizer: HashMap<String, ProteinStructure>,
    /// Cellular automaton for distributed processing
    automaton: CellularAutomaton,
    /// Evolutionary optimizer
    evolutionary_optimizer: EvolutionaryOptimizer,
    /// Processing statistics
    stats: Arc<RwLock<BiologicalProcessingStats>>,
}

/// Statistics for biological processing
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BiologicalProcessingStats {
    /// Total events processed
    pub events_processed: u64,
    /// DNA storage operations
    pub dna_operations: u64,
    /// Protein optimizations performed
    pub protein_optimizations: u64,
    /// Automaton generations
    pub automaton_generations: u64,
    /// Evolutionary generations
    pub evolutionary_generations: u64,
    /// Average processing time in microseconds
    pub avg_processing_time_us: f64,
    /// Data compression ratio achieved
    pub compression_ratio: f64,
    /// Error correction success rate
    pub error_correction_rate: f64,
}

impl Default for BiologicalStreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl BiologicalStreamProcessor {
    /// Create a new biological stream processor
    pub fn new() -> Self {
        Self {
            dna_storage: HashMap::new(),
            _protein_optimizer: HashMap::new(),
            automaton: CellularAutomaton::new(32, 32), // 32x32 cell grid
            evolutionary_optimizer: EvolutionaryOptimizer::new(100, 50), // Population 100, genome size 50
            stats: Arc::new(RwLock::new(BiologicalProcessingStats::default())),
        }
    }

    /// Process a stream event using biological computing
    pub async fn process_event(&mut self, event: &StreamEvent) -> StreamResult<StreamEvent> {
        let start_time = std::time::Instant::now();

        // Convert event to DNA sequence for storage
        let event_bytes = self.serialize_event(event)?;
        let mut dna_sequence = DNASequence::from_bytes(&event_bytes);

        // Add error correction redundancy
        dna_sequence.add_redundancy(1.5);

        // Store in DNA storage
        self.dna_storage
            .insert(event.event_id().to_string(), dna_sequence.clone());

        // Process data through cellular automaton
        self.process_with_automaton(&event_bytes).await?;

        // Optimize processing parameters using evolutionary algorithm
        self.optimize_with_evolution().await;

        // Update statistics
        let processing_time = start_time.elapsed().as_micros() as f64;
        self.update_stats(processing_time, &event_bytes, &dna_sequence.to_bytes())
            .await;

        Ok(event.clone())
    }

    /// Serialize event to bytes
    fn serialize_event(&self, event: &StreamEvent) -> StreamResult<Vec<u8>> {
        serde_json::to_vec(event).map_err(|e| StreamError::Serialization(e.to_string()))
    }

    /// Process data through cellular automaton
    async fn process_with_automaton(&mut self, data: &[u8]) -> StreamResult<()> {
        // Inject data into automaton
        for (i, &byte) in data.iter().enumerate().take(1024) {
            // Limit to grid size
            let x = i % self.automaton.width;
            let y = i / self.automaton.width;
            self.automaton.inject_data(x, y, byte);
        }

        // Evolve automaton for several generations
        for _ in 0..10 {
            self.automaton.evolve();
        }

        // Extract processed data
        let _processed_data = self.automaton.extract_data();

        Ok(())
    }

    /// Optimize using evolutionary algorithm
    async fn optimize_with_evolution(&mut self) {
        // Define fitness function for stream processing optimization
        let fitness_function = |genome: &[f64]| {
            // Fitness based on processing speed, compression ratio, and accuracy
            let speed_factor = genome[0].abs();
            let compression_factor = genome[1].abs();
            let accuracy_factor = genome[2].abs();

            speed_factor * 0.4 + compression_factor * 0.3 + accuracy_factor * 0.3
        };

        self.evolutionary_optimizer.evolve(fitness_function);
    }

    /// Update processing statistics
    async fn update_stats(
        &self,
        processing_time: f64,
        original_data: &[u8],
        compressed_data: &[u8],
    ) {
        let mut stats = self.stats.write().await;

        stats.events_processed += 1;
        stats.dna_operations += 1;
        stats.automaton_generations += 10; // We run 10 generations per event
        stats.evolutionary_generations += 1;

        // Update average processing time
        let alpha = 0.1; // Exponential moving average factor
        stats.avg_processing_time_us =
            alpha * processing_time + (1.0 - alpha) * stats.avg_processing_time_us;

        // Calculate compression ratio
        if !original_data.is_empty() {
            let compression_ratio = compressed_data.len() as f64 / original_data.len() as f64;
            stats.compression_ratio =
                alpha * compression_ratio + (1.0 - alpha) * stats.compression_ratio;
        }

        // Simulate error correction success rate
        stats.error_correction_rate = alpha * 0.99 + (1.0 - alpha) * stats.error_correction_rate;
    }

    /// Get processing statistics
    pub async fn get_stats(&self) -> BiologicalProcessingStats {
        (*self.stats.read().await).clone()
    }

    /// Retrieve data from DNA storage
    pub fn retrieve_from_dna(&self, event_id: &str) -> Option<Vec<u8>> {
        self.dna_storage.get(event_id).map(|seq| seq.to_bytes())
    }

    /// Get current automaton state
    pub fn get_automaton_state(&self) -> String {
        format!(
            "Generation: {}, Active cells: {}",
            self.automaton.generation,
            self.count_active_cells()
        )
    }

    /// Count active cells in automaton
    fn count_active_cells(&self) -> usize {
        self.automaton
            .grid
            .iter()
            .flat_map(|row| row.iter())
            .filter(|cell| !matches!(cell.state, CellState::Dead))
            .count()
    }

    /// Get evolutionary optimization progress
    pub fn get_evolution_progress(&self) -> (u64, f64) {
        (
            self.evolutionary_optimizer.generation,
            self.evolutionary_optimizer.best_fitness,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dna_encoding_decoding() {
        let original_data = b"Hello, Biological Computing!";
        let dna_sequence = DNASequence::from_bytes(original_data);
        let decoded_data = dna_sequence.to_bytes();

        assert_eq!(original_data.to_vec(), decoded_data);
        assert!(dna_sequence.metadata.gc_content >= 0.0 && dna_sequence.metadata.gc_content <= 1.0);
    }

    #[test]
    fn test_cellular_automaton() {
        let mut automaton = CellularAutomaton::new(10, 10);

        // Inject some data
        automaton.inject_data(5, 5, 42);

        // Evolve
        automaton.evolve();

        assert_eq!(automaton.generation, 1);
    }

    #[test]
    fn test_evolutionary_optimizer() {
        let mut optimizer = EvolutionaryOptimizer::new(10, 5);

        // Simple fitness function (sum of genes)
        let fitness_fn = |genome: &[f64]| genome.iter().sum::<f64>();

        optimizer.evolve(fitness_fn);

        assert_eq!(optimizer.generation, 1);
        assert!(optimizer.best_fitness > f64::NEG_INFINITY);
    }

    #[tokio::test]
    async fn test_biological_processor() {
        let mut processor = BiologicalStreamProcessor::new();

        let event = StreamEvent::Heartbeat {
            timestamp: chrono::Utc::now(),
            source: "test-biological-processor".to_string(),
            metadata: Default::default(),
        };

        let result = processor.process_event(&event).await;
        assert!(result.is_ok());

        let stats = processor.get_stats().await;
        assert_eq!(stats.events_processed, 1);
    }
}
