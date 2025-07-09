//! Streaming Execution and Spilling Support
//!
//! This module provides streaming execution capabilities for large result sets
//! and memory-efficient spilling strategies for operations that exceed memory limits.

use crate::algebra::{Binding, Solution, Term, Variable};
use anyhow::{anyhow, Result};
use oxirs_core::model::NamedNode;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{remove_file, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tempfile::NamedTempFile;

/// Configuration for streaming and spilling operations
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum memory usage before spilling (in bytes)
    pub memory_limit: usize,
    /// Temporary directory for spill files
    pub temp_dir: Option<PathBuf>,
    /// Maximum number of solutions to buffer
    pub buffer_size: usize,
    /// Enable compression for spill files
    pub compress_spills: bool,
    /// Spilling strategy
    pub spill_strategy: SpillStrategy,
    /// Enable adaptive buffering
    pub adaptive_buffering: bool,
    /// Enable parallel spilling
    pub parallel_spilling: bool,
    /// Compression algorithm choice
    pub compression_algorithm: CompressionAlgorithm,
}

/// Spilling strategy options
#[derive(Debug, Clone)]
pub enum SpillStrategy {
    /// Spill oldest data first (FIFO)
    Fifo,
    /// Spill largest chunks first
    LargestFirst,
    /// Spill based on access frequency (LRU)
    LeastRecentlyUsed,
    /// Adaptive strategy based on workload
    Adaptive,
}

/// Compression algorithm options
#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Fast LZ4 compression
    Lz4,
    /// Balanced gzip compression
    Gzip,
    /// High compression zstd
    Zstd,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            memory_limit: 1024 * 1024 * 1024, // 1GB default
            temp_dir: None,
            buffer_size: 10000,
            compress_spills: true,
            spill_strategy: SpillStrategy::Adaptive,
            adaptive_buffering: true,
            parallel_spilling: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
        }
    }
}

/// Memory usage tracker with adaptive management
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    current_usage: Arc<Mutex<usize>>,
    peak_usage: Arc<Mutex<usize>>,
    limit: usize,
    allocation_history: Arc<Mutex<Vec<AllocationEvent>>>,
    pressure_threshold: f64,
    prediction_window: usize,
}

/// Allocation event for tracking patterns
#[derive(Debug, Clone)]
struct AllocationEvent {
    timestamp: std::time::Instant,
    size: usize,
    operation: AllocationType,
}

#[derive(Debug, Clone)]
enum AllocationType {
    Allocate,
    Deallocate,
}

impl MemoryTracker {
    pub fn new(limit: usize) -> Self {
        Self {
            current_usage: Arc::new(Mutex::new(0)),
            peak_usage: Arc::new(Mutex::new(0)),
            limit,
            allocation_history: Arc::new(Mutex::new(Vec::new())),
            pressure_threshold: 0.8, // Start adaptive behavior at 80%
            prediction_window: 100,  // Track last 100 allocations
        }
    }

    /// Create tracker with custom pressure threshold
    pub fn with_pressure_threshold(limit: usize, threshold: f64) -> Self {
        Self {
            current_usage: Arc::new(Mutex::new(0)),
            peak_usage: Arc::new(Mutex::new(0)),
            limit,
            allocation_history: Arc::new(Mutex::new(Vec::new())),
            pressure_threshold: threshold,
            prediction_window: 100,
        }
    }

    pub fn allocate(&self, size: usize) -> Result<bool> {
        let mut current = self.current_usage.lock().unwrap();
        let new_usage = *current + size;

        // Record allocation attempt
        self.record_allocation_event(size, AllocationType::Allocate);

        if new_usage > self.limit {
            return Ok(false); // Cannot allocate, need to spill
        }

        *current = new_usage;

        let mut peak = self.peak_usage.lock().unwrap();
        if new_usage > *peak {
            *peak = new_usage;
        }

        Ok(true)
    }

    pub fn deallocate(&self, size: usize) {
        let mut current = self.current_usage.lock().unwrap();
        *current = current.saturating_sub(size);

        // Record deallocation
        self.record_allocation_event(size, AllocationType::Deallocate);
    }

    /// Record allocation event for pattern analysis
    fn record_allocation_event(&self, size: usize, operation: AllocationType) {
        let mut history = self.allocation_history.lock().unwrap();

        history.push(AllocationEvent {
            timestamp: std::time::Instant::now(),
            size,
            operation,
        });

        // Keep only recent events
        let history_len = history.len();
        if history_len > self.prediction_window {
            history.drain(0..history_len - self.prediction_window);
        }
    }

    pub fn current_usage(&self) -> usize {
        *self.current_usage.lock().unwrap()
    }

    pub fn peak_usage(&self) -> usize {
        *self.peak_usage.lock().unwrap()
    }

    pub fn should_spill(&self) -> bool {
        let usage_ratio = self.current_usage() as f64 / self.limit as f64;
        usage_ratio > self.pressure_threshold
    }

    /// Adaptive spilling decision based on allocation patterns
    pub fn should_spill_adaptive(&self) -> bool {
        let current_ratio = self.current_usage() as f64 / self.limit as f64;

        // Basic threshold check
        if current_ratio > self.pressure_threshold {
            return true;
        }

        // Check allocation velocity (rate of growth)
        let allocation_velocity = self.calculate_allocation_velocity();
        let predicted_usage = self.predict_memory_usage(allocation_velocity);

        // Spill early if we predict memory pressure
        if predicted_usage > self.limit as f64 * 0.9 {
            return true;
        }

        false
    }

    /// Calculate recent allocation velocity (bytes per second)
    fn calculate_allocation_velocity(&self) -> f64 {
        let history = self.allocation_history.lock().unwrap();

        if history.len() < 2 {
            return 0.0;
        }

        let now = std::time::Instant::now();
        let window_duration = std::time::Duration::from_secs(5); // 5 second window

        let mut net_allocation = 0i64;
        let mut oldest_timestamp = now;

        for event in history.iter().rev() {
            if now.duration_since(event.timestamp) > window_duration {
                break;
            }

            match event.operation {
                AllocationType::Allocate => net_allocation += event.size as i64,
                AllocationType::Deallocate => net_allocation -= event.size as i64,
            }

            oldest_timestamp = event.timestamp;
        }

        let elapsed = now.duration_since(oldest_timestamp).as_secs_f64();
        if elapsed > 0.0 {
            net_allocation as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Predict memory usage based on current velocity
    fn predict_memory_usage(&self, velocity: f64) -> f64 {
        let current = self.current_usage() as f64;
        let prediction_horizon = 2.0; // 2 seconds ahead

        current + (velocity * prediction_horizon).max(0.0)
    }

    /// Get detailed memory statistics
    pub fn get_detailed_stats(&self) -> MemoryStats {
        let history = self.allocation_history.lock().unwrap();

        let total_allocations = history
            .iter()
            .filter(|e| matches!(e.operation, AllocationType::Allocate))
            .count();

        let total_deallocations = history
            .iter()
            .filter(|e| matches!(e.operation, AllocationType::Deallocate))
            .count();

        let avg_allocation_size = if total_allocations > 0 {
            history
                .iter()
                .filter(|e| matches!(e.operation, AllocationType::Allocate))
                .map(|e| e.size)
                .sum::<usize>()
                / total_allocations
        } else {
            0
        };

        MemoryStats {
            current_usage: self.current_usage(),
            peak_usage: self.peak_usage(),
            total_allocations,
            total_deallocations,
            avg_allocation_size,
            allocation_velocity: self.calculate_allocation_velocity(),
            pressure_ratio: self.current_usage() as f64 / self.limit as f64,
        }
    }

    /// Adaptive pressure threshold based on workload
    pub fn adjust_pressure_threshold(&mut self, workload_intensity: f64) {
        // Lower threshold for high-intensity workloads to spill earlier
        self.pressure_threshold = (0.6 + (0.3 * (1.0 - workload_intensity))).clamp(0.5, 0.9);
    }
}

/// Detailed memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub total_allocations: usize,
    pub total_deallocations: usize,
    pub avg_allocation_size: usize,
    pub allocation_velocity: f64,
    pub pressure_ratio: f64,
}

/// Streaming solution iterator with memory management
pub struct StreamingSolution {
    solutions: VecDeque<Solution>,
    spill_files: Vec<SpillFile>,
    current_spill_idx: usize,
    memory_tracker: MemoryTracker,
    config: StreamingConfig,
    finished: bool,
}

/// Spill file for temporary storage
#[derive(Debug)]
struct SpillFile {
    path: PathBuf,
    size: usize,
    compressed: bool,
}

impl StreamingSolution {
    pub fn new(config: StreamingConfig) -> Self {
        let memory_tracker = MemoryTracker::new(config.memory_limit);

        Self {
            solutions: VecDeque::new(),
            spill_files: Vec::new(),
            current_spill_idx: 0,
            memory_tracker,
            config,
            finished: false,
        }
    }

    /// Add a solution to the stream
    pub fn add_solution(&mut self, solution: Solution) -> Result<()> {
        let solution_size = self.estimate_solution_size(&solution);

        if !self.memory_tracker.allocate(solution_size)? {
            // Need to spill to disk
            self.spill_to_disk()?;
            // Try again after spilling
            if !self.memory_tracker.allocate(solution_size)? {
                return Err(anyhow!("Cannot allocate memory even after spilling"));
            }
        }

        self.solutions.push_back(solution);

        // Check if we need to spill proactively using adaptive strategy
        let should_spill = if self.config.adaptive_buffering {
            self.memory_tracker.should_spill_adaptive()
        } else {
            self.memory_tracker.should_spill()
        };

        if self.solutions.len() >= self.config.buffer_size || should_spill {
            self.spill_to_disk()?;
        }

        Ok(())
    }

    /// Estimate the memory size of a solution
    fn estimate_solution_size(&self, solution: &Solution) -> usize {
        let mut size = std::mem::size_of::<Solution>();
        for binding in solution {
            size += binding
                .iter()
                .map(|(var, term)| var.as_str().len() + self.estimate_term_size(term))
                .sum::<usize>();
        }
        size
    }

    /// Estimate the memory size of a term
    #[allow(clippy::only_used_in_recursion)]
    fn estimate_term_size(&self, term: &Term) -> usize {
        match term {
            Term::Iri(iri) => iri.as_str().len(),
            Term::Literal(lit) => lit.value.len() + lit.language.as_ref().map_or(0, |l| l.len()),
            Term::BlankNode(bn) => bn.len(),
            Term::Variable(var) => var.as_str().len(),
            Term::QuotedTriple(triple) => {
                // Estimate size of quoted triple as sum of its parts
                self.estimate_term_size(&triple.subject)
                    + self.estimate_term_size(&triple.predicate)
                    + self.estimate_term_size(&triple.object)
                    + 6 // << >> brackets
            }
            Term::PropertyPath(path) => {
                // Estimate property path size based on complexity
                match path.complexity() {
                    c if c < 10 => 20,
                    c if c < 100 => 50,
                    _ => 100,
                }
            }
        }
    }

    /// Spill current solutions to disk
    fn spill_to_disk(&mut self) -> Result<()> {
        if self.solutions.is_empty() {
            return Ok(());
        }

        let temp_file = if let Some(ref temp_dir) = self.config.temp_dir {
            NamedTempFile::new_in(temp_dir)?
        } else {
            NamedTempFile::new()?
        };

        // Serialize solutions to temporary file
        let serialized_solutions: Vec<SerializableSolution> = self
            .solutions
            .iter()
            .map(SerializableSolution::from_solution)
            .collect();

        let data = if self.config.compress_spills {
            self.compress_data(&serialized_solutions)?
        } else {
            bincode::serialize(&serialized_solutions)?
        };

        {
            let mut writer = BufWriter::new(&temp_file);
            writer.write_all(&data)?;
            writer.flush()?;
        }

        // Persist the temp file to prevent automatic deletion
        let original_path = temp_file.path().to_path_buf();
        let new_path = original_path.with_extension("spill");
        temp_file.persist(&new_path)?;
        let path = new_path;

        // Track spill file
        let spill_file = SpillFile {
            path,
            size: data.len(),
            compressed: self.config.compress_spills,
        };
        self.spill_files.push(spill_file);

        // Clear in-memory solutions and deallocate memory
        let total_size: usize = self
            .solutions
            .iter()
            .map(|sol| self.estimate_solution_size(sol))
            .sum();
        self.memory_tracker.deallocate(total_size);
        self.solutions.clear();

        Ok(())
    }

    /// Compress data using configured compression algorithm
    fn compress_data(&self, data: &[SerializableSolution]) -> Result<Vec<u8>> {
        let serialized = bincode::serialize(data)?;

        match self.config.compression_algorithm {
            CompressionAlgorithm::None => Ok(serialized),
            CompressionAlgorithm::Lz4 => {
                // Fast LZ4 compression for performance-critical scenarios
                Ok(lz4_flex::compress_prepend_size(&serialized))
            }
            CompressionAlgorithm::Gzip => {
                // Balanced gzip compression (existing implementation)
                use std::io::Write;
                let mut encoder =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
                encoder.write_all(&serialized)?;
                Ok(encoder.finish()?)
            }
            CompressionAlgorithm::Zstd => {
                // High compression zstd for space-critical scenarios
                zstd::bulk::compress(&serialized, 3)
                    .map_err(|e| anyhow!("Zstd compression failed: {}", e))
            }
        }
    }

    /// Decompress data using configured compression algorithm
    fn decompress_data(&self, compressed: &[u8]) -> Result<Vec<SerializableSolution>> {
        let decompressed = match self.config.compression_algorithm {
            CompressionAlgorithm::None => {
                // Data is not compressed, use directly
                compressed.to_vec()
            }
            CompressionAlgorithm::Lz4 => {
                // LZ4 decompression
                lz4_flex::decompress_size_prepended(compressed)
                    .map_err(|e| anyhow!("LZ4 decompression failed: {}", e))?
            }
            CompressionAlgorithm::Gzip => {
                // Gzip decompression (existing implementation)
                use std::io::Read;
                let mut decoder = flate2::read::GzDecoder::new(compressed);
                let mut decompressed = Vec::new();
                decoder.read_to_end(&mut decompressed)?;
                decompressed
            }
            CompressionAlgorithm::Zstd => {
                // Zstd decompression
                zstd::bulk::decompress(compressed, 1024 * 1024 * 100) // 100MB max
                    .map_err(|e| anyhow!("Zstd decompression failed: {}", e))?
            }
        };

        Ok(bincode::deserialize(&decompressed)?)
    }

    /// Load solutions from next spill file
    fn load_from_spill(&mut self) -> Result<bool> {
        if self.current_spill_idx >= self.spill_files.len() {
            return Ok(false);
        }

        let spill_file = &self.spill_files[self.current_spill_idx];
        let file = File::open(&spill_file.path)?;
        let mut reader = BufReader::new(file);

        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        let serialized_solutions = if spill_file.compressed {
            self.decompress_data(&data)?
        } else {
            bincode::deserialize(&data)?
        };

        // Convert back to solutions
        for serialized in serialized_solutions {
            let solution = serialized.to_solution();
            self.solutions.push_back(solution);
        }

        self.current_spill_idx += 1;
        Ok(true)
    }

    /// Mark the stream as finished (no more solutions will be added)
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// Get statistics about memory usage and spilling
    pub fn get_stats(&self) -> StreamingStats {
        StreamingStats {
            current_memory: self.memory_tracker.current_usage(),
            peak_memory: self.memory_tracker.peak_usage(),
            spill_files: self.spill_files.len(),
            total_spill_size: self.spill_files.iter().map(|f| f.size).sum(),
            in_memory_solutions: self.solutions.len(),
        }
    }
}

impl Iterator for StreamingSolution {
    type Item = Result<Solution>;

    fn next(&mut self) -> Option<Self::Item> {
        // First try to get from in-memory solutions
        if let Some(solution) = self.solutions.pop_front() {
            // Deallocate memory for this solution
            let size = self.estimate_solution_size(&solution);
            self.memory_tracker.deallocate(size);
            return Some(Ok(solution));
        }

        // If no in-memory solutions, try to load from spill files only if they exist
        if self.current_spill_idx < self.spill_files.len() {
            match self.load_from_spill() {
                Ok(true) => {
                    // Successfully loaded from spill, try again
                    if let Some(solution) = self.solutions.pop_front() {
                        let size = self.estimate_solution_size(&solution);
                        self.memory_tracker.deallocate(size);
                        return Some(Ok(solution));
                    }
                }
                Ok(false) => {
                    // No more spill files
                }
                Err(e) => {
                    return Some(Err(e));
                }
            }
        }

        None
    }
}

impl Drop for StreamingSolution {
    fn drop(&mut self) {
        // Clean up spill files
        for spill_file in &self.spill_files {
            let _ = remove_file(&spill_file.path);
        }
    }
}

/// Serializable version of Solution for disk storage
#[derive(Serialize, Deserialize)]
struct SerializableSolution {
    bindings: Vec<SerializableBinding>,
}

#[derive(Serialize, Deserialize)]
struct SerializableBinding {
    variable: String,
    term: SerializableTerm,
}

#[derive(Serialize, Deserialize)]
enum SerializableTerm {
    Iri(String),
    Literal {
        value: String,
        language: Option<String>,
        datatype: Option<String>,
    },
    BlankNode(String),
    Variable(String),
}

impl SerializableSolution {
    fn from_solution(solution: &Solution) -> Self {
        let mut bindings = Vec::new();
        for binding in solution {
            for (var, term) in binding {
                bindings.push(SerializableBinding {
                    variable: var.as_str().to_string(),
                    term: SerializableTerm::from_term(term),
                });
            }
        }
        Self { bindings }
    }

    fn to_solution(&self) -> Solution {
        let mut solution = Solution::new();
        let mut current_binding = Binding::new();

        for binding in &self.bindings {
            current_binding.insert(
                Variable::new(&binding.variable).unwrap(),
                binding.term.to_term(),
            );
        }

        if !current_binding.is_empty() {
            solution.push(current_binding);
        }

        solution
    }
}

impl SerializableTerm {
    fn from_term(term: &Term) -> Self {
        match term {
            Term::Iri(iri) => Self::Iri(iri.as_str().to_string()),
            Term::Literal(lit) => Self::Literal {
                value: lit.value.clone(),
                language: lit.language.clone(),
                datatype: lit.datatype.as_ref().map(|dt| dt.as_str().to_string()),
            },
            Term::BlankNode(bn) => Self::BlankNode(bn.clone()),
            Term::Variable(var) => Self::Variable(var.as_str().to_string()),
            Term::QuotedTriple(triple) => {
                // For quoted triples, serialize as a string representation
                Self::Literal {
                    value: format!(
                        "<<{} {} {}>>",
                        triple.subject, triple.predicate, triple.object
                    ),
                    language: None,
                    datatype: Some("http://example.org/quoted-triple".to_string()),
                }
            }
            Term::PropertyPath(path) => {
                // For property paths, serialize as a string representation
                Self::Literal {
                    value: path.to_string(),
                    language: None,
                    datatype: Some("http://example.org/property-path".to_string()),
                }
            }
        }
    }

    fn to_term(&self) -> Term {
        match self {
            Self::Iri(iri) => Term::Iri(NamedNode::new(iri).unwrap()),
            Self::Literal {
                value,
                language,
                datatype,
            } => Term::Literal(crate::algebra::Literal {
                value: value.clone(),
                language: language.clone(),
                datatype: datatype.as_ref().map(|dt| NamedNode::new(dt).unwrap()),
            }),
            Self::BlankNode(bn) => Term::BlankNode(bn.clone()),
            Self::Variable(var) => Term::Variable(Variable::new(var).unwrap()),
        }
    }
}

/// Spillable hash join for memory-efficient joins
pub struct SpillableHashJoin {
    config: StreamingConfig,
    memory_tracker: MemoryTracker,
    hash_buckets: Vec<HashMap<String, Vec<Solution>>>,
    spill_buckets: Vec<Vec<SpillFile>>,
    num_buckets: usize,
}

impl SpillableHashJoin {
    pub fn new(config: StreamingConfig) -> Self {
        let num_buckets = 16; // Can be made configurable
        let memory_tracker = MemoryTracker::new(config.memory_limit);

        Self {
            config,
            memory_tracker,
            hash_buckets: (0..num_buckets).map(|_| HashMap::new()).collect(),
            spill_buckets: (0..num_buckets).map(|_| Vec::new()).collect(),
            num_buckets,
        }
    }

    /// Execute spillable hash join
    pub fn execute(
        &mut self,
        left: Vec<Solution>,
        right: Vec<Solution>,
        join_vars: &[Variable],
    ) -> Result<Vec<Solution>> {
        // Phase 1: Build hash table from left side with spilling
        self.build_phase(left, join_vars)?;

        // Phase 2: Probe with right side
        let mut results = Vec::new();
        self.probe_phase(right, join_vars, &mut results)?;

        // Phase 3: Handle spilled buckets
        self.handle_spilled_buckets(join_vars, &mut results)?;

        Ok(results)
    }

    /// Build phase: create hash table from left relations
    fn build_phase(&mut self, left: Vec<Solution>, join_vars: &[Variable]) -> Result<()> {
        for solution in left {
            let hash_key = self.create_hash_key(&solution, join_vars);
            let bucket_idx = self.hash_to_bucket(&hash_key);

            let solution_size = self.estimate_solution_size(&solution);

            if !self.memory_tracker.allocate(solution_size)? {
                // Spill this bucket
                self.spill_bucket(bucket_idx)?;
                // Try to allocate again
                if !self.memory_tracker.allocate(solution_size)? {
                    return Err(anyhow!("Cannot allocate memory even after spilling bucket"));
                }
            }

            self.hash_buckets[bucket_idx]
                .entry(hash_key)
                .or_default()
                .push(solution);
        }

        Ok(())
    }

    /// Probe phase: join with right relations
    fn probe_phase(
        &mut self,
        right: Vec<Solution>,
        join_vars: &[Variable],
        results: &mut Vec<Solution>,
    ) -> Result<()> {
        for right_solution in right {
            let hash_key = self.create_hash_key(&right_solution, join_vars);
            let bucket_idx = self.hash_to_bucket(&hash_key);

            // Check in-memory bucket
            if let Some(left_solutions) = self.hash_buckets[bucket_idx].get(&hash_key) {
                for left_solution in left_solutions {
                    if let Some(joined) =
                        self.join_solutions(left_solution, &right_solution, join_vars)
                    {
                        results.push(joined);
                    }
                }
            }

            // Note: For spilled buckets, we'll handle them separately in handle_spilled_buckets
        }

        Ok(())
    }

    /// Handle spilled buckets by loading them back and processing
    fn handle_spilled_buckets(
        &mut self,
        join_vars: &[Variable],
        results: &mut Vec<Solution>,
    ) -> Result<()> {
        for bucket_idx in 0..self.num_buckets {
            if !self.spill_buckets[bucket_idx].is_empty() {
                // Process each spilled file for this bucket
                for spill_file in &self.spill_buckets[bucket_idx] {
                    let solutions = self.load_spilled_solutions(spill_file)?;
                    // This is a simplified approach - in practice, you'd want to
                    // handle cases where the spilled data is still too large
                    self.process_spilled_bucket_solutions(solutions, join_vars, results)?;
                }
            }
        }

        Ok(())
    }

    /// Create hash key from solution using join variables
    fn create_hash_key(&self, solution: &Solution, join_vars: &[Variable]) -> String {
        let mut key_parts = Vec::new();

        for binding in solution {
            for join_var in join_vars {
                if let Some(term) = binding.get(join_var) {
                    key_parts.push(format!("{join_var}:{term:?}"));
                }
            }
        }

        key_parts.join("|")
    }

    /// Hash key to bucket index
    fn hash_to_bucket(&self, key: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.num_buckets
    }

    /// Spill a bucket to disk
    fn spill_bucket(&mut self, bucket_idx: usize) -> Result<()> {
        let bucket = &mut self.hash_buckets[bucket_idx];
        if bucket.is_empty() {
            return Ok(());
        }

        let temp_file = if let Some(ref temp_dir) = self.config.temp_dir {
            NamedTempFile::new_in(temp_dir)?
        } else {
            NamedTempFile::new()?
        };

        let (file, path) = temp_file.into_parts();
        let mut writer = BufWriter::new(file);

        // Flatten bucket solutions for serialization
        let mut all_solutions = Vec::new();
        for solutions in bucket.values() {
            all_solutions.extend(solutions.iter().cloned());
        }

        let serialized_solutions: Vec<SerializableSolution> = all_solutions
            .iter()
            .map(SerializableSolution::from_solution)
            .collect();

        let data = bincode::serialize(&serialized_solutions)?;
        writer.write_all(&data)?;
        writer.flush()?;

        // Track spill file
        let spill_file = SpillFile {
            path: path.to_path_buf(),
            size: data.len(),
            compressed: false, // Can be made configurable
        };
        self.spill_buckets[bucket_idx].push(spill_file);

        // Calculate total size first, then clear the bucket and deallocate memory
        let total_size: usize = all_solutions
            .iter()
            .map(|sol| {
                let mut size = std::mem::size_of::<Solution>();
                for binding in sol {
                    size += binding.len()
                        * (std::mem::size_of::<Variable>() + std::mem::size_of::<Term>());
                    size += binding
                        .iter()
                        .map(|(var, term)| {
                            let term_size = match term {
                                Term::Iri(iri) => iri.as_str().len(),
                                Term::Literal(lit) => {
                                    lit.value.len() + lit.language.as_ref().map_or(0, |l| l.len())
                                }
                                Term::BlankNode(bn) => bn.len(),
                                Term::Variable(var) => var.as_str().len(),
                                Term::QuotedTriple(_) => 100, // Estimate for quoted triple
                                Term::PropertyPath(_) => 50,  // Estimate for property path
                            };
                            var.as_str().len() + term_size
                        })
                        .sum::<usize>();
                }
                size
            })
            .sum();
        self.memory_tracker.deallocate(total_size);
        bucket.clear();

        Ok(())
    }

    /// Load solutions from spill file
    fn load_spilled_solutions(&self, spill_file: &SpillFile) -> Result<Vec<Solution>> {
        let file = File::open(&spill_file.path)?;
        let mut reader = BufReader::new(file);

        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        let serialized_solutions: Vec<SerializableSolution> = bincode::deserialize(&data)?;

        Ok(serialized_solutions
            .into_iter()
            .map(|s| s.to_solution())
            .collect())
    }

    /// Process solutions from spilled bucket
    fn process_spilled_bucket_solutions(
        &self,
        solutions: Vec<Solution>,
        join_vars: &[Variable],
        results: &mut Vec<Solution>,
    ) -> Result<()> {
        // Rebuild hash table for this spilled bucket
        let mut bucket_hash_table = HashMap::new();

        for solution in solutions {
            let hash_key = self.create_hash_key(&solution, join_vars);
            bucket_hash_table
                .entry(hash_key)
                .or_insert_with(Vec::new)
                .push(solution);
        }

        // Note: In a complete implementation, we would need to store the right-side
        // solutions that correspond to this bucket and join them here.
        // For now, we'll add the solutions to results as a placeholder.
        // This represents the left-side solutions that would be joined.

        for (_, bucket_solutions) in bucket_hash_table {
            results.extend(bucket_solutions);
        }

        Ok(())
    }

    /// Join two solutions
    fn join_solutions(
        &self,
        left: &Solution,
        right: &Solution,
        join_vars: &[Variable],
    ) -> Option<Solution> {
        // Check if join variables have compatible values
        for left_binding in left {
            for right_binding in right {
                let mut compatible = true;
                for join_var in join_vars {
                    let left_val = left_binding.get(join_var);
                    let right_val = right_binding.get(join_var);

                    match (left_val, right_val) {
                        (Some(l), Some(r)) if l != r => {
                            compatible = false;
                            break;
                        }
                        _ => {}
                    }
                }

                if compatible {
                    // Create joined solution
                    let mut joined = Solution::new();
                    let mut new_binding = Binding::new();

                    // Add all bindings from left
                    for (var, term) in left_binding {
                        new_binding.insert(var.clone(), term.clone());
                    }

                    // Add non-conflicting bindings from right
                    for (var, term) in right_binding {
                        if !new_binding.contains_key(var) {
                            new_binding.insert(var.clone(), term.clone());
                        }
                    }

                    joined.push(new_binding);
                    return Some(joined);
                }
            }
        }

        None
    }

    /// Estimate memory size of a solution
    fn estimate_solution_size(&self, solution: &Solution) -> usize {
        let mut size = std::mem::size_of::<Solution>();
        for binding in solution {
            size += binding.len() * (std::mem::size_of::<Variable>() + std::mem::size_of::<Term>());
            size += binding
                .iter()
                .map(|(var, term)| var.as_str().len() + self.estimate_term_size(term))
                .sum::<usize>();
        }
        size
    }

    /// Estimate memory size of a term
    fn estimate_term_size(&self, term: &Term) -> usize {
        match term {
            Term::Iri(iri) => iri.as_str().len(),
            Term::Literal(lit) => lit.value.len() + lit.language.as_ref().map_or(0, |l| l.len()),
            Term::BlankNode(bn) => bn.len(),
            Term::Variable(var) => var.as_str().len(),
            Term::QuotedTriple(_) => 100, // Estimate for quoted triple
            Term::PropertyPath(_) => 50,  // Estimate for property path
        }
    }
}

/// Statistics about streaming execution
#[derive(Debug, Clone)]
pub struct StreamingStats {
    pub current_memory: usize,
    pub peak_memory: usize,
    pub spill_files: usize,
    pub total_spill_size: usize,
    pub in_memory_solutions: usize,
}

impl Default for SpillableHashJoin {
    fn default() -> Self {
        Self::new(StreamingConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_solution_basic() {
        let config = StreamingConfig {
            memory_limit: 1024, // Small limit to force spilling
            buffer_size: 2,
            ..Default::default()
        };

        let mut stream = StreamingSolution::new(config);

        // Add some test solutions
        let mut solution1 = Solution::new();
        let mut binding1 = Binding::new();
        binding1.insert(
            Variable::new("x").unwrap(),
            Term::Iri(NamedNode::new("http://example.org/1").unwrap()),
        );
        solution1.push(binding1);

        let mut solution2 = Solution::new();
        let mut binding2 = Binding::new();
        binding2.insert(
            Variable::new("y").unwrap(),
            Term::Iri(NamedNode::new("http://example.org/2").unwrap()),
        );
        solution2.push(binding2);

        stream.add_solution(solution1).unwrap();
        stream.add_solution(solution2).unwrap();
        stream.finish();

        // Should be able to iterate through solutions
        let mut count = 0;
        for result in &mut stream {
            assert!(result.is_ok());
            count += 1;
        }

        assert_eq!(count, 2);
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = MemoryTracker::new(1000);

        assert!(tracker.allocate(500).unwrap());
        assert_eq!(tracker.current_usage(), 500);

        assert!(tracker.allocate(400).unwrap());
        assert_eq!(tracker.current_usage(), 900);

        assert!(!tracker.allocate(200).unwrap()); // Should exceed limit

        tracker.deallocate(400);
        assert_eq!(tracker.current_usage(), 500);

        assert!(tracker.allocate(200).unwrap());
    }

    #[test]
    fn test_spillable_hash_join() {
        let config = StreamingConfig {
            memory_limit: 2048,
            ..Default::default()
        };

        let mut join = SpillableHashJoin::new(config);

        // Create proper Solution structures
        let mut left_solution = Solution::new();
        let mut left_binding = Binding::new();
        left_binding.insert(
            Variable::new("x").unwrap(),
            Term::Iri(NamedNode::new("http://example.org/1").unwrap()),
        );
        left_solution.push(left_binding);

        let mut right_solution = Solution::new();
        let mut right_binding = Binding::new();
        right_binding.insert(
            Variable::new("x").unwrap(),
            Term::Iri(NamedNode::new("http://example.org/1").unwrap()),
        );
        right_solution.push(right_binding);

        let left = vec![left_solution];
        let right = vec![right_solution];
        let join_vars = vec![Variable::new("x").unwrap()];
        let results = join.execute(left, right, &join_vars).unwrap();

        assert!(!results.is_empty());
    }
}
