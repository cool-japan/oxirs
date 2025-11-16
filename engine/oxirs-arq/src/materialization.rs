//! Query Result Materialization Strategies
//!
//! This module provides sophisticated materialization strategies for SPARQL query results,
//! balancing memory usage, latency, and throughput using scirs2-core features.

use crate::algebra::Term;
use anyhow::{anyhow, Result};
use oxirs_core::model::Variable;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};

// Use scirs2-core for advanced features
use scirs2_core::ndarray_ext::Array1;
use scirs2_core::profiling::Profiler;
use scirs2_core::random::{rng, Rng}; // Use scirs2-core for random number generation
use scirs2_stats::{mean, std};

/// Solution type alias - mapping of variables to terms
pub type Solution = HashMap<Variable, Term>;

/// Materialization strategy determines how query results are stored and processed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MaterializationStrategy {
    /// Stream results without materialization (lowest memory, highest latency for reuse)
    Streaming,
    /// Materialize to memory (high memory, low latency)
    InMemory,
    /// Hybrid: Stream until threshold, then materialize
    #[default]
    Adaptive,
    /// Materialize to disk with memory-mapped access
    MemoryMapped,
    /// Chunked processing with configurable chunk size
    Chunked,
    /// Lazy evaluation with caching
    Lazy,
}

/// Configuration for materialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializationConfig {
    /// Default strategy
    pub default_strategy: MaterializationStrategy,
    /// Memory limit for in-memory materialization (bytes)
    pub memory_limit: usize,
    /// Threshold for adaptive switching (number of results)
    pub adaptive_threshold: usize,
    /// Chunk size for chunked processing
    pub chunk_size: usize,
    /// Enable profiling for materialization decisions
    pub enable_profiling: bool,
    /// Estimated result size for strategy selection
    pub estimated_result_size: Option<usize>,
    /// Enable compression for disk-based materialization
    pub enable_compression: bool,
}

impl Default for MaterializationConfig {
    fn default() -> Self {
        Self {
            default_strategy: MaterializationStrategy::Adaptive,
            memory_limit: 1024 * 1024 * 1024, // 1GB
            adaptive_threshold: 10_000,
            chunk_size: 1000,
            enable_profiling: true,
            estimated_result_size: None,
            enable_compression: true,
        }
    }
}

/// Materialization statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MaterializationStats {
    /// Total results materialized
    pub total_results: usize,
    /// Memory used (bytes)
    pub memory_used: usize,
    /// Disk space used (bytes)
    pub disk_used: usize,
    /// Strategy used
    pub strategy_used: Option<MaterializationStrategy>,
    /// Time taken for materialization (ms)
    pub materialization_time_ms: f64,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
}

/// Materialized result container
pub struct MaterializedResults {
    /// Strategy used
    strategy: MaterializationStrategy,
    /// In-memory results (for InMemory strategy)
    in_memory: Vec<Solution>,
    /// Streaming iterator state (for Streaming strategy)
    stream_buffer: VecDeque<Solution>,
    /// Chunked results (for Chunked strategy)
    chunks: Vec<Vec<Solution>>,
    /// Lazy results with cache (for Lazy strategy)
    lazy_cache: HashMap<usize, Solution>,
    /// Temporary file path for memory-mapped results
    temp_file_path: Option<String>,
    /// Statistics
    stats: Arc<RwLock<MaterializationStats>>,
    /// Configuration
    config: MaterializationConfig,
}

impl MaterializedResults {
    /// Create new materialized results with the given strategy
    pub fn new(strategy: MaterializationStrategy, config: MaterializationConfig) -> Self {
        Self {
            strategy,
            in_memory: Vec::new(),
            stream_buffer: VecDeque::new(),
            chunks: Vec::new(),
            lazy_cache: HashMap::new(),
            temp_file_path: None,
            stats: Arc::new(RwLock::new(MaterializationStats {
                strategy_used: Some(strategy),
                ..Default::default()
            })),
            config,
        }
    }

    /// Add a solution to the materialized results
    pub fn add_solution(&mut self, solution: Solution) -> Result<()> {
        match self.strategy {
            MaterializationStrategy::InMemory => {
                self.in_memory.push(solution);
                self.update_stats();
            }
            MaterializationStrategy::Streaming => {
                self.stream_buffer.push_back(solution);
                // Keep buffer bounded
                if self.stream_buffer.len() > self.config.chunk_size {
                    self.stream_buffer.pop_front();
                }
            }
            MaterializationStrategy::Adaptive => {
                self.in_memory.push(solution);
                // Check if we should switch strategy
                if self.in_memory.len() > self.config.adaptive_threshold {
                    self.switch_to_disk()?;
                }
            }
            MaterializationStrategy::Chunked => {
                // Add to current chunk
                self.in_memory.push(solution);
                if self.in_memory.len() >= self.config.chunk_size {
                    self.flush_chunk()?;
                }
            }
            MaterializationStrategy::Lazy => {
                // Cache only, actual data loaded on demand
                let idx = self.lazy_cache.len();
                self.lazy_cache.insert(idx, solution);
            }
            MaterializationStrategy::MemoryMapped => {
                // For now, collect in memory and flush to mmap later
                self.in_memory.push(solution);
            }
        }
        Ok(())
    }

    /// Get solution at index
    pub fn get_solution(&mut self, index: usize) -> Option<&Solution> {
        match self.strategy {
            MaterializationStrategy::InMemory | MaterializationStrategy::Adaptive => {
                self.in_memory.get(index)
            }
            MaterializationStrategy::Lazy => {
                if self.lazy_cache.contains_key(&index) {
                    let mut stats = self.stats.write().unwrap();
                    stats.cache_hits += 1;
                    drop(stats);
                    self.lazy_cache.get(&index)
                } else {
                    let mut stats = self.stats.write().unwrap();
                    stats.cache_misses += 1;
                    None
                }
            }
            _ => None, // Other strategies don't support random access
        }
    }

    /// Get total number of results
    pub fn len(&self) -> usize {
        match self.strategy {
            MaterializationStrategy::InMemory | MaterializationStrategy::Adaptive => {
                self.in_memory.len()
            }
            MaterializationStrategy::Lazy => self.lazy_cache.len(),
            MaterializationStrategy::Chunked => {
                self.chunks.len() * self.config.chunk_size + self.in_memory.len()
            }
            _ => 0,
        }
    }

    /// Check if results are empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get iterator over all results
    pub fn iter(&self) -> ResultIterator<'_> {
        ResultIterator {
            results: self,
            current_index: 0,
        }
    }

    /// Flush current chunk to disk
    fn flush_chunk(&mut self) -> Result<()> {
        if self.in_memory.is_empty() {
            return Ok(());
        }

        // Store chunk in memory for now (simplified implementation)
        let chunk = std::mem::take(&mut self.in_memory);
        self.chunks.push(chunk);

        Ok(())
    }

    /// Switch from in-memory to disk-based storage
    fn switch_to_disk(&mut self) -> Result<()> {
        // Serialize all in-memory data
        let serialized = bincode::serialize(&self.in_memory)
            .map_err(|e| anyhow!("Failed to serialize results: {}", e))?;

        // Create memory-mapped file
        use std::env::temp_dir;
        use std::fs::File;
        use std::io::Write;

        let random_id: u64 = rng().random();
        let temp_path = temp_dir().join(format!("sparql_results_{}.bin", random_id));
        let mut file = File::create(&temp_path)?;
        file.write_all(&serialized)?;
        drop(file);

        self.temp_file_path = Some(temp_path.to_string_lossy().to_string());
        self.in_memory.clear();
        self.strategy = MaterializationStrategy::MemoryMapped;

        let mut stats = self.stats.write().unwrap();
        stats.strategy_used = Some(MaterializationStrategy::MemoryMapped);
        stats.disk_used = serialized.len();

        Ok(())
    }

    /// Update statistics
    fn update_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.total_results = self.len();
        // Estimate memory usage (simplified)
        stats.memory_used = self.in_memory.len() * std::mem::size_of::<Solution>();
    }

    /// Get statistics
    pub fn get_stats(&self) -> MaterializationStats {
        self.stats.read().unwrap().clone()
    }

    /// Analyze result patterns using scirs2-stats
    pub fn analyze_patterns(&self) -> Result<MaterializationAnalysis> {
        if self.in_memory.is_empty() {
            return Ok(MaterializationAnalysis::default());
        }

        // Analyze cardinality distribution across variables
        let mut var_cardinalities: HashMap<String, Vec<f64>> = HashMap::new();

        for solution in &self.in_memory {
            for (var, _term) in solution.iter() {
                let var_name = format!("{}", var);
                var_cardinalities.entry(var_name).or_default().push(1.0); // Count occurrences
            }
        }

        // Calculate statistics for each variable using scirs2-stats
        let mut analysis = MaterializationAnalysis::default();

        for (var_name, counts) in var_cardinalities {
            if !counts.is_empty() {
                let arr = Array1::from_vec(counts.clone());
                let arr_view = arr.view();

                let mean_val = mean(&arr_view).unwrap_or(0.0);
                let std_val = std(&arr_view, 1, None).unwrap_or(0.0);

                analysis.variable_stats.insert(
                    var_name.clone(),
                    VariableStats {
                        mean_cardinality: mean_val,
                        std_cardinality: std_val,
                        total_occurrences: counts.len(),
                    },
                );
            }
        }

        analysis.total_solutions = self.in_memory.len();
        analysis.estimated_memory = self.in_memory.len() * std::mem::size_of::<Solution>();

        Ok(analysis)
    }
}

/// Iterator over materialized results
pub struct ResultIterator<'a> {
    results: &'a MaterializedResults,
    current_index: usize,
}

impl<'a> Iterator for ResultIterator<'a> {
    type Item = &'a Solution;

    fn next(&mut self) -> Option<Self::Item> {
        let solution = match self.results.strategy {
            MaterializationStrategy::InMemory | MaterializationStrategy::Adaptive => {
                self.results.in_memory.get(self.current_index)
            }
            _ => None,
        };

        if solution.is_some() {
            self.current_index += 1;
        }

        solution
    }
}

/// Analysis of materialized results
#[derive(Debug, Clone, Default)]
pub struct MaterializationAnalysis {
    /// Total number of solutions
    pub total_solutions: usize,
    /// Estimated memory usage
    pub estimated_memory: usize,
    /// Statistics per variable
    pub variable_stats: HashMap<String, VariableStats>,
}

/// Statistics for a single variable
#[derive(Debug, Clone)]
pub struct VariableStats {
    /// Mean cardinality
    pub mean_cardinality: f64,
    /// Standard deviation of cardinality
    pub std_cardinality: f64,
    /// Total occurrences
    pub total_occurrences: usize,
}

/// Materialization strategy selector
pub struct MaterializationSelector {
    config: MaterializationConfig,
    #[allow(dead_code)]
    profiler: Option<Profiler>,
}

impl MaterializationSelector {
    /// Create a new selector
    pub fn new(config: MaterializationConfig) -> Self {
        let profiler = if config.enable_profiling {
            Some(Profiler::new())
        } else {
            None
        };

        Self { config, profiler }
    }

    /// Select the best materialization strategy based on query characteristics
    pub fn select_strategy(&self, estimated_results: Option<usize>) -> MaterializationStrategy {
        // Use estimated results or fall back to query analysis
        let result_count = estimated_results.or(self.config.estimated_result_size);

        match result_count {
            Some(count) if count < 1000 => MaterializationStrategy::InMemory,
            Some(count) if count < self.config.adaptive_threshold => {
                MaterializationStrategy::InMemory
            }
            Some(count) if count < 100_000 => MaterializationStrategy::Chunked,
            Some(_) => MaterializationStrategy::MemoryMapped,
            None => MaterializationStrategy::Adaptive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Literal, Term};

    #[test]
    fn test_in_memory_materialization() {
        let config = MaterializationConfig::default();
        let mut results = MaterializedResults::new(MaterializationStrategy::InMemory, config);

        // Add some solutions
        for i in 0..100 {
            let mut solution = Solution::new();
            let var = Variable::new(format!("x{}", i)).unwrap();
            solution.insert(
                var,
                Term::Literal(Literal {
                    value: i.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            results.add_solution(solution).unwrap();
        }

        assert_eq!(results.len(), 100);
        assert!(results.get_solution(50).is_some());

        let stats = results.get_stats();
        assert_eq!(stats.total_results, 100);
    }

    #[test]
    fn test_adaptive_materialization() {
        let config = MaterializationConfig {
            adaptive_threshold: 10,
            ..Default::default()
        };

        let mut results = MaterializedResults::new(MaterializationStrategy::Adaptive, config);

        // Add solutions beyond threshold
        for i in 0..20 {
            let mut solution = Solution::new();
            let var = Variable::new(format!("x{}", i)).unwrap();
            solution.insert(
                var,
                Term::Literal(Literal {
                    value: i.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            results.add_solution(solution).unwrap();
        }

        // Strategy should have switched
        let stats = results.get_stats();
        assert!(stats.strategy_used == Some(MaterializationStrategy::MemoryMapped));
    }

    #[test]
    fn test_strategy_selection() {
        let config = MaterializationConfig::default();
        let selector = MaterializationSelector::new(config);

        // Small result set
        let strategy = selector.select_strategy(Some(100));
        assert_eq!(strategy, MaterializationStrategy::InMemory);

        // Large result set
        let strategy = selector.select_strategy(Some(1_000_000));
        assert_eq!(strategy, MaterializationStrategy::MemoryMapped);
    }

    #[test]
    fn test_result_analysis() {
        let config = MaterializationConfig::default();
        let mut results = MaterializedResults::new(MaterializationStrategy::InMemory, config);

        for i in 0..50 {
            let mut solution = Solution::new();
            let var = Variable::new("x".to_string()).unwrap();
            solution.insert(
                var,
                Term::Literal(Literal {
                    value: i.to_string(),
                    language: None,
                    datatype: None,
                }),
            );
            results.add_solution(solution).unwrap();
        }

        let analysis = results.analyze_patterns().unwrap();
        assert_eq!(analysis.total_solutions, 50);
        // Variable name format is "?x"
        let has_x_var = analysis.variable_stats.keys().any(|k| k.contains("x"));
        assert!(has_x_var);
    }
}
