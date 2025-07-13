//! Streaming Query Execution Engine
//!
//! This module provides streaming execution capabilities for handling large datasets
//! that don't fit in memory, with sophisticated spilling and memory management.

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tempfile::TempDir;
use tracing::{debug, info, span, warn, Level};

use crate::algebra::{Algebra, Binding, Expression, Solution, Term, TriplePattern, Variable};
use oxirs_core::model::NamedNode;

/// Streaming execution engine for large datasets
pub struct StreamingExecutor {
    config: StreamingConfig,
    memory_monitor: MemoryMonitor,
    spill_manager: Arc<Mutex<SpillManager>>,
    #[allow(dead_code)]
    temp_dir: TempDir,
    active_streams: HashMap<String, Box<dyn DataStream>>,
    execution_stats: StreamingStats,
}

/// Configuration for streaming execution
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum memory usage before spilling (bytes)
    pub max_memory_usage: usize,
    /// Memory threshold for spilling (0.0 to 1.0)
    pub spill_threshold: f64,
    /// Batch size for processing
    pub batch_size: usize,
    /// Number of parallel workers
    pub parallel_workers: usize,
    /// Compression level for spilled data (0-9)
    pub compression_level: u32,
    /// Enable memory mapping for large files
    pub enable_memory_mapping: bool,
    /// Buffer size for I/O operations
    pub io_buffer_size: usize,
    /// Enable adaptive batch sizing
    pub adaptive_batching: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_memory_usage: 512 * 1024 * 1024, // 512MB
            spill_threshold: 0.8,
            batch_size: 10000,
            parallel_workers: 4,
            compression_level: 6,
            enable_memory_mapping: true,
            io_buffer_size: 64 * 1024, // 64KB
            adaptive_batching: true,
        }
    }
}

/// Memory monitoring and management
pub struct MemoryMonitor {
    inner: Arc<Mutex<MemoryMonitorInner>>,
}

/// Internal state for memory monitor
struct MemoryMonitorInner {
    current_usage: usize,
    peak_usage: usize,
    max_allowed: usize,
    allocation_history: VecDeque<MemoryAllocation>,
}

/// Memory allocation record
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub timestamp: Instant,
    pub size: usize,
    pub operation: String,
    pub freed: bool,
}

/// Spill manager for handling memory overflow
pub struct SpillManager {
    spill_directory: PathBuf,
    active_spills: HashMap<String, SpillInfo>,
    spill_counter: usize,
    compression_enabled: bool,
    compression_level: u32,
}

/// Information about a spilled data structure
#[derive(Debug, Clone)]
pub struct SpillInfo {
    pub file_path: PathBuf,
    pub original_size: usize,
    pub compressed_size: usize,
    pub data_type: SpillDataType,
    pub creation_time: Instant,
    pub access_count: usize,
}

/// Types of data that can be spilled
#[derive(Debug, Clone)]
pub enum SpillDataType {
    Solutions,
    HashTable,
    SortBuffer,
    IntermediateResults,
    Index,
}

/// Generic data stream interface
pub trait DataStream: Send + Sync {
    /// Get the next batch of data
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>>;

    /// Check if there is more data available
    fn has_more(&self) -> bool;

    /// Get estimated size of remaining data
    fn estimated_size(&self) -> Option<usize>;

    /// Reset the stream to the beginning
    fn reset(&mut self) -> Result<()>;

    /// Get stream statistics
    fn get_stats(&self) -> StreamStats;
}

/// Statistics for data streams
#[derive(Debug, Clone, Default)]
pub struct StreamStats {
    pub rows_processed: usize,
    pub bytes_processed: usize,
    pub processing_time: Duration,
    pub spill_operations: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Streaming statistics
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    pub total_memory_used: usize,
    pub peak_memory_used: usize,
    pub spill_operations: usize,
    pub total_spill_size: usize,
    pub total_execution_time: Duration,
    pub rows_processed: usize,
    pub cache_hit_rate: f64,
}

/// Streaming hash join implementation
pub struct StreamingHashJoin {
    left_stream: Box<dyn DataStream>,
    right_stream: Box<dyn DataStream>,
    join_variables: Vec<Variable>,
    hash_table: HashMap<String, Vec<Solution>>,
    memory_monitor: Arc<MemoryMonitor>,
    spill_manager: Arc<Mutex<SpillManager>>,
    #[allow(dead_code)]
    config: StreamingConfig,
    left_exhausted: bool,
    current_batch: Option<Vec<Solution>>,
    spilled_partitions: Vec<String>, // Spill IDs for partitioned hash tables
    #[allow(dead_code)]
    current_spill_index: usize,
}

/// Streaming sort-merge join implementation
pub struct StreamingSortMergeJoin {
    left_stream: Box<dyn DataStream>,
    right_stream: Box<dyn DataStream>,
    join_variables: Vec<Variable>,
    left_buffer: VecDeque<Solution>,
    right_buffer: VecDeque<Solution>,
    #[allow(dead_code)]
    memory_monitor: Arc<MemoryMonitor>,
    #[allow(dead_code)]
    spill_manager: Arc<Mutex<SpillManager>>,
    #[allow(dead_code)]
    config: StreamingConfig,
}

impl StreamingSortMergeJoin {
    fn new(
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
        join_variables: Vec<Variable>,
        memory_monitor: Arc<MemoryMonitor>,
        spill_manager: Arc<Mutex<SpillManager>>,
        config: StreamingConfig,
    ) -> Result<Self> {
        Ok(Self {
            left_stream: left,
            right_stream: right,
            join_variables,
            left_buffer: VecDeque::new(),
            right_buffer: VecDeque::new(),
            memory_monitor,
            spill_manager,
            config,
        })
    }
}

impl DataStream for StreamingSortMergeJoin {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        // Proper sort-merge join implementation
        let mut result_batch = Vec::new();

        // Ensure both buffers have sorted data
        self.refill_sorted_buffers()?;

        // Perform sort-merge join
        while !self.left_buffer.is_empty() && !self.right_buffer.is_empty() {
            let left_solution = &self.left_buffer[0];
            let right_solution = &self.right_buffer[0];

            // Compare join keys
            let comparison = self.compare_join_keys(left_solution, right_solution);

            match comparison {
                std::cmp::Ordering::Less => {
                    // Left key is smaller, advance left
                    self.left_buffer.pop_front();
                }
                std::cmp::Ordering::Greater => {
                    // Right key is smaller, advance right
                    self.right_buffer.pop_front();
                }
                std::cmp::Ordering::Equal => {
                    // Keys match, find all matching solutions
                    let left_key = self.extract_join_key(left_solution);

                    // Collect all left solutions with this key
                    let mut matching_left = Vec::new();
                    while !self.left_buffer.is_empty() {
                        let current_left = &self.left_buffer[0];
                        if self.extract_join_key(current_left) == left_key {
                            matching_left.push(self.left_buffer.pop_front().unwrap());
                        } else {
                            break;
                        }
                    }

                    // Collect all right solutions with this key
                    let mut matching_right = Vec::new();
                    while !self.right_buffer.is_empty() {
                        let current_right = &self.right_buffer[0];
                        if self.extract_join_key(current_right) == left_key {
                            matching_right.push(self.right_buffer.pop_front().unwrap());
                        } else {
                            break;
                        }
                    }

                    // Join all matching solutions
                    for left_sol in &matching_left {
                        for right_sol in &matching_right {
                            if let Some(joined) = self.join_solutions(left_sol, right_sol) {
                                result_batch.push(joined);
                            }
                        }
                    }

                    // Check batch size limit
                    if result_batch.len() >= self.config.batch_size {
                        break;
                    }
                }
            }

            // Refill buffers if needed
            if self.left_buffer.len() < 10 || self.right_buffer.len() < 10 {
                self.refill_sorted_buffers()?;
            }
        }

        if result_batch.is_empty() && !self.has_more() {
            Ok(None)
        } else {
            Ok(Some(result_batch))
        }
    }

    fn has_more(&self) -> bool {
        !self.left_buffer.is_empty()
            || !self.right_buffer.is_empty()
            || self.left_stream.has_more()
            || self.right_stream.has_more()
    }

    fn estimated_size(&self) -> Option<usize> {
        let left_est = self.left_stream.estimated_size().unwrap_or(0);
        let right_est = self.right_stream.estimated_size().unwrap_or(0);
        Some(left_est + right_est + self.left_buffer.len() + self.right_buffer.len())
    }

    fn reset(&mut self) -> Result<()> {
        self.left_buffer.clear();
        self.right_buffer.clear();
        self.left_stream.reset()?;
        self.right_stream.reset()
    }

    fn get_stats(&self) -> StreamStats {
        StreamStats::default()
    }
}

impl StreamingSortMergeJoin {
    /// Refill buffers with sorted data from input streams
    fn refill_sorted_buffers(&mut self) -> Result<()> {
        // Fill left buffer if needed
        if self.left_buffer.is_empty() {
            if let Some(mut batch) = self.left_stream.next_batch()? {
                // Sort the batch by join variables
                batch.sort_by(|a, b| self.compare_solution_keys(a, b));
                self.left_buffer.extend(batch);
            }
        }

        // Fill right buffer if needed
        if self.right_buffer.is_empty() {
            if let Some(mut batch) = self.right_stream.next_batch()? {
                // Sort the batch by join variables
                batch.sort_by(|a, b| self.compare_solution_keys(a, b));
                self.right_buffer.extend(batch);
            }
        }

        Ok(())
    }

    /// Compare join keys of two solutions
    fn compare_join_keys(&self, left: &Solution, right: &Solution) -> std::cmp::Ordering {
        self.compare_solution_keys(left, right)
    }

    /// Compare solutions by their join variable values
    fn compare_solution_keys(&self, left: &Solution, right: &Solution) -> std::cmp::Ordering {
        for var in &self.join_variables {
            let left_val = Self::get_solution_value(left, var);
            let right_val = Self::get_solution_value(right, var);

            match (left_val, right_val) {
                (Some(left_term), Some(right_term)) => {
                    let cmp = format!("{left_term:?}").cmp(&format!("{right_term:?}"));
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                (Some(_), None) => return std::cmp::Ordering::Greater,
                (None, Some(_)) => return std::cmp::Ordering::Less,
                (None, None) => continue,
            }
        }
        std::cmp::Ordering::Equal
    }

    /// Extract join key from solution
    fn extract_join_key(&self, solution: &Solution) -> String {
        self.join_variables
            .iter()
            .map(|var| {
                Self::get_solution_value(solution, var)
                    .map(|term| format!("{term:?}"))
                    .unwrap_or_else(|| "NULL".to_string())
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    /// Helper function to get a value from a solution
    fn get_solution_value<'a>(solution: &'a Solution, var: &Variable) -> Option<&'a Term> {
        solution.first().and_then(|binding| binding.get(var))
    }

    /// Join two compatible solutions
    fn join_solutions(&self, left: &Solution, right: &Solution) -> Option<Solution> {
        // Check that join variables have compatible values
        for var in &self.join_variables {
            let left_val = Self::get_solution_value(left, var);
            let right_val = Self::get_solution_value(right, var);

            match (left_val, right_val) {
                (Some(l), Some(r)) if l != r => return None,
                _ => {}
            }
        }

        // Merge the solutions by combining the first bindings from each
        let mut result_binding = Binding::new();

        // Add variables from left solution
        if let Some(left_binding) = left.first() {
            for (var, term) in left_binding.iter() {
                result_binding.insert(var.clone(), term.clone());
            }
        }

        // Add variables from right solution (overwriting if conflicts)
        if let Some(right_binding) = right.first() {
            for (var, term) in right_binding.iter() {
                result_binding.insert(var.clone(), term.clone());
            }
        }

        Some(vec![result_binding])
    }
}

/// Streaming aggregation operator
pub struct StreamingAggregation {
    #[allow(dead_code)]
    input_stream: Box<dyn DataStream>,
    #[allow(dead_code)]
    group_variables: Vec<Variable>,
    #[allow(dead_code)]
    aggregation_functions: Vec<AggregationFunction>,
    #[allow(dead_code)]
    partial_results: HashMap<String, AggregationState>,
    #[allow(dead_code)]
    memory_monitor: Arc<MemoryMonitor>,
    #[allow(dead_code)]
    spill_manager: Arc<Mutex<SpillManager>>,
    #[allow(dead_code)]
    config: StreamingConfig,
}

/// Aggregation function types
#[derive(Debug, Clone)]
pub enum AggregationFunction {
    Count,
    Sum(Variable),
    Avg(Variable),
    Min(Variable),
    Max(Variable),
    GroupConcat(Variable, Option<String>),
}

/// State for aggregation computation
#[derive(Debug, Clone)]
pub struct AggregationState {
    pub count: usize,
    pub sum: f64,
    pub min: Option<Term>,
    pub max: Option<Term>,
    pub values: Vec<Term>,
}

/// Memory-mapped file stream for large datasets
pub struct MemoryMappedStream {
    #[allow(dead_code)]
    file_path: PathBuf,
    #[allow(dead_code)]
    current_position: usize,
    #[allow(dead_code)]
    total_size: usize,
    #[allow(dead_code)]
    batch_size: usize,
    #[allow(dead_code)]
    stats: StreamStats,
}

/// Compressed spill file stream
pub struct CompressedSpillStream {
    #[allow(dead_code)]
    file_path: PathBuf,
    #[allow(dead_code)]
    reader: Option<Box<dyn BufRead>>,
    #[allow(dead_code)]
    batch_size: usize,
    #[allow(dead_code)]
    stats: StreamStats,
}

impl StreamingExecutor {
    /// Create a new streaming executor
    pub fn new(config: StreamingConfig) -> Result<Self> {
        let temp_dir = TempDir::new()?;
        let memory_monitor = MemoryMonitor::new(config.max_memory_usage);
        let spill_manager = Arc::new(Mutex::new(SpillManager::new(
            temp_dir.path().to_path_buf(),
            config.compression_level,
        )?));

        Ok(Self {
            config,
            memory_monitor,
            spill_manager: spill_manager.clone(),
            temp_dir,
            active_streams: HashMap::new(),
            execution_stats: StreamingStats::default(),
        })
    }

    /// Execute algebra with streaming support
    pub fn execute_streaming(&mut self, algebra: &Algebra) -> Result<Box<dyn DataStream>> {
        let _span = span!(Level::INFO, "streaming_execution").entered();
        let start_time = Instant::now();

        let result_stream = self.execute_algebra_streaming(algebra)?;

        self.execution_stats.total_execution_time += start_time.elapsed();
        Ok(result_stream)
    }

    /// Execute algebra node with streaming
    fn execute_algebra_streaming(&mut self, algebra: &Algebra) -> Result<Box<dyn DataStream>> {
        match algebra {
            Algebra::Join { left, right } => {
                let left_stream = self.execute_algebra_streaming(left)?;
                let right_stream = self.execute_algebra_streaming(right)?;
                let join_variables = self.extract_join_variables(left, right);
                self.create_streaming_join(left_stream, right_stream, join_variables)
            }
            Algebra::Union { left, right } => {
                let left_stream = self.execute_algebra_streaming(left)?;
                let right_stream = self.execute_algebra_streaming(right)?;
                self.create_streaming_union(left_stream, right_stream)
            }
            Algebra::Bgp(patterns) => {
                // Create efficient streaming BGP execution
                self.create_bgp_stream(patterns)
            }
            Algebra::Filter { pattern, condition } => {
                let input_stream = self.execute_algebra_streaming(pattern)?;
                let filtered_stream = StreamingSelection {
                    input: input_stream,
                    condition: condition.clone(),
                };
                Ok(Box::new(filtered_stream))
            }
            _ => Err(anyhow!("Unsupported algebra node for streaming")),
        }
    }

    // Note: These methods are placeholders for future algebra integration
    // Currently commented out as the algebra structure has been updated

    /*
    /// Execute binary operation with streaming
    fn execute_binary_streaming(
        &mut self,
        op: &BinaryOperator,
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
    ) -> Result<Box<dyn DataStream>> {
        // Implementation would depend on actual BinaryOperator definition
        Err(anyhow!("Binary operations not yet implemented"))
    }

    /// Execute unary operation with streaming
    fn execute_unary_streaming(
        &mut self,
        op: &UnaryOperator,
        input: Box<dyn DataStream>,
    ) -> Result<Box<dyn DataStream>> {
        // Implementation would depend on actual UnaryOperator definition
        Err(anyhow!("Unary operations not yet implemented"))
    }
    */

    /// Create streaming hash join
    fn create_streaming_join(
        &mut self,
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
        join_variables: Vec<Variable>,
    ) -> Result<Box<dyn DataStream>> {
        // Choose join algorithm based on estimated sizes
        let left_size = left.estimated_size().unwrap_or(0);
        let right_size = right.estimated_size().unwrap_or(0);

        if left_size + right_size > self.config.max_memory_usage {
            // Use sort-merge join for large datasets
            Ok(Box::new(StreamingSortMergeJoin::new(
                left,
                right,
                join_variables,
                Arc::new(self.memory_monitor.clone()),
                self.spill_manager.clone(),
                self.config.clone(),
            )?))
        } else {
            // Use hash join for smaller datasets
            Ok(Box::new(StreamingHashJoin::new(
                left,
                right,
                join_variables,
                Arc::new(self.memory_monitor.clone()),
                self.spill_manager.clone(),
                self.config.clone(),
            )?))
        }
    }

    /// Create streaming union
    fn create_streaming_union(
        &mut self,
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingUnion::new(left, right)))
    }

    /// Create streaming minus operation
    #[allow(dead_code)]
    fn create_streaming_minus(
        &mut self,
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingMinus::new(
            left,
            right,
            Arc::new(self.memory_monitor.clone()),
            self.spill_manager.clone(),
        )))
    }

    /// Create streaming projection
    #[allow(dead_code)]
    fn create_streaming_projection(
        &mut self,
        input: Box<dyn DataStream>,
        variables: Vec<Variable>,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingProjection::new(input, variables)))
    }

    /// Create streaming selection
    #[allow(dead_code)]
    fn create_streaming_selection(
        &mut self,
        input: Box<dyn DataStream>,
        condition: crate::algebra::Expression,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingSelection::new(input, condition)))
    }

    /// Create streaming sort
    #[allow(dead_code)]
    fn create_streaming_sort(
        &mut self,
        input: Box<dyn DataStream>,
        sort_variables: Vec<Variable>,
    ) -> Result<Box<dyn DataStream>> {
        Ok(Box::new(StreamingSort::new(
            input,
            sort_variables,
            Arc::new(self.memory_monitor.clone()),
            self.spill_manager.clone(),
            self.config.clone(),
        )?))
    }

    /// Create BGP stream with optimized pattern execution
    fn create_bgp_stream(&mut self, patterns: &[TriplePattern]) -> Result<Box<dyn DataStream>> {
        if patterns.is_empty() {
            return Ok(Box::new(EmptyStream::new()));
        }

        if patterns.len() == 1 {
            // Single pattern - create direct stream
            self.create_pattern_stream(&patterns[0])
        } else {
            // Multiple patterns - create join chain with streaming optimization
            self.create_optimized_bgp_join_stream(patterns)
        }
    }

    /// Create optimized BGP join stream
    fn create_optimized_bgp_join_stream(
        &mut self,
        patterns: &[TriplePattern],
    ) -> Result<Box<dyn DataStream>> {
        // Order patterns for optimal streaming (most selective first)
        let mut ordered_patterns = patterns.to_vec();
        self.order_patterns_for_streaming(&mut ordered_patterns);

        // Create initial stream from most selective pattern
        let mut result_stream = self.create_pattern_stream(&ordered_patterns[0])?;

        // Chain remaining patterns with joins
        for pattern in &ordered_patterns[1..] {
            let pattern_stream = self.create_pattern_stream(pattern)?;
            let join_variables = self.find_join_variables_between_streams(
                result_stream.as_ref(),
                pattern_stream.as_ref(),
            )?;
            result_stream =
                self.create_streaming_join(result_stream, pattern_stream, join_variables)?;
        }

        Ok(result_stream)
    }

    /// Create pattern stream from triple pattern
    fn create_pattern_stream(&mut self, pattern: &TriplePattern) -> Result<Box<dyn DataStream>> {
        // Create streaming scan based on pattern selectivity
        let estimated_cardinality = self.estimate_pattern_cardinality(pattern);

        if estimated_cardinality > self.config.max_memory_usage / 1000 {
            // Large result set - use streaming scan with spilling
            Ok(Box::new(StreamingPatternScan::new(
                pattern.clone(),
                Arc::new(self.memory_monitor.clone()),
                self.spill_manager.clone(),
                self.config.clone(),
            )?))
        } else {
            // Small result set - use buffered scan
            Ok(Box::new(BufferedPatternScan::new(
                pattern.clone(),
                self.config.batch_size,
            )?))
        }
    }

    /// Order patterns for optimal streaming execution
    fn order_patterns_for_streaming(&self, patterns: &mut [TriplePattern]) {
        // Sort by estimated cardinality (ascending - most selective first)
        patterns.sort_by(|a, b| {
            let card_a = self.estimate_pattern_cardinality(a);
            let card_b = self.estimate_pattern_cardinality(b);
            card_a.cmp(&card_b)
        });
    }

    /// Estimate pattern cardinality for streaming optimization
    fn estimate_pattern_cardinality(&self, pattern: &TriplePattern) -> usize {
        // Simplified cardinality estimation
        let mut cardinality = 1000000; // Base estimate

        // Reduce cardinality based on bound terms
        if !matches!(pattern.subject, Term::Variable(_)) {
            cardinality /= 100;
        }
        if !matches!(pattern.predicate, Term::Variable(_)) {
            cardinality /= 50;
        }
        if !matches!(pattern.object, Term::Variable(_)) {
            cardinality /= 100;
        }

        cardinality.max(1)
    }

    /// Find join variables between two streams
    fn find_join_variables_between_streams(
        &self,
        _left_stream: &dyn DataStream,
        _right_stream: &dyn DataStream,
    ) -> Result<Vec<Variable>> {
        // Simplified - in practice would analyze stream schemas
        // For now return empty vector indicating cartesian product
        Ok(Vec::new())
    }

    /// Extract variables that are shared between left and right algebra expressions
    fn extract_join_variables(&self, left: &Algebra, right: &Algebra) -> Vec<Variable> {
        let left_vars = self.extract_variables_from_algebra(left);
        let right_vars = self.extract_variables_from_algebra(right);

        // Find intersection of variables
        left_vars
            .into_iter()
            .filter(|var| right_vars.contains(var))
            .collect()
    }

    /// Extract all variables from an algebra expression
    fn extract_variables_from_algebra(&self, algebra: &Algebra) -> Vec<Variable> {
        let mut variables = Vec::new();

        match algebra {
            Algebra::Join { left, right } => {
                variables.extend(self.extract_variables_from_algebra(left));
                variables.extend(self.extract_variables_from_algebra(right));
            }
            Algebra::Union { left, right } => {
                variables.extend(self.extract_variables_from_algebra(left));
                variables.extend(self.extract_variables_from_algebra(right));
            }
            Algebra::Bgp(patterns) => {
                // Extract variables from BGP patterns
                for pattern in patterns {
                    if let Term::Variable(var) = &pattern.subject {
                        variables.push(var.clone());
                    }
                    if let Term::Variable(var) = &pattern.predicate {
                        variables.push(var.clone());
                    }
                    if let Term::Variable(var) = &pattern.object {
                        variables.push(var.clone());
                    }
                }
            }
            Algebra::Filter { pattern, condition } => {
                variables.extend(self.extract_variables_from_algebra(pattern));
                variables.extend(self.extract_variables_from_expression(condition));
            }
            Algebra::Project {
                pattern,
                variables: proj_vars,
            } => {
                variables.extend(self.extract_variables_from_algebra(pattern));
                variables.extend(proj_vars.clone());
            }
            _ => {
                // For other algebra types, we'd need to implement variable extraction
                debug!("Variable extraction not implemented for algebra type");
            }
        }

        // Remove duplicates
        variables.sort();
        variables.dedup();
        variables
    }

    /// Extract variables from a SPARQL expression
    fn extract_variables_from_expression(
        &self,
        expr: &crate::algebra::Expression,
    ) -> Vec<Variable> {
        use crate::algebra::Expression;
        let mut variables = Vec::new();

        match expr {
            Expression::Variable(var) => {
                variables.push(var.clone());
            }
            Expression::Binary { left, right, .. } => {
                variables.extend(self.extract_variables_from_expression(left));
                variables.extend(self.extract_variables_from_expression(right));
            }
            Expression::Unary { operand, .. } => {
                variables.extend(self.extract_variables_from_expression(operand));
            }
            Expression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                variables.extend(self.extract_variables_from_expression(condition));
                variables.extend(self.extract_variables_from_expression(then_expr));
                variables.extend(self.extract_variables_from_expression(else_expr));
            }
            Expression::Bound(var) => {
                variables.push(var.clone());
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    variables.extend(self.extract_variables_from_expression(arg));
                }
            }
            Expression::Exists(algebra) | Expression::NotExists(algebra) => {
                variables.extend(self.extract_variables_from_algebra(algebra));
            }
            _ => {
                // For other expression types (Literal, Iri), no variables to extract
            }
        }

        variables
    }

    /// Get streaming execution statistics
    pub fn get_stats(&self) -> &StreamingStats {
        &self.execution_stats
    }

    /// Clean up temporary files and resources
    pub fn cleanup(&mut self) -> Result<()> {
        self.active_streams.clear();
        self.spill_manager.lock().unwrap().cleanup_all()?;
        Ok(())
    }
}

// Implementation of memory monitor
impl MemoryMonitor {
    fn new(max_allowed: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MemoryMonitorInner {
                current_usage: 0,
                peak_usage: 0,
                max_allowed,
                allocation_history: VecDeque::new(),
            })),
        }
    }

    fn allocate(&self, size: usize, operation: &str) -> bool {
        let mut inner = self.inner.lock().unwrap();
        if inner.current_usage + size > inner.max_allowed {
            return false;
        }

        inner.current_usage += size;
        inner.peak_usage = inner.peak_usage.max(inner.current_usage);

        inner.allocation_history.push_back(MemoryAllocation {
            timestamp: Instant::now(),
            size,
            operation: operation.to_string(),
            freed: false,
        });

        // Keep history bounded
        if inner.allocation_history.len() > 10000 {
            inner.allocation_history.pop_front();
        }

        true
    }

    fn deallocate(&self, size: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.current_usage = inner.current_usage.saturating_sub(size);
    }

    #[allow(dead_code)]
    fn should_spill(&self, threshold: f64) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.current_usage as f64 > inner.max_allowed as f64 * threshold
    }

    #[allow(dead_code)]
    fn get_usage_percentage(&self) -> f64 {
        let inner = self.inner.lock().unwrap();
        inner.current_usage as f64 / inner.max_allowed as f64
    }

    fn get_current_usage(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.current_usage
    }
}

impl Clone for MemoryMonitor {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// Implementation of spill manager
impl SpillManager {
    fn new(spill_directory: PathBuf, compression_level: u32) -> Result<Self> {
        std::fs::create_dir_all(&spill_directory)?;

        Ok(Self {
            spill_directory,
            active_spills: HashMap::new(),
            spill_counter: 0,
            compression_enabled: compression_level > 0,
            compression_level,
        })
    }

    fn spill_data<T: Serialize>(&mut self, data: &T, data_type: SpillDataType) -> Result<String> {
        self.spill_counter += 1;
        let spill_id = format!("spill_{c}", c = self.spill_counter);
        let file_path = self.spill_directory.join(format!("{spill_id}.bin"));

        let start_time = Instant::now();
        let serialized = bincode::serialize(data)?;
        let original_size = serialized.len();

        let final_data = if self.compression_enabled {
            self.compress_data(&serialized)?
        } else {
            serialized
        };

        std::fs::write(&file_path, &final_data)?;

        let spill_info = SpillInfo {
            file_path: file_path.clone(),
            original_size,
            compressed_size: final_data.len(),
            data_type,
            creation_time: start_time,
            access_count: 0,
        };

        self.active_spills.insert(spill_id.clone(), spill_info);
        info!("Spilled {} bytes to {}", original_size, file_path.display());

        Ok(spill_id)
    }

    fn read_spill<T: for<'de> Deserialize<'de>>(&mut self, spill_id: &str) -> Result<T> {
        let spill_info = self
            .active_spills
            .get_mut(spill_id)
            .ok_or_else(|| anyhow!("Spill not found: {}", spill_id))?;

        spill_info.access_count += 1;
        let data = std::fs::read(&spill_info.file_path)?;

        let decompressed = if self.compression_enabled {
            self.decompress_data(&data)?
        } else {
            data
        };

        let deserialized = bincode::deserialize(&decompressed)?;
        Ok(deserialized)
    }

    fn delete_spill(&mut self, spill_id: &str) -> Result<()> {
        if let Some(spill_info) = self.active_spills.remove(spill_id) {
            std::fs::remove_file(&spill_info.file_path)?;
            debug!("Deleted spill file: {}", spill_info.file_path.display());
        }
        Ok(())
    }

    fn cleanup_all(&mut self) -> Result<()> {
        for spill_id in self.active_spills.keys().cloned().collect::<Vec<_>>() {
            self.delete_spill(&spill_id)?;
        }
        Ok(())
    }

    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;

        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }
}

impl Clone for SpillManager {
    fn clone(&self) -> Self {
        Self {
            spill_directory: self.spill_directory.clone(),
            active_spills: self.active_spills.clone(),
            spill_counter: self.spill_counter,
            compression_enabled: self.compression_enabled,
            compression_level: self.compression_level,
        }
    }
}

// Streaming join implementations
impl StreamingHashJoin {
    fn new(
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
        join_variables: Vec<Variable>,
        memory_monitor: Arc<MemoryMonitor>,
        spill_manager: Arc<Mutex<SpillManager>>,
        config: StreamingConfig,
    ) -> Result<Self> {
        Ok(Self {
            left_stream: left,
            right_stream: right,
            join_variables,
            hash_table: HashMap::new(),
            memory_monitor,
            spill_manager,
            config,
            left_exhausted: false,
            current_batch: None,
            spilled_partitions: Vec::new(),
            current_spill_index: 0,
        })
    }
}

impl DataStream for StreamingHashJoin {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        // Implementation of streaming hash join logic with memory monitoring
        if !self.left_exhausted {
            // Build phase: read from left stream and build hash table with spill management
            while let Some(batch) = self.left_stream.next_batch()? {
                for solution in batch {
                    let key = self.extract_join_key(&solution);
                    let estimated_size = std::mem::size_of_val(&solution) + key.len();

                    // Check if we need to spill before adding to hash table
                    if !self
                        .memory_monitor
                        .allocate(estimated_size, "hash_join_build")
                    {
                        // Spill current hash table to disk
                        self.spill_hash_table()?;

                        // Try allocation again after spilling
                        if !self
                            .memory_monitor
                            .allocate(estimated_size, "hash_join_build")
                        {
                            return Err(anyhow!("Cannot allocate memory even after spilling"));
                        }
                    }

                    self.hash_table.entry(key).or_default().push(solution);
                }
            }
            self.left_exhausted = true;
        }

        // Probe phase: read from right stream and probe hash table
        if let Some(right_batch) = self.right_stream.next_batch()? {
            let mut result_batch = Vec::new();

            for right_solution in right_batch {
                let key = self.extract_join_key(&right_solution);
                if let Some(left_solutions) = self.hash_table.get(&key) {
                    for left_solution in left_solutions {
                        if let Some(joined) = self.join_solutions(left_solution, &right_solution) {
                            result_batch.push(joined);
                        }
                    }
                }
            }

            Ok(if result_batch.is_empty() {
                None
            } else {
                Some(result_batch)
            })
        } else {
            Ok(None)
        }
    }

    fn has_more(&self) -> bool {
        !self.left_exhausted || self.right_stream.has_more()
    }

    fn estimated_size(&self) -> Option<usize> {
        None // Dynamic size
    }

    fn reset(&mut self) -> Result<()> {
        self.left_stream.reset()?;
        self.right_stream.reset()?;
        self.hash_table.clear();
        self.left_exhausted = false;
        self.current_batch = None;
        Ok(())
    }

    fn get_stats(&self) -> StreamStats {
        StreamStats::default()
    }
}

impl StreamingHashJoin {
    fn extract_join_key(&self, solution: &Solution) -> String {
        self.join_variables
            .iter()
            .map(|var| {
                Self::get_solution_value(solution, var)
                    .map(|term| format!("{term:?}"))
                    .unwrap_or_else(|| "NULL".to_string())
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    /// Helper function to get a value from a solution
    fn get_solution_value<'a>(solution: &'a Solution, var: &Variable) -> Option<&'a Term> {
        // For streaming, we typically work with the first (primary) binding
        solution.first().and_then(|binding| binding.get(var))
    }

    fn join_solutions(&self, left: &Solution, right: &Solution) -> Option<Solution> {
        // Check that join variables have compatible values
        for var in &self.join_variables {
            let left_val = Self::get_solution_value(left, var);
            let right_val = Self::get_solution_value(right, var);

            match (left_val, right_val) {
                (Some(l), Some(r)) if l != r => return None,
                _ => {}
            }
        }

        // Merge the solutions by combining the first bindings from each
        let mut result_binding = Binding::new();

        // Add variables from left solution
        if let Some(left_binding) = left.first() {
            for (var, term) in left_binding.iter() {
                result_binding.insert(var.clone(), term.clone());
            }
        }

        // Add variables from right solution (overwriting if conflicts)
        if let Some(right_binding) = right.first() {
            for (var, term) in right_binding.iter() {
                result_binding.insert(var.clone(), term.clone());
            }
        }

        Some(vec![result_binding])
    }

    /// Spill current hash table to disk to free memory
    fn spill_hash_table(&mut self) -> Result<()> {
        if self.hash_table.is_empty() {
            return Ok(());
        }

        // Spill the current hash table
        let spill_id = self
            .spill_manager
            .lock()
            .unwrap()
            .spill_data(&self.hash_table, SpillDataType::HashTable)?;

        self.spilled_partitions.push(spill_id);

        // Calculate memory to deallocate
        let total_size: usize = self
            .hash_table
            .iter()
            .map(|(key, solutions)| key.len() + solutions.len() * std::mem::size_of::<Solution>())
            .sum();

        // Clear hash table and deallocate memory
        self.hash_table.clear();
        self.memory_monitor.deallocate(total_size);

        debug!("Spilled hash table partition with {} entries", total_size);
        Ok(())
    }

    /// Load spilled hash table partition back into memory
    #[allow(dead_code)]
    fn load_spilled_partition(&mut self, spill_id: &str) -> Result<HashMap<String, Vec<Solution>>> {
        let partition: HashMap<String, Vec<Solution>> =
            self.spill_manager.lock().unwrap().read_spill(spill_id)?;
        Ok(partition)
    }
}

// Placeholder implementations for other streaming operators
pub struct StreamingUnion {
    left: Box<dyn DataStream>,
    right: Box<dyn DataStream>,
    left_exhausted: bool,
}

impl StreamingUnion {
    fn new(left: Box<dyn DataStream>, right: Box<dyn DataStream>) -> Self {
        Self {
            left,
            right,
            left_exhausted: false,
        }
    }
}

impl DataStream for StreamingUnion {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if !self.left_exhausted {
            if let Some(batch) = self.left.next_batch()? {
                return Ok(Some(batch));
            } else {
                self.left_exhausted = true;
            }
        }
        self.right.next_batch()
    }

    fn has_more(&self) -> bool {
        !self.left_exhausted || self.right.has_more()
    }

    fn estimated_size(&self) -> Option<usize> {
        None
    }

    fn reset(&mut self) -> Result<()> {
        self.left.reset()?;
        self.right.reset()?;
        self.left_exhausted = false;
        Ok(())
    }

    fn get_stats(&self) -> StreamStats {
        StreamStats::default()
    }
}

// Additional placeholder implementations
pub struct StreamingMinus {
    left: Box<dyn DataStream>,
    right: Box<dyn DataStream>,
    #[allow(dead_code)]
    memory_monitor: Arc<MemoryMonitor>,
    #[allow(dead_code)]
    spill_manager: Arc<Mutex<SpillManager>>,
}

impl StreamingMinus {
    #[allow(dead_code)]
    fn new(
        left: Box<dyn DataStream>,
        right: Box<dyn DataStream>,
        memory_monitor: Arc<MemoryMonitor>,
        spill_manager: Arc<Mutex<SpillManager>>,
    ) -> Self {
        Self {
            left,
            right,
            memory_monitor,
            spill_manager,
        }
    }
}

impl DataStream for StreamingMinus {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        // Proper SPARQL MINUS implementation
        if let Some(left_batch) = self.left.next_batch()? {
            let mut filtered_batch = Vec::new();

            // For each solution in the left batch, check if it should be excluded by the right
            for left_solution in left_batch {
                let mut exclude = false;

                // Reset right stream to check all right solutions against this left solution
                self.right.reset()?;

                // Check if any right solution makes this left solution incompatible
                while let Some(right_batch) = self.right.next_batch()? {
                    for right_solution in &right_batch {
                        if self.solutions_compatible(&left_solution, right_solution) {
                            exclude = true;
                            break;
                        }
                    }
                    if exclude {
                        break;
                    }
                }

                // If no compatible right solution found, include the left solution
                if !exclude {
                    filtered_batch.push(left_solution);
                }
            }

            Ok(if filtered_batch.is_empty() {
                None
            } else {
                Some(filtered_batch)
            })
        } else {
            Ok(None)
        }
    }

    fn has_more(&self) -> bool {
        self.left.has_more()
    }

    fn estimated_size(&self) -> Option<usize> {
        self.left.estimated_size()
    }

    fn reset(&mut self) -> Result<()> {
        self.left.reset()?;
        self.right.reset()?;
        Ok(())
    }

    fn get_stats(&self) -> StreamStats {
        let mut stats = self.left.get_stats();
        let right_stats = self.right.get_stats();
        stats.rows_processed += right_stats.rows_processed;
        stats.bytes_processed += right_stats.bytes_processed;
        stats.processing_time += right_stats.processing_time;
        stats
    }
}

impl StreamingMinus {
    /// Check if two solutions are compatible according to SPARQL MINUS semantics
    /// Two solutions are compatible if they don't disagree on any shared variables
    fn solutions_compatible(&self, left: &Solution, right: &Solution) -> bool {
        // Get the first binding from each solution (primary binding)
        let left_binding = match left.first() {
            Some(binding) => binding,
            None => return false,
        };

        let right_binding = match right.first() {
            Some(binding) => binding,
            None => return false,
        };

        // Check all shared variables
        for (var, left_term) in left_binding.iter() {
            if let Some(right_term) = right_binding.get(var) {
                // If they have different values for the same variable, they're incompatible
                if left_term != right_term {
                    return false;
                }
            }
        }

        // If no disagreements found, they are compatible
        true
    }
}

pub struct StreamingProjection {
    input: Box<dyn DataStream>,
    variables: Vec<Variable>,
}

impl StreamingProjection {
    #[allow(dead_code)]
    fn new(input: Box<dyn DataStream>, variables: Vec<Variable>) -> Self {
        Self { input, variables }
    }
}

impl DataStream for StreamingProjection {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if let Some(batch) = self.input.next_batch()? {
            let projected: Vec<Solution> = batch
                .into_iter()
                .map(|solution| {
                    let mut projected_binding = Binding::new();
                    for var in &self.variables {
                        if let Some(term) = StreamingHashJoin::get_solution_value(&solution, var) {
                            projected_binding.insert(var.clone(), term.clone());
                        }
                    }
                    vec![projected_binding]
                })
                .collect();
            Ok(Some(projected))
        } else {
            Ok(None)
        }
    }

    fn has_more(&self) -> bool {
        self.input.has_more()
    }

    fn estimated_size(&self) -> Option<usize> {
        self.input.estimated_size()
    }

    fn reset(&mut self) -> Result<()> {
        self.input.reset()
    }

    fn get_stats(&self) -> StreamStats {
        self.input.get_stats()
    }
}

pub struct StreamingSelection {
    input: Box<dyn DataStream>,
    condition: crate::algebra::Expression,
}

impl StreamingSelection {
    #[allow(dead_code)]
    fn new(input: Box<dyn DataStream>, condition: crate::algebra::Expression) -> Self {
        Self { input, condition }
    }
}

impl DataStream for StreamingSelection {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if let Some(batch) = self.input.next_batch()? {
            let filtered: Vec<Solution> = batch
                .into_iter()
                .filter(|solution| {
                    // Evaluate condition with proper expression evaluation
                    self.evaluate_condition(solution).unwrap_or(false)
                })
                .collect();

            if filtered.is_empty() && self.input.has_more() {
                // Try next batch if this one was completely filtered out
                self.next_batch()
            } else {
                Ok(Some(filtered))
            }
        } else {
            Ok(None)
        }
    }

    fn has_more(&self) -> bool {
        self.input.has_more()
    }

    fn estimated_size(&self) -> Option<usize> {
        self.input.estimated_size()
    }

    fn reset(&mut self) -> Result<()> {
        self.input.reset()
    }

    fn get_stats(&self) -> StreamStats {
        self.input.get_stats()
    }
}

impl StreamingSelection {
    /// Evaluate the filter condition against a solution
    fn evaluate_condition(&self, solution: &Solution) -> Result<bool> {
        use crate::algebra::{BinaryOperator, UnaryOperator};

        // Get the primary binding from the solution
        let binding = match solution.first() {
            Some(binding) => binding,
            None => return Ok(false),
        };

        // Basic expression evaluation
        match &self.condition {
            Expression::Variable(var) => {
                // Variable exists and is bound
                Ok(binding.contains_key(var))
            }
            Expression::Literal(_literal) => {
                // For simplicity, treat all literals as truthy
                // TODO: Implement proper boolean literal evaluation
                Ok(true)
            }
            Expression::Binary { op, left, right } => {
                match op {
                    BinaryOperator::Equal => {
                        let left_val = self.evaluate_expression(left, binding)?;
                        let right_val = self.evaluate_expression(right, binding)?;
                        Ok(left_val == right_val)
                    }
                    BinaryOperator::NotEqual => {
                        let left_val = self.evaluate_expression(left, binding)?;
                        let right_val = self.evaluate_expression(right, binding)?;
                        Ok(left_val != right_val)
                    }
                    BinaryOperator::And => {
                        let left_result = self.evaluate_condition_expr(left, binding)?;
                        let right_result = self.evaluate_condition_expr(right, binding)?;
                        Ok(left_result && right_result)
                    }
                    BinaryOperator::Or => {
                        let left_result = self.evaluate_condition_expr(left, binding)?;
                        let right_result = self.evaluate_condition_expr(right, binding)?;
                        Ok(left_result || right_result)
                    }
                    _ => {
                        // For other binary operators, default to true
                        warn!("Unsupported binary operator in filter: {:?}", op);
                        Ok(true)
                    }
                }
            }
            Expression::Unary { op, operand } => {
                match op {
                    UnaryOperator::Not => {
                        let result = self.evaluate_condition_expr(operand, binding)?;
                        Ok(!result)
                    }
                    _ => {
                        // For other unary operators, default to true
                        warn!("Unsupported unary operator in filter: {:?}", op);
                        Ok(true)
                    }
                }
            }
            Expression::Bound(var) => Ok(binding.contains_key(var)),
            _ => {
                // For other expression types, default to true
                warn!("Unsupported expression type in filter, defaulting to true");
                Ok(true)
            }
        }
    }

    /// Helper to evaluate sub-expressions that return boolean values
    fn evaluate_condition_expr(&self, expr: &Expression, binding: &Binding) -> Result<bool> {
        // Create a temporary solution with just this binding
        let temp_solution = vec![binding.clone()];

        // Use a temporary StreamingSelection to evaluate recursively
        let temp_filter = StreamingSelection {
            input: Box::new(EmptyStream::new()),
            condition: expr.clone(),
        };

        temp_filter.evaluate_condition(&temp_solution)
    }

    /// Helper to evaluate expressions that return Term values
    fn evaluate_expression(&self, expr: &Expression, binding: &Binding) -> Result<Option<Term>> {
        match expr {
            Expression::Variable(var) => Ok(binding.get(var).cloned()),
            Expression::Literal(literal) => {
                // Convert literal to Term
                Ok(Some(Term::Literal(literal.clone())))
            }
            _ => {
                // For complex expressions, we'd need a full expression evaluator
                Ok(None)
            }
        }
    }
}

pub struct StreamingSort {
    input: Box<dyn DataStream>,
    sort_variables: Vec<Variable>,
    #[allow(dead_code)]
    memory_monitor: Arc<MemoryMonitor>,
    spill_manager: Arc<Mutex<SpillManager>>,
    config: StreamingConfig,
    sorted_batches: Vec<String>, // Spill IDs
    current_batch_index: usize,
    fully_sorted: bool,
}

impl StreamingSort {
    #[allow(dead_code)]
    fn new(
        input: Box<dyn DataStream>,
        sort_variables: Vec<Variable>,
        memory_monitor: Arc<MemoryMonitor>,
        spill_manager: Arc<Mutex<SpillManager>>,
        config: StreamingConfig,
    ) -> Result<Self> {
        Ok(Self {
            input,
            sort_variables,
            memory_monitor,
            spill_manager,
            config,
            sorted_batches: Vec::new(),
            current_batch_index: 0,
            fully_sorted: false,
        })
    }
}

impl DataStream for StreamingSort {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if !self.fully_sorted {
            // Sort all input data first (with spilling)
            let mut all_data = Vec::new();

            while let Some(batch) = self.input.next_batch()? {
                all_data.extend(batch);

                // Check if we need to spill
                if all_data.len() > self.config.batch_size {
                    all_data.sort_by(|a, b| self.compare_solutions(a, b));
                    let spill_id = self
                        .spill_manager
                        .lock()
                        .unwrap()
                        .spill_data(&all_data, SpillDataType::SortBuffer)?;
                    self.sorted_batches.push(spill_id);
                    all_data.clear();
                }
            }

            // Handle remaining data
            if !all_data.is_empty() {
                all_data.sort_by(|a, b| self.compare_solutions(a, b));
                let spill_id = self
                    .spill_manager
                    .lock()
                    .unwrap()
                    .spill_data(&all_data, SpillDataType::SortBuffer)?;
                self.sorted_batches.push(spill_id);
            }

            self.fully_sorted = true;
        }

        // Return sorted batches
        if self.current_batch_index < self.sorted_batches.len() {
            let spill_id = &self.sorted_batches[self.current_batch_index];
            let batch: Vec<Solution> = self.spill_manager.lock().unwrap().read_spill(spill_id)?;
            self.current_batch_index += 1;
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }

    fn has_more(&self) -> bool {
        if !self.fully_sorted {
            self.input.has_more()
        } else {
            self.current_batch_index < self.sorted_batches.len()
        }
    }

    fn estimated_size(&self) -> Option<usize> {
        self.input.estimated_size()
    }

    fn reset(&mut self) -> Result<()> {
        self.input.reset()?;
        for spill_id in &self.sorted_batches {
            self.spill_manager.lock().unwrap().delete_spill(spill_id)?;
        }
        self.sorted_batches.clear();
        self.current_batch_index = 0;
        self.fully_sorted = false;
        Ok(())
    }

    fn get_stats(&self) -> StreamStats {
        self.input.get_stats()
    }
}

impl StreamingSort {
    fn compare_solutions(&self, a: &Solution, b: &Solution) -> std::cmp::Ordering {
        for var in &self.sort_variables {
            let a_val = StreamingHashJoin::get_solution_value(a, var);
            let b_val = StreamingHashJoin::get_solution_value(b, var);

            match (a_val, b_val) {
                (Some(a_term), Some(b_term)) => {
                    let cmp = format!("{a_term:?}").cmp(&format!("{b_term:?}"));
                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                (Some(_), None) => return std::cmp::Ordering::Greater,
                (None, Some(_)) => return std::cmp::Ordering::Less,
                (None, None) => continue,
            }
        }
        std::cmp::Ordering::Equal
    }
}

// Empty stream implementation
pub struct EmptyStream {
    exhausted: bool,
}

impl EmptyStream {
    fn new() -> Self {
        Self { exhausted: false }
    }
}

impl DataStream for EmptyStream {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if self.exhausted {
            Ok(None)
        } else {
            self.exhausted = true;
            Ok(Some(vec![]))
        }
    }

    fn has_more(&self) -> bool {
        !self.exhausted
    }

    fn estimated_size(&self) -> Option<usize> {
        Some(0)
    }

    fn reset(&mut self) -> Result<()> {
        self.exhausted = false;
        Ok(())
    }

    fn get_stats(&self) -> StreamStats {
        StreamStats::default()
    }
}

/// Streaming pattern scan for large result sets with spilling support
pub struct StreamingPatternScan {
    pattern: TriplePattern,
    memory_monitor: Arc<MemoryMonitor>,
    spill_manager: Arc<Mutex<SpillManager>>,
    config: StreamingConfig,
    current_batch: Vec<Solution>,
    batch_index: usize,
    total_results: usize,
    spilled_batches: Vec<String>,
}

impl StreamingPatternScan {
    pub fn new(
        pattern: TriplePattern,
        memory_monitor: Arc<MemoryMonitor>,
        spill_manager: Arc<Mutex<SpillManager>>,
        config: StreamingConfig,
    ) -> Result<Self> {
        Ok(Self {
            pattern,
            memory_monitor,
            spill_manager,
            config,
            current_batch: Vec::new(),
            batch_index: 0,
            total_results: 0,
            spilled_batches: Vec::new(),
        })
    }

    /// Generate solutions for the pattern (simplified simulation)
    fn generate_pattern_solutions(&mut self) -> Result<Vec<Solution>> {
        let mut solutions = Vec::new();

        // Generate fewer solutions for more bound patterns
        let solution_count = match (
            matches!(self.pattern.subject, Term::Variable(_)),
            matches!(self.pattern.predicate, Term::Variable(_)),
            matches!(self.pattern.object, Term::Variable(_)),
        ) {
            (true, true, true) => self.config.batch_size * 10,
            (false, true, true) => self.config.batch_size * 5,
            (true, false, true) => self.config.batch_size * 3,
            (true, true, false) => self.config.batch_size * 5,
            (false, false, true) => self.config.batch_size * 2,
            (false, true, false) => self.config.batch_size,
            (true, false, false) => self.config.batch_size * 2,
            (false, false, false) => 1,
        };

        for i in 0..solution_count.min(self.config.batch_size) {
            let mut binding = Binding::new();

            for var in self.pattern.variables() {
                let value =
                    Term::Iri(NamedNode::new(format!("http://example.org/resource_{i}")).unwrap());
                binding.insert(var, value);
            }

            if !binding.is_empty() {
                solutions.push(vec![binding]);
            }
        }

        Ok(solutions)
    }

    /// Check if spilling is needed based on memory pressure
    fn should_spill(&self) -> bool {
        let current_usage = self.memory_monitor.get_current_usage();
        let max_usage = self.memory_monitor.inner.lock().unwrap().max_allowed;
        (current_usage as f64 / max_usage as f64) > self.config.spill_threshold
    }

    /// Spill current batch to disk
    fn spill_current_batch(&mut self) -> Result<()> {
        if !self.current_batch.is_empty() {
            let spill_id = self
                .spill_manager
                .lock()
                .unwrap()
                .spill_data(&self.current_batch, SpillDataType::Solutions)?;
            self.spilled_batches.push(spill_id);
            self.current_batch.clear();

            debug!("Spilled batch {} for pattern scan", self.batch_index);
        }
        Ok(())
    }
}

impl DataStream for StreamingPatternScan {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        // First, try to return any spilled batches
        if !self.spilled_batches.is_empty() {
            let spill_id = self.spilled_batches.remove(0);
            let spilled_solutions: Vec<Solution> =
                self.spill_manager.lock().unwrap().read_spill(&spill_id)?;
            return Ok(Some(spilled_solutions));
        }

        // Generate new solutions if we haven't reached the end
        if self.batch_index < 10 {
            // Limit to 10 batches for demo
            let solutions = self.generate_pattern_solutions()?;

            if solutions.is_empty() {
                return Ok(None);
            }

            // Check if we need to spill due to memory pressure
            if self.should_spill() {
                self.current_batch = solutions;
                self.spill_current_batch()?;
                self.batch_index += 1;
                return self.next_batch(); // Recursive call to get spilled data
            }

            self.batch_index += 1;
            self.total_results += solutions.len();
            Ok(Some(solutions))
        } else {
            Ok(None)
        }
    }

    fn has_more(&self) -> bool {
        !self.spilled_batches.is_empty() || self.batch_index < 10
    }

    fn estimated_size(&self) -> Option<usize> {
        Some(self.total_results + self.current_batch.len())
    }

    fn reset(&mut self) -> Result<()> {
        self.batch_index = 0;
        self.total_results = 0;
        self.current_batch.clear();
        self.spilled_batches.clear();
        Ok(())
    }

    fn get_stats(&self) -> StreamStats {
        StreamStats {
            rows_processed: self.total_results,
            bytes_processed: 0,                      // Estimated
            processing_time: Duration::from_secs(0), // Not tracked here
            spill_operations: self.spilled_batches.len(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

/// Buffered pattern scan for smaller result sets
pub struct BufferedPatternScan {
    pattern: TriplePattern,
    batch_size: usize,
    solutions: Vec<Solution>,
    current_index: usize,
    exhausted: bool,
}

impl BufferedPatternScan {
    pub fn new(pattern: TriplePattern, batch_size: usize) -> Result<Self> {
        let mut scan = Self {
            pattern,
            batch_size,
            solutions: Vec::new(),
            current_index: 0,
            exhausted: false,
        };

        scan.generate_all_solutions()?;
        Ok(scan)
    }

    /// Generate all solutions for the pattern upfront
    fn generate_all_solutions(&mut self) -> Result<()> {
        let solution_count = match (
            matches!(self.pattern.subject, Term::Variable(_)),
            matches!(self.pattern.predicate, Term::Variable(_)),
            matches!(self.pattern.object, Term::Variable(_)),
        ) {
            (true, true, true) => 1000,
            (false, true, true) => 100,
            (true, false, true) => 50,
            (true, true, false) => 100,
            (false, false, true) => 20,
            (false, true, false) => 10,
            (true, false, false) => 20,
            (false, false, false) => 1,
        };

        for i in 0..solution_count {
            let mut binding = Binding::new();

            for var in self.pattern.variables() {
                let value =
                    Term::Iri(NamedNode::new(format!("http://example.org/item_{i}")).unwrap());
                binding.insert(var, value);
            }

            if !binding.is_empty() {
                self.solutions.push(vec![binding]);
            }
        }

        Ok(())
    }
}

impl DataStream for BufferedPatternScan {
    fn next_batch(&mut self) -> Result<Option<Vec<Solution>>> {
        if self.exhausted || self.current_index >= self.solutions.len() {
            return Ok(None);
        }

        let end_index = (self.current_index + self.batch_size).min(self.solutions.len());
        let batch = self.solutions[self.current_index..end_index].to_vec();

        self.current_index = end_index;
        if self.current_index >= self.solutions.len() {
            self.exhausted = true;
        }

        Ok(Some(batch))
    }

    fn has_more(&self) -> bool {
        !self.exhausted && self.current_index < self.solutions.len()
    }

    fn estimated_size(&self) -> Option<usize> {
        Some(self.solutions.len())
    }

    fn reset(&mut self) -> Result<()> {
        self.current_index = 0;
        self.exhausted = false;
        Ok(())
    }

    fn get_stats(&self) -> StreamStats {
        StreamStats {
            rows_processed: self.solutions.len(),
            bytes_processed: 0,                      // Estimated
            processing_time: Duration::from_secs(0), // Not tracked here
            spill_operations: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_executor_creation() {
        let config = StreamingConfig::default();
        let executor = StreamingExecutor::new(config);
        assert!(executor.is_ok());
    }

    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::new(1000);

        assert!(monitor.allocate(500, "test"));
        assert_eq!(monitor.get_current_usage(), 500);

        assert!(monitor.allocate(400, "test2"));
        assert_eq!(monitor.get_current_usage(), 900);

        assert!(!monitor.allocate(200, "test3")); // Should fail

        monitor.deallocate(400);
        assert_eq!(monitor.get_current_usage(), 500);
    }

    #[test]
    fn test_streaming_union() {
        let left = Box::new(EmptyStream::new());
        let right = Box::new(EmptyStream::new());
        let mut union = StreamingUnion::new(left, right);

        let result = union.next_batch().unwrap();
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }
}
