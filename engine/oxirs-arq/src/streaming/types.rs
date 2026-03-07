//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Binding, Expression, Solution, Term, TriplePattern, Variable};
use anyhow::{anyhow, Result};
use oxirs_core::model::NamedNode;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use super::functions::{evaluate_literal_as_boolean, DataStream};

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
/// Streaming pattern scan for large result sets with spilling support
pub struct StreamingPatternScan {
    pattern: TriplePattern,
    memory_monitor: Arc<MemoryMonitor>,
    pub(super) spill_manager: Arc<Mutex<SpillManager>>,
    config: StreamingConfig,
    pub(super) current_batch: Vec<Solution>,
    pub(super) batch_index: usize,
    pub(super) total_results: usize,
    pub(super) spilled_batches: Vec<String>,
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
    pub(super) fn generate_pattern_solutions(&mut self) -> Result<Vec<Solution>> {
        let mut solutions = Vec::new();
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
                let value = Term::Iri(
                    NamedNode::new(format!("http://example.org/resource_{i}"))
                        .expect("generated URL should be valid"),
                );
                binding.insert(var, value);
            }
            if !binding.is_empty() {
                solutions.push(vec![binding]);
            }
        }
        Ok(solutions)
    }
    /// Check if spilling is needed based on memory pressure
    pub(super) fn should_spill(&self) -> bool {
        let current_usage = self.memory_monitor.get_current_usage();
        let max_usage = self
            .memory_monitor
            .inner
            .lock()
            .expect("lock poisoned")
            .max_allowed;
        (current_usage as f64 / max_usage as f64) > self.config.spill_threshold
    }
    /// Spill current batch to disk
    pub(super) fn spill_current_batch(&mut self) -> Result<()> {
        if !self.current_batch.is_empty() {
            let spill_id = self
                .spill_manager
                .lock()
                .expect("lock poisoned")
                .spill_data(&self.current_batch, SpillDataType::Solutions)?;
            self.spilled_batches.push(spill_id);
            self.current_batch.clear();
            debug!("Spilled batch {} for pattern scan", self.batch_index);
        }
        Ok(())
    }
}
/// Buffered pattern scan for smaller result sets
pub struct BufferedPatternScan {
    pattern: TriplePattern,
    pub(super) batch_size: usize,
    pub(super) solutions: Vec<Solution>,
    pub(super) current_index: usize,
    pub(super) exhausted: bool,
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
    pub(super) fn generate_all_solutions(&mut self) -> Result<()> {
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
                let value = Term::Iri(
                    NamedNode::new(format!("http://example.org/item_{i}"))
                        .expect("generated URL should be valid"),
                );
                binding.insert(var, value);
            }
            if !binding.is_empty() {
                self.solutions.push(vec![binding]);
            }
        }
        Ok(())
    }
}
/// Spill manager for handling memory overflow
pub struct SpillManager {
    pub(super) spill_directory: PathBuf,
    pub(super) active_spills: HashMap<String, SpillInfo>,
    pub(super) spill_counter: usize,
    pub(super) compression_enabled: bool,
    pub(super) compression_level: u32,
}
impl SpillManager {
    pub(super) fn new(spill_directory: PathBuf, compression_level: u32) -> Result<Self> {
        std::fs::create_dir_all(&spill_directory)?;
        Ok(Self {
            spill_directory,
            active_spills: HashMap::new(),
            spill_counter: 0,
            compression_enabled: compression_level > 0,
            compression_level,
        })
    }
    pub(super) fn spill_data<T: Serialize>(
        &mut self,
        data: &T,
        data_type: SpillDataType,
    ) -> Result<String> {
        self.spill_counter += 1;
        let spill_id = format!("spill_{c}", c = self.spill_counter);
        let file_path = self.spill_directory.join(format!("{spill_id}.bin"));
        let start_time = Instant::now();
        let serialized = oxicode::serde::encode_to_vec(&data, oxicode::config::standard())?;
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
    pub(super) fn read_spill<T: for<'de> Deserialize<'de>>(&mut self, spill_id: &str) -> Result<T> {
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
        let deserialized =
            oxicode::serde::decode_from_slice(&decompressed, oxicode::config::standard())
                .map(|(v, _)| v)?;
        Ok(deserialized)
    }
    pub(super) fn delete_spill(&mut self, spill_id: &str) -> Result<()> {
        if let Some(spill_info) = self.active_spills.remove(spill_id) {
            std::fs::remove_file(&spill_info.file_path)?;
            debug!("Deleted spill file: {}", spill_info.file_path.display());
        }
        Ok(())
    }
    pub(super) fn cleanup_all(&mut self) -> Result<()> {
        for spill_id in self.active_spills.keys().cloned().collect::<Vec<_>>() {
            self.delete_spill(&spill_id)?;
        }
        Ok(())
    }
    pub(super) fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }
    pub(super) fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        use flate2::read::GzDecoder;
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
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
/// Types of data that can be spilled
#[derive(Debug, Clone)]
pub enum SpillDataType {
    Solutions,
    HashTable,
    SortBuffer,
    IntermediateResults,
    Index,
}
/// Memory monitoring and management
pub struct MemoryMonitor {
    pub(super) inner: Arc<Mutex<MemoryMonitorInner>>,
}
impl MemoryMonitor {
    pub(super) fn new(max_allowed: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MemoryMonitorInner {
                current_usage: 0,
                peak_usage: 0,
                max_allowed,
                allocation_history: VecDeque::new(),
            })),
        }
    }
    pub(super) fn allocate(&self, size: usize, operation: &str) -> bool {
        let mut inner = self.inner.lock().expect("lock poisoned");
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
        if inner.allocation_history.len() > 10000 {
            inner.allocation_history.pop_front();
        }
        true
    }
    pub(super) fn deallocate(&self, size: usize) {
        let mut inner = self.inner.lock().expect("lock poisoned");
        inner.current_usage = inner.current_usage.saturating_sub(size);
    }
    #[allow(dead_code)]
    pub(super) fn should_spill(&self, threshold: f64) -> bool {
        let inner = self.inner.lock().expect("lock poisoned");
        inner.current_usage as f64 > inner.max_allowed as f64 * threshold
    }
    #[allow(dead_code)]
    pub(super) fn get_usage_percentage(&self) -> f64 {
        let inner = self.inner.lock().expect("lock poisoned");
        inner.current_usage as f64 / inner.max_allowed as f64
    }
    pub(super) fn get_current_usage(&self) -> usize {
        let inner = self.inner.lock().expect("lock poisoned");
        inner.current_usage
    }
}
/// Memory allocation record
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    pub timestamp: Instant,
    pub size: usize,
    pub operation: String,
    pub freed: bool,
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
pub struct StreamingUnion {
    pub(super) left: Box<dyn DataStream>,
    pub(super) right: Box<dyn DataStream>,
    pub(super) left_exhausted: bool,
}
impl StreamingUnion {
    pub(super) fn new(left: Box<dyn DataStream>, right: Box<dyn DataStream>) -> Self {
        Self {
            left,
            right,
            left_exhausted: false,
        }
    }
}
pub struct StreamingMinus {
    pub(super) left: Box<dyn DataStream>,
    pub(super) right: Box<dyn DataStream>,
    #[allow(dead_code)]
    memory_monitor: Arc<MemoryMonitor>,
    #[allow(dead_code)]
    spill_manager: Arc<Mutex<SpillManager>>,
}
impl StreamingMinus {
    #[allow(dead_code)]
    pub(super) fn new(
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
impl StreamingMinus {
    /// Check if two solutions are compatible according to SPARQL MINUS semantics
    /// Two solutions are compatible if they don't disagree on any shared variables
    pub(super) fn solutions_compatible(&self, left: &Solution, right: &Solution) -> bool {
        let left_binding = match left.first() {
            Some(binding) => binding,
            None => return false,
        };
        let right_binding = match right.first() {
            Some(binding) => binding,
            None => return false,
        };
        for (var, left_term) in left_binding.iter() {
            if let Some(right_term) = right_binding.get(var) {
                if left_term != right_term {
                    return false;
                }
            }
        }
        true
    }
}
pub struct StreamingSelection {
    pub(super) input: Box<dyn DataStream>,
    pub(super) condition: crate::algebra::Expression,
}
impl StreamingSelection {
    #[allow(dead_code)]
    pub(super) fn new(input: Box<dyn DataStream>, condition: crate::algebra::Expression) -> Self {
        Self { input, condition }
    }
}
impl StreamingSelection {
    /// Evaluate the filter condition against a solution
    pub(super) fn evaluate_condition(&self, solution: &Solution) -> Result<bool> {
        use crate::algebra::{BinaryOperator, UnaryOperator};
        let binding = match solution.first() {
            Some(binding) => binding,
            None => return Ok(false),
        };
        match &self.condition {
            Expression::Variable(var) => Ok(binding.contains_key(var)),
            Expression::Literal(literal) => evaluate_literal_as_boolean(literal),
            Expression::Binary { op, left, right } => match op {
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
                    warn!("Unsupported binary operator in filter: {:?}", op);
                    Ok(true)
                }
            },
            Expression::Unary { op, operand } => match op {
                UnaryOperator::Not => {
                    let result = self.evaluate_condition_expr(operand, binding)?;
                    Ok(!result)
                }
                _ => {
                    warn!("Unsupported unary operator in filter: {:?}", op);
                    Ok(true)
                }
            },
            Expression::Bound(var) => Ok(binding.contains_key(var)),
            _ => {
                warn!("Unsupported expression type in filter, defaulting to true");
                Ok(true)
            }
        }
    }
    /// Helper to evaluate sub-expressions that return boolean values
    pub(super) fn evaluate_condition_expr(
        &self,
        expr: &Expression,
        binding: &Binding,
    ) -> Result<bool> {
        let temp_solution = vec![binding.clone()];
        let temp_filter = StreamingSelection {
            input: Box::new(EmptyStream::new()),
            condition: expr.clone(),
        };
        temp_filter.evaluate_condition(&temp_solution)
    }
    /// Helper to evaluate expressions that return Term values
    pub(super) fn evaluate_expression(
        &self,
        expr: &Expression,
        binding: &Binding,
    ) -> Result<Option<Term>> {
        match expr {
            Expression::Variable(var) => Ok(binding.get(var).cloned()),
            Expression::Literal(literal) => Ok(Some(Term::Literal(literal.clone()))),
            _ => Ok(None),
        }
    }
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
    pub(super) left_stream: Box<dyn DataStream>,
    pub(super) right_stream: Box<dyn DataStream>,
    join_variables: Vec<Variable>,
    pub(super) hash_table: HashMap<String, Vec<Solution>>,
    pub(super) memory_monitor: Arc<MemoryMonitor>,
    spill_manager: Arc<Mutex<SpillManager>>,
    #[allow(dead_code)]
    config: StreamingConfig,
    pub(super) left_exhausted: bool,
    pub(super) current_batch: Option<Vec<Solution>>,
    spilled_partitions: Vec<String>,
    #[allow(dead_code)]
    current_spill_index: usize,
}
impl StreamingHashJoin {
    pub(super) fn new(
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
impl StreamingHashJoin {
    pub(super) fn extract_join_key(&self, solution: &Solution) -> String {
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
    pub(super) fn get_solution_value<'a>(
        solution: &'a Solution,
        var: &Variable,
    ) -> Option<&'a Term> {
        solution.first().and_then(|binding| binding.get(var))
    }
    pub(super) fn join_solutions(&self, left: &Solution, right: &Solution) -> Option<Solution> {
        for var in &self.join_variables {
            let left_val = Self::get_solution_value(left, var);
            let right_val = Self::get_solution_value(right, var);
            match (left_val, right_val) {
                (Some(l), Some(r)) if l != r => return None,
                _ => {}
            }
        }
        let mut result_binding = Binding::new();
        if let Some(left_binding) = left.first() {
            for (var, term) in left_binding.iter() {
                result_binding.insert(var.clone(), term.clone());
            }
        }
        if let Some(right_binding) = right.first() {
            for (var, term) in right_binding.iter() {
                result_binding.insert(var.clone(), term.clone());
            }
        }
        Some(vec![result_binding])
    }
    /// Spill current hash table to disk to free memory
    pub(super) fn spill_hash_table(&mut self) -> Result<()> {
        if self.hash_table.is_empty() {
            return Ok(());
        }
        let spill_id = self
            .spill_manager
            .lock()
            .expect("lock poisoned")
            .spill_data(&self.hash_table, SpillDataType::HashTable)?;
        self.spilled_partitions.push(spill_id);
        let total_size: usize = self
            .hash_table
            .iter()
            .map(|(key, solutions)| key.len() + solutions.len() * std::mem::size_of::<Solution>())
            .sum();
        self.hash_table.clear();
        self.memory_monitor.deallocate(total_size);
        debug!("Spilled hash table partition with {} entries", total_size);
        Ok(())
    }
    /// Load spilled hash table partition back into memory
    #[allow(dead_code)]
    pub(super) fn load_spilled_partition(
        &mut self,
        spill_id: &str,
    ) -> Result<HashMap<String, Vec<Solution>>> {
        let partition: HashMap<String, Vec<Solution>> = self
            .spill_manager
            .lock()
            .expect("lock poisoned")
            .read_spill(spill_id)?;
        Ok(partition)
    }
}
pub struct StreamingProjection {
    pub(super) input: Box<dyn DataStream>,
    pub(super) variables: Vec<Variable>,
}
impl StreamingProjection {
    #[allow(dead_code)]
    pub(super) fn new(input: Box<dyn DataStream>, variables: Vec<Variable>) -> Self {
        Self { input, variables }
    }
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
/// Internal state for memory monitor
pub(super) struct MemoryMonitorInner {
    pub(super) current_usage: usize,
    pub(super) peak_usage: usize,
    pub(super) max_allowed: usize,
    pub(super) allocation_history: VecDeque<MemoryAllocation>,
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
/// Streaming sort-merge join implementation
pub struct StreamingSortMergeJoin {
    pub(super) left_stream: Box<dyn DataStream>,
    pub(super) right_stream: Box<dyn DataStream>,
    join_variables: Vec<Variable>,
    pub(super) left_buffer: VecDeque<Solution>,
    pub(super) right_buffer: VecDeque<Solution>,
    #[allow(dead_code)]
    memory_monitor: Arc<MemoryMonitor>,
    #[allow(dead_code)]
    spill_manager: Arc<Mutex<SpillManager>>,
    #[allow(dead_code)]
    pub(super) config: StreamingConfig,
}
impl StreamingSortMergeJoin {
    pub(super) fn new(
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
impl StreamingSortMergeJoin {
    /// Refill buffers with sorted data from input streams
    pub(super) fn refill_sorted_buffers(&mut self) -> Result<()> {
        if self.left_buffer.is_empty() {
            if let Some(mut batch) = self.left_stream.next_batch()? {
                batch.sort_by(|a, b| self.compare_solution_keys(a, b));
                self.left_buffer.extend(batch);
            }
        }
        if self.right_buffer.is_empty() {
            if let Some(mut batch) = self.right_stream.next_batch()? {
                batch.sort_by(|a, b| self.compare_solution_keys(a, b));
                self.right_buffer.extend(batch);
            }
        }
        Ok(())
    }
    /// Compare join keys of two solutions
    pub(super) fn compare_join_keys(
        &self,
        left: &Solution,
        right: &Solution,
    ) -> std::cmp::Ordering {
        self.compare_solution_keys(left, right)
    }
    /// Compare solutions by their join variable values
    pub(super) fn compare_solution_keys(
        &self,
        left: &Solution,
        right: &Solution,
    ) -> std::cmp::Ordering {
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
    pub(super) fn extract_join_key(&self, solution: &Solution) -> String {
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
    pub(super) fn get_solution_value<'a>(
        solution: &'a Solution,
        var: &Variable,
    ) -> Option<&'a Term> {
        solution.first().and_then(|binding| binding.get(var))
    }
    /// Join two compatible solutions
    pub(super) fn join_solutions(&self, left: &Solution, right: &Solution) -> Option<Solution> {
        for var in &self.join_variables {
            let left_val = Self::get_solution_value(left, var);
            let right_val = Self::get_solution_value(right, var);
            match (left_val, right_val) {
                (Some(l), Some(r)) if l != r => return None,
                _ => {}
            }
        }
        let mut result_binding = Binding::new();
        if let Some(left_binding) = left.first() {
            for (var, term) in left_binding.iter() {
                result_binding.insert(var.clone(), term.clone());
            }
        }
        if let Some(right_binding) = right.first() {
            for (var, term) in right_binding.iter() {
                result_binding.insert(var.clone(), term.clone());
            }
        }
        Some(vec![result_binding])
    }
}
pub struct StreamingSort {
    pub(super) input: Box<dyn DataStream>,
    sort_variables: Vec<Variable>,
    #[allow(dead_code)]
    memory_monitor: Arc<MemoryMonitor>,
    pub(super) spill_manager: Arc<Mutex<SpillManager>>,
    pub(super) config: StreamingConfig,
    pub(super) sorted_batches: Vec<String>,
    pub(super) current_batch_index: usize,
    pub(super) fully_sorted: bool,
}
impl StreamingSort {
    #[allow(dead_code)]
    pub(super) fn new(
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
impl StreamingSort {
    pub(super) fn compare_solutions(&self, a: &Solution, b: &Solution) -> std::cmp::Ordering {
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
pub struct EmptyStream {
    pub(super) exhausted: bool,
}
impl EmptyStream {
    pub(super) fn new() -> Self {
        Self { exhausted: false }
    }
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
