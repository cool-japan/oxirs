//! Memory optimization for SHACL validation
//!
//! This module provides comprehensive memory optimization techniques including:
//! - String interning for reducing memory usage of repeated strings
//! - Compact data structures for efficient memory layout
//! - Memory pool allocation for reducing allocation overhead
//! - Memory monitoring and pressure detection
//! - Garbage collection optimization hints

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use oxirs_core::model::{NamedNode, Term};

use crate::{ConstraintComponentId, PropertyPath, Result, ShaclError, ShapeId};

/// Memory optimization engine for SHACL validation
#[derive(Debug)]
pub struct MemoryOptimizer {
    /// String interner for reducing string duplication
    string_interner: Arc<RwLock<StringInterner>>,

    /// Memory pool for allocations
    memory_pool: Arc<Mutex<MemoryPool>>,

    /// Memory monitoring system
    memory_monitor: MemoryMonitor,

    /// Optimization configuration
    config: MemoryOptimizationConfig,

    /// Performance statistics
    stats: MemoryOptimizationStats,
}

/// Configuration for memory optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// Enable string interning
    pub enable_string_interning: bool,

    /// Enable memory pooling
    pub enable_memory_pooling: bool,

    /// Enable memory monitoring
    pub enable_monitoring: bool,

    /// Maximum memory usage threshold (bytes)
    pub max_memory_threshold: usize,

    /// Memory pressure warning threshold (0.0-1.0)
    pub pressure_warning_threshold: f64,

    /// Memory pressure critical threshold (0.0-1.0)
    pub pressure_critical_threshold: f64,

    /// Enable garbage collection hints
    pub enable_gc_hints: bool,

    /// Memory compaction interval
    pub compaction_interval: Duration,

    /// Pool size limits
    pub pool_limits: PoolLimits,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_string_interning: true,
            enable_memory_pooling: true,
            enable_monitoring: true,
            max_memory_threshold: 1024 * 1024 * 1024, // 1GB
            pressure_warning_threshold: 0.7,
            pressure_critical_threshold: 0.9,
            enable_gc_hints: true,
            compaction_interval: Duration::from_secs(300), // 5 minutes
            pool_limits: PoolLimits::default(),
        }
    }
}

/// Memory pool size limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolLimits {
    /// Maximum number of small objects in pool
    pub max_small_objects: usize,

    /// Maximum number of medium objects in pool
    pub max_medium_objects: usize,

    /// Maximum number of large objects in pool
    pub max_large_objects: usize,

    /// Small object size threshold (bytes)
    pub small_size_threshold: usize,

    /// Medium object size threshold (bytes)
    pub medium_size_threshold: usize,
}

impl Default for PoolLimits {
    fn default() -> Self {
        Self {
            max_small_objects: 10000,
            max_medium_objects: 1000,
            max_large_objects: 100,
            small_size_threshold: 256,
            medium_size_threshold: 4096,
        }
    }
}

/// String interner for reducing memory usage of duplicate strings
#[derive(Debug)]
pub struct StringInterner {
    /// Interned strings map
    strings: HashMap<String, InternedString>,

    /// Reverse lookup for debugging
    reverse_map: HashMap<InternedString, String>,

    /// Next string ID
    next_id: u32,

    /// Statistics
    stats: InternerStats,
}

/// Interned string handle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InternedString(u32);

/// Statistics for string interning
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InternerStats {
    /// Total strings interned
    pub total_strings: usize,

    /// Total memory saved (estimated)
    pub memory_saved: usize,

    /// Number of lookups
    pub lookups: usize,

    /// Number of cache hits
    pub hits: usize,

    /// Average string length
    pub avg_string_length: f64,
}

impl StringInterner {
    /// Create a new string interner
    pub fn new() -> Self {
        Self {
            strings: HashMap::new(),
            reverse_map: HashMap::new(),
            next_id: 0,
            stats: InternerStats::default(),
        }
    }

    /// Intern a string, returning a handle
    pub fn intern(&mut self, s: &str) -> InternedString {
        self.stats.lookups += 1;

        if let Some(&id) = self.strings.get(s) {
            self.stats.hits += 1;
            return id;
        }

        let id = InternedString(self.next_id);
        self.next_id += 1;

        let owned_string = s.to_string();
        self.strings.insert(owned_string.clone(), id);
        self.reverse_map.insert(id, owned_string);

        // Update statistics
        self.stats.total_strings += 1;
        self.stats.memory_saved += s.len(); // Approximation
        self.update_avg_length(s.len());

        id
    }

    /// Get the string for an interned handle
    pub fn get(&self, id: InternedString) -> Option<&String> {
        self.reverse_map.get(&id)
    }

    /// Get interning statistics
    pub fn stats(&self) -> &InternerStats {
        &self.stats
    }

    /// Clear all interned strings
    pub fn clear(&mut self) {
        self.strings.clear();
        self.reverse_map.clear();
        self.next_id = 0;
        self.stats = InternerStats::default();
    }

    /// Compact the interner by removing unused entries
    pub fn compact(&mut self, used_ids: &HashSet<InternedString>) {
        let old_size = self.strings.len();

        // Remove unused strings
        self.strings.retain(|_, &id| used_ids.contains(&id));
        self.reverse_map.retain(|&id, _| used_ids.contains(&id));

        let removed = old_size - self.strings.len();
        self.stats.memory_saved += removed * 50; // Rough estimate
    }

    /// Update average string length
    fn update_avg_length(&mut self, new_length: usize) {
        let total_length = self.stats.avg_string_length * (self.stats.total_strings - 1) as f64
            + new_length as f64;
        self.stats.avg_string_length = total_length / self.stats.total_strings as f64;
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory pool for efficient allocation of validation objects
#[derive(Debug)]
pub struct MemoryPool {
    /// Small object pool
    small_pool: Vec<PooledObject>,

    /// Medium object pool
    medium_pool: Vec<PooledObject>,

    /// Large object pool
    large_pool: Vec<PooledObject>,

    /// Pool statistics
    stats: PoolStats,

    /// Configuration
    limits: PoolLimits,
}

/// Pooled object wrapper
#[derive(Debug)]
struct PooledObject {
    /// Object data
    data: Vec<u8>,

    /// Allocation timestamp
    allocated_at: Instant,

    /// Whether object is currently in use
    in_use: bool,
}

/// Memory pool statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoolStats {
    /// Total allocations
    pub total_allocations: usize,

    /// Pool hits
    pub pool_hits: usize,

    /// Pool misses
    pub pool_misses: usize,

    /// Currently allocated objects
    pub active_objects: usize,

    /// Peak object count
    pub peak_objects: usize,

    /// Total memory managed (bytes)
    pub total_memory_managed: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(limits: PoolLimits) -> Self {
        Self {
            small_pool: Vec::with_capacity(limits.max_small_objects),
            medium_pool: Vec::with_capacity(limits.max_medium_objects),
            large_pool: Vec::with_capacity(limits.max_large_objects),
            stats: PoolStats::default(),
            limits,
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        self.stats.total_allocations += 1;

        // Determine pool type and capacity first
        let pool_capacity = self.pool_capacity(size);

        // Check if we can find an available object
        let pool_index = if size <= self.limits.small_size_threshold {
            0
        } else if size <= self.limits.medium_size_threshold {
            1
        } else {
            2
        };

        let pool = match pool_index {
            0 => &mut self.small_pool,
            1 => &mut self.medium_pool,
            _ => &mut self.large_pool,
        };

        // Try to find an available object in the pool
        for obj in pool.iter_mut() {
            if !obj.in_use && obj.data.capacity() >= size {
                obj.in_use = true;
                obj.allocated_at = Instant::now();
                self.stats.pool_hits += 1;
                self.stats.active_objects += 1;

                if self.stats.active_objects > self.stats.peak_objects {
                    self.stats.peak_objects = self.stats.active_objects;
                }

                return Ok(obj.data.as_mut_ptr());
            }
        }

        // No available object found, create a new one if pool not full
        if pool.len() < pool_capacity {
            let mut data = Vec::with_capacity(size);
            data.resize(size, 0);

            let ptr = data.as_mut_ptr();

            pool.push(PooledObject {
                data,
                allocated_at: Instant::now(),
                in_use: true,
            });

            self.stats.pool_misses += 1;
            self.stats.active_objects += 1;
            self.stats.total_memory_managed += size;

            if self.stats.active_objects > self.stats.peak_objects {
                self.stats.peak_objects = self.stats.active_objects;
            }

            Ok(ptr)
        } else {
            // Pool is full, fall back to regular allocation
            self.stats.pool_misses += 1;
            Err(ShaclError::MemoryPool("Pool capacity exceeded".to_string()))
        }
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&mut self, ptr: *mut u8) {
        // Find the object in pools and mark as not in use
        for pool in [
            &mut self.small_pool,
            &mut self.medium_pool,
            &mut self.large_pool,
        ] {
            for obj in pool.iter_mut() {
                if obj.data.as_mut_ptr() == ptr && obj.in_use {
                    obj.in_use = false;
                    self.stats.active_objects -= 1;
                    return;
                }
            }
        }
    }

    /// Compact the memory pool by removing old unused objects
    pub fn compact(&mut self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;

        let compact_pool = |pool: &mut Vec<PooledObject>| {
            let old_len = pool.len();
            pool.retain(|obj| obj.in_use || obj.allocated_at > cutoff);
            old_len - pool.len()
        };

        let removed = compact_pool(&mut self.small_pool)
            + compact_pool(&mut self.medium_pool)
            + compact_pool(&mut self.large_pool);

        if removed > 0 {
            self.stats.total_memory_managed = self
                .small_pool
                .iter()
                .chain(self.medium_pool.iter())
                .chain(self.large_pool.iter())
                .map(|obj| obj.data.capacity())
                .sum();
        }
    }

    /// Get pool statistics
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Select appropriate pool based on size
    fn select_pool(&mut self, size: usize) -> &mut Vec<PooledObject> {
        if size <= self.limits.small_size_threshold {
            &mut self.small_pool
        } else if size <= self.limits.medium_size_threshold {
            &mut self.medium_pool
        } else {
            &mut self.large_pool
        }
    }

    /// Get pool capacity based on size
    fn pool_capacity(&self, size: usize) -> usize {
        if size <= self.limits.small_size_threshold {
            self.limits.max_small_objects
        } else if size <= self.limits.medium_size_threshold {
            self.limits.max_medium_objects
        } else {
            self.limits.max_large_objects
        }
    }
}

/// Memory monitoring system
#[derive(Debug)]
pub struct MemoryMonitor {
    /// Memory usage samples
    samples: Vec<MemorySample>,

    /// Current memory pressure level
    pressure_level: MemoryPressureLevel,

    /// Monitoring statistics
    stats: MonitoringStats,

    /// Last collection time
    last_collection: Instant,
}

/// Memory usage sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    /// Timestamp
    pub timestamp: Instant,

    /// Memory usage in bytes
    pub memory_usage: usize,

    /// Memory pressure (0.0-1.0)
    pub pressure: f64,
}

/// Memory pressure levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    /// Normal memory usage
    Normal,

    /// Warning level - should start optimizing
    Warning,

    /// Critical level - must take immediate action
    Critical,

    /// Emergency level - risk of out-of-memory
    Emergency,
}

/// Monitoring statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonitoringStats {
    /// Total samples collected
    pub total_samples: usize,

    /// Average memory usage
    pub avg_memory_usage: f64,

    /// Peak memory usage
    pub peak_memory_usage: usize,

    /// Number of pressure warnings
    pub pressure_warnings: usize,

    /// Number of critical pressure events
    pub critical_pressure_events: usize,

    /// Time spent in high pressure
    pub high_pressure_duration: Duration,
}

impl MemoryMonitor {
    /// Create a new memory monitor
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            pressure_level: MemoryPressureLevel::Normal,
            stats: MonitoringStats::default(),
            last_collection: Instant::now(),
        }
    }

    /// Record a memory usage sample
    pub fn record_sample(&mut self, memory_usage: usize, max_memory: usize) {
        let pressure = memory_usage as f64 / max_memory as f64;
        let timestamp = Instant::now();

        self.samples.push(MemorySample {
            timestamp,
            memory_usage,
            pressure,
        });

        // Keep only recent samples (last 1000)
        if self.samples.len() > 1000 {
            self.samples.remove(0);
        }

        // Update pressure level
        let old_level = self.pressure_level.clone();
        self.pressure_level = if pressure >= 0.95 {
            MemoryPressureLevel::Emergency
        } else if pressure >= 0.9 {
            MemoryPressureLevel::Critical
        } else if pressure >= 0.7 {
            MemoryPressureLevel::Warning
        } else {
            MemoryPressureLevel::Normal
        };

        // Update statistics
        self.stats.total_samples += 1;
        self.stats.avg_memory_usage = (self.stats.avg_memory_usage
            * (self.stats.total_samples - 1) as f64
            + memory_usage as f64)
            / self.stats.total_samples as f64;

        if memory_usage > self.stats.peak_memory_usage {
            self.stats.peak_memory_usage = memory_usage;
        }

        if matches!(self.pressure_level, MemoryPressureLevel::Warning)
            && !matches!(old_level, MemoryPressureLevel::Warning)
        {
            self.stats.pressure_warnings += 1;
        }

        if matches!(
            self.pressure_level,
            MemoryPressureLevel::Critical | MemoryPressureLevel::Emergency
        ) && !matches!(
            old_level,
            MemoryPressureLevel::Critical | MemoryPressureLevel::Emergency
        ) {
            self.stats.critical_pressure_events += 1;
        }

        // Update high pressure duration
        if matches!(
            self.pressure_level,
            MemoryPressureLevel::Critical | MemoryPressureLevel::Emergency
        ) {
            self.stats.high_pressure_duration += timestamp.duration_since(self.last_collection);
        }

        self.last_collection = timestamp;
    }

    /// Get current memory pressure level
    pub fn pressure_level(&self) -> &MemoryPressureLevel {
        &self.pressure_level
    }

    /// Get monitoring statistics
    pub fn stats(&self) -> &MonitoringStats {
        &self.stats
    }

    /// Get recent memory usage trend
    pub fn memory_trend(&self) -> MemoryTrend {
        if self.samples.len() < 10 {
            return MemoryTrend::Stable;
        }

        let recent_avg = self
            .samples
            .iter()
            .rev()
            .take(5)
            .map(|s| s.memory_usage)
            .sum::<usize>() as f64
            / 5.0;

        let older_avg = self
            .samples
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|s| s.memory_usage)
            .sum::<usize>() as f64
            / 5.0;

        let change_ratio = (recent_avg - older_avg) / older_avg;

        if change_ratio > 0.1 {
            MemoryTrend::Increasing
        } else if change_ratio < -0.1 {
            MemoryTrend::Decreasing
        } else {
            MemoryTrend::Stable
        }
    }

    /// Clear old samples
    pub fn clear_old_samples(&mut self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;
        self.samples.retain(|sample| sample.timestamp > cutoff);
    }
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage trend
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryTrend {
    /// Memory usage is increasing
    Increasing,

    /// Memory usage is decreasing
    Decreasing,

    /// Memory usage is stable
    Stable,
}

/// Memory optimization statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryOptimizationStats {
    /// String interning statistics
    pub interning_stats: InternerStats,

    /// Memory pool statistics
    pub pool_stats: PoolStats,

    /// Memory monitoring statistics
    pub monitoring_stats: MonitoringStats,

    /// Total memory saved (estimated)
    pub total_memory_saved: usize,

    /// Number of optimization operations
    pub optimization_operations: usize,

    /// Time spent on optimization
    pub optimization_time: Duration,
}

impl MemoryOptimizer {
    /// Create a new memory optimizer
    pub fn new(config: MemoryOptimizationConfig) -> Self {
        Self {
            string_interner: Arc::new(RwLock::new(StringInterner::new())),
            memory_pool: Arc::new(Mutex::new(MemoryPool::new(config.pool_limits.clone()))),
            memory_monitor: MemoryMonitor::new(),
            config,
            stats: MemoryOptimizationStats::default(),
        }
    }

    /// Intern a string for memory optimization
    pub fn intern_string(&mut self, s: &str) -> Result<InternedString> {
        if !self.config.enable_string_interning {
            return Err(ShaclError::MemoryOptimization(
                "String interning is disabled".to_string(),
            ));
        }

        let start = Instant::now();
        let mut interner = self.string_interner.write().unwrap();
        let result = interner.intern(s);

        self.stats.optimization_time += start.elapsed();
        self.stats.optimization_operations += 1;

        Ok(result)
    }

    /// Get an interned string
    pub fn get_interned_string(&self, id: InternedString) -> Option<String> {
        let interner = self.string_interner.read().unwrap();
        interner.get(id).cloned()
    }

    /// Allocate memory from pool
    pub fn pool_allocate(&mut self, size: usize) -> Result<*mut u8> {
        if !self.config.enable_memory_pooling {
            return Err(ShaclError::MemoryOptimization(
                "Memory pooling is disabled".to_string(),
            ));
        }

        let start = Instant::now();
        let mut pool = self.memory_pool.lock().unwrap();
        let result = pool.allocate(size);

        self.stats.optimization_time += start.elapsed();
        self.stats.optimization_operations += 1;

        result
    }

    /// Deallocate memory back to pool
    pub fn pool_deallocate(&mut self, ptr: *mut u8) {
        if self.config.enable_memory_pooling {
            let mut pool = self.memory_pool.lock().unwrap();
            pool.deallocate(ptr);
        }
    }

    /// Record memory usage sample
    pub fn record_memory_usage(&mut self, memory_usage: usize) {
        if self.config.enable_monitoring {
            self.memory_monitor
                .record_sample(memory_usage, self.config.max_memory_threshold);
        }
    }

    /// Get current memory pressure level
    pub fn memory_pressure_level(&self) -> &MemoryPressureLevel {
        self.memory_monitor.pressure_level()
    }

    /// Check if memory pressure is high
    pub fn is_memory_pressure_high(&self) -> bool {
        matches!(
            self.memory_monitor.pressure_level(),
            MemoryPressureLevel::Critical | MemoryPressureLevel::Emergency
        )
    }

    /// Perform memory optimization operations
    pub fn optimize(&mut self) -> Result<OptimizationResult> {
        let start = Instant::now();
        let mut result = OptimizationResult::default();

        // Compact string interner if enabled
        if self.config.enable_string_interning {
            let used_strings = self.collect_used_strings()?;
            let mut interner = self.string_interner.write().unwrap();
            let old_size = interner.strings.len();
            interner.compact(&used_strings);
            let new_size = interner.strings.len();
            result.strings_removed = old_size - new_size;
        }

        // Compact memory pool if enabled
        if self.config.enable_memory_pooling {
            let mut pool = self.memory_pool.lock().unwrap();
            pool.compact(self.config.compaction_interval);
            result.pool_objects_freed =
                pool.stats().total_allocations - pool.stats().active_objects;
        }

        // Trigger garbage collection hint if enabled
        if self.config.enable_gc_hints && self.is_memory_pressure_high() {
            #[cfg(not(target_family = "wasm"))]
            {
                // Force garbage collection (Rust doesn't have direct GC, but we can drop caches)
                self.clear_caches();
                result.gc_triggered = true;
            }
        }

        // Update statistics
        self.update_statistics();

        result.optimization_duration = start.elapsed();
        self.stats.optimization_time += result.optimization_duration;
        self.stats.optimization_operations += 1;

        Ok(result)
    }

    /// Clear all caches to free memory
    pub fn clear_caches(&mut self) {
        if self.config.enable_string_interning {
            let mut interner = self.string_interner.write().unwrap();
            interner.clear();
        }
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> MemoryOptimizationStats {
        let mut stats = self.stats.clone();

        if self.config.enable_string_interning {
            let interner = self.string_interner.read().unwrap();
            stats.interning_stats = interner.stats().clone();
        }

        if self.config.enable_memory_pooling {
            let pool = self.memory_pool.lock().unwrap();
            stats.pool_stats = pool.stats().clone();
        }

        if self.config.enable_monitoring {
            stats.monitoring_stats = self.memory_monitor.stats().clone();
        }

        stats.total_memory_saved = stats.interning_stats.memory_saved;

        stats
    }

    /// Collect currently used string IDs (placeholder implementation)
    fn collect_used_strings(&self) -> Result<HashSet<InternedString>> {
        // In a real implementation, this would traverse all active validation objects
        // to find which interned strings are still in use
        Ok(HashSet::new())
    }

    /// Update internal statistics
    fn update_statistics(&mut self) {
        // Update stats from component statistics
        if self.config.enable_string_interning {
            let interner = self.string_interner.read().unwrap();
            self.stats.interning_stats = interner.stats().clone();
        }

        if self.config.enable_memory_pooling {
            let pool = self.memory_pool.lock().unwrap();
            self.stats.pool_stats = pool.stats().clone();
        }

        if self.config.enable_monitoring {
            self.stats.monitoring_stats = self.memory_monitor.stats().clone();
        }
    }
}

/// Result of memory optimization operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Number of strings removed from interner
    pub strings_removed: usize,

    /// Number of pool objects freed
    pub pool_objects_freed: usize,

    /// Whether garbage collection was triggered
    pub gc_triggered: bool,

    /// Time spent on optimization
    pub optimization_duration: Duration,

    /// Estimated memory freed (bytes)
    pub memory_freed: usize,
}

/// Compact data structures for memory efficiency
pub mod compact {
    use super::*;

    /// Compact representation of validation violations
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CompactViolation {
        /// Focus node (interned)
        pub focus_node: InternedString,

        /// Shape ID (interned)
        pub shape_id: InternedString,

        /// Constraint component (interned)
        pub constraint_component: InternedString,

        /// Severity level (single byte)
        pub severity: u8,

        /// Message (interned if available)
        pub message: Option<InternedString>,
    }

    /// Compact representation of shapes
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CompactShape {
        /// Shape ID (interned)
        pub id: InternedString,

        /// Shape type (single byte)
        pub shape_type: u8,

        /// Target definitions (compact)
        pub targets: Vec<CompactTarget>,

        /// Constraint definitions (compact)
        pub constraints: Vec<CompactConstraint>,
    }

    /// Compact representation of targets
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CompactTarget {
        /// Target type (single byte)
        pub target_type: u8,

        /// Target value (interned)
        pub value: InternedString,
    }

    /// Compact representation of constraints
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct CompactConstraint {
        /// Constraint type (single byte)
        pub constraint_type: u8,

        /// Constraint parameters (binary encoded)
        pub parameters: Vec<u8>,
    }

    /// Converter between regular and compact representations
    #[derive(Debug)]
    pub struct CompactConverter {
        /// String interner for compact representations
        interner: Arc<RwLock<StringInterner>>,
    }

    impl CompactConverter {
        /// Create a new compact converter
        pub fn new(interner: Arc<RwLock<StringInterner>>) -> Self {
            Self { interner }
        }

        /// Convert violation to compact representation
        pub fn compact_violation(
            &self,
            violation: &crate::validation::ValidationViolation,
        ) -> Result<CompactViolation> {
            let mut interner = self.interner.write().unwrap();

            let focus_node = interner.intern(&format!("{:?}", violation.focus_node));
            let shape_id = interner.intern(&violation.source_shape.to_string());
            let constraint_component =
                interner.intern(&violation.source_constraint_component.to_string());

            let severity = match violation.result_severity {
                crate::Severity::Info => 0,
                crate::Severity::Warning => 1,
                crate::Severity::Violation => 2,
            };

            let message = violation
                .result_message
                .as_ref()
                .map(|msg| interner.intern(msg));

            Ok(CompactViolation {
                focus_node,
                shape_id,
                constraint_component,
                severity,
                message,
            })
        }

        /// Convert compact violation back to regular representation
        pub fn expand_violation(
            &self,
            compact: &CompactViolation,
        ) -> Result<crate::validation::ValidationViolation> {
            let interner = self.interner.read().unwrap();

            let focus_node_str = interner.get(compact.focus_node).ok_or_else(|| {
                ShaclError::MemoryOptimization("Invalid focus node reference".to_string())
            })?;

            let shape_id_str = interner.get(compact.shape_id).ok_or_else(|| {
                ShaclError::MemoryOptimization("Invalid shape ID reference".to_string())
            })?;

            let constraint_component_str =
                interner.get(compact.constraint_component).ok_or_else(|| {
                    ShaclError::MemoryOptimization(
                        "Invalid constraint component reference".to_string(),
                    )
                })?;

            // This is a simplified conversion - a real implementation would need
            // proper parsing of the string representations back to their original types
            Err(ShaclError::MemoryOptimization(
                "Compact violation expansion not fully implemented".to_string(),
            ))
        }
    }
}

// Implement required error conversions

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_interner() {
        let mut interner = StringInterner::new();

        let id1 = interner.intern("test_string");
        let id2 = interner.intern("test_string");
        let id3 = interner.intern("different_string");

        assert_eq!(id1, id2); // Same string should return same ID
        assert_ne!(id1, id3); // Different strings should return different IDs

        assert_eq!(interner.get(id1), Some(&"test_string".to_string()));
        assert_eq!(interner.get(id3), Some(&"different_string".to_string()));

        let stats = interner.stats();
        assert_eq!(stats.total_strings, 2); // Only 2 unique strings
        assert_eq!(stats.lookups, 3); // 3 total lookups
        assert_eq!(stats.hits, 1); // 1 cache hit
    }

    #[test]
    fn test_memory_monitor() {
        let mut monitor = MemoryMonitor::new();

        // Test normal pressure
        monitor.record_sample(1000, 10000); // 10% usage
        assert_eq!(monitor.pressure_level(), &MemoryPressureLevel::Normal);

        // Test warning pressure
        monitor.record_sample(7500, 10000); // 75% usage
        assert_eq!(monitor.pressure_level(), &MemoryPressureLevel::Warning);

        // Test critical pressure
        monitor.record_sample(9500, 10000); // 95% usage
        assert_eq!(monitor.pressure_level(), &MemoryPressureLevel::Emergency);

        let stats = monitor.stats();
        assert_eq!(stats.total_samples, 3);
        assert!(stats.peak_memory_usage == 9500);
    }

    #[test]
    fn test_memory_optimizer_config() {
        let config = MemoryOptimizationConfig::default();

        assert!(config.enable_string_interning);
        assert!(config.enable_memory_pooling);
        assert!(config.enable_monitoring);
        assert_eq!(config.max_memory_threshold, 1024 * 1024 * 1024);
        assert_eq!(config.pressure_warning_threshold, 0.7);
        assert_eq!(config.pressure_critical_threshold, 0.9);
    }

    #[test]
    fn test_memory_optimizer_creation() {
        let config = MemoryOptimizationConfig::default();
        let optimizer = MemoryOptimizer::new(config);

        assert_eq!(
            optimizer.memory_pressure_level(),
            &MemoryPressureLevel::Normal
        );
        assert!(!optimizer.is_memory_pressure_high());
    }
}
