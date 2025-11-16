//! GPU memory pool management for efficient allocation and reuse
//!
//! Enhanced with leak detection, metrics tracking, and adaptive sizing

use super::{GpuBuffer, GpuConfig};
use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// GPU memory pool for efficient buffer management with advanced tracking
#[derive(Debug)]
pub struct GpuMemoryPool {
    device_id: i32,
    available_buffers: Arc<Mutex<VecDeque<GpuBuffer>>>,
    allocated_buffers: Arc<Mutex<Vec<GpuBuffer>>>,
    total_memory: usize,
    used_memory: usize,
    buffer_size: usize,
    max_buffers: usize,
    /// Allocation tracking for leak detection
    allocation_times: Arc<Mutex<Vec<(usize, Instant)>>>,
    /// Performance operation timings
    operation_timings: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    /// Performance metrics
    allocation_count: usize,
    deallocation_count: usize,
    peak_memory_usage: usize,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool with advanced metrics and leak detection
    pub fn new(config: &GpuConfig, buffer_size: usize) -> Result<Self> {
        let max_buffers = config.memory_pool_size / (buffer_size * std::mem::size_of::<f32>());

        Ok(Self {
            device_id: config.device_id,
            available_buffers: Arc::new(Mutex::new(VecDeque::new())),
            allocated_buffers: Arc::new(Mutex::new(Vec::new())),
            total_memory: config.memory_pool_size,
            used_memory: 0,
            buffer_size,
            max_buffers,
            allocation_times: Arc::new(Mutex::new(Vec::new())),
            operation_timings: Arc::new(Mutex::new(HashMap::new())),
            allocation_count: 0,
            deallocation_count: 0,
            peak_memory_usage: 0,
        })
    }

    /// Get a buffer from the pool or allocate a new one (with performance tracking)
    pub fn get_buffer(&mut self) -> Result<GpuBuffer> {
        let start_time = Instant::now();

        // Try to get a buffer from the available pool
        {
            let mut available = self
                .available_buffers
                .lock()
                .map_err(|e| anyhow!("Failed to lock available buffers: {}", e))?;

            if let Some(buffer) = available.pop_front() {
                // Track timing
                let elapsed = start_time.elapsed();
                self.record_operation_time("buffer_acquire_reuse", elapsed);

                // Track allocation for leak detection
                let ptr_value = buffer.ptr() as usize;
                self.allocation_times
                    .lock()
                    .unwrap()
                    .push((ptr_value, Instant::now()));

                return Ok(buffer);
            }
        }

        // No available buffers, check if we can allocate a new one
        if self.allocated_buffers.lock().unwrap().len() >= self.max_buffers {
            let elapsed = start_time.elapsed();
            self.record_operation_time("buffer_acquire_failed", elapsed);
            return Err(anyhow!("Memory pool exhausted"));
        }

        // Allocate a new buffer
        let alloc_start = Instant::now();
        let buffer = GpuBuffer::new(self.buffer_size, self.device_id)?;
        let alloc_elapsed = alloc_start.elapsed();
        self.record_operation_time("buffer_alloc", alloc_elapsed);

        // Update metrics
        self.used_memory += self.buffer_size * std::mem::size_of::<f32>();
        self.allocation_count += 1;
        if self.used_memory > self.peak_memory_usage {
            self.peak_memory_usage = self.used_memory;
        }

        // Track allocation for leak detection
        let ptr_value = buffer.ptr() as usize;
        self.allocation_times
            .lock()
            .unwrap()
            .push((ptr_value, Instant::now()));

        // Record total acquisition time
        let total_elapsed = start_time.elapsed();
        self.record_operation_time("buffer_acquire_new", total_elapsed);

        Ok(buffer)
    }

    /// Record timing for an operation
    fn record_operation_time(&self, operation: &str, duration: Duration) {
        if let Ok(mut timings) = self.operation_timings.lock() {
            timings
                .entry(operation.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    /// Return a buffer to the pool (with performance tracking)
    pub fn return_buffer(&mut self, buffer: GpuBuffer) -> Result<()> {
        let start_time = Instant::now();

        let ptr_value = buffer.ptr() as usize;

        // Remove from allocated buffers
        {
            let mut allocated = self
                .allocated_buffers
                .lock()
                .map_err(|e| anyhow!("Failed to lock allocated buffers: {}", e))?;

            // Find and remove the buffer
            allocated.retain(|b| b.ptr() != buffer.ptr());
        }

        // Remove from allocation tracking
        {
            let mut alloc_times = self.allocation_times.lock().unwrap();
            alloc_times.retain(|(ptr, _)| *ptr != ptr_value);
        }

        // Update metrics
        self.deallocation_count += 1;

        // Add to available buffers
        self.available_buffers
            .lock()
            .map_err(|e| anyhow!("Failed to lock available buffers: {}", e))?
            .push_back(buffer);

        // Record timing
        let elapsed = start_time.elapsed();
        self.record_operation_time("buffer_return", elapsed);

        Ok(())
    }

    /// Get pool statistics
    pub fn stats(&self) -> MemoryPoolStats {
        let allocated_count = self.allocated_buffers.lock().unwrap().len();
        let available_count = self.available_buffers.lock().unwrap().len();

        MemoryPoolStats {
            total_buffers: allocated_count + available_count,
            allocated_buffers: allocated_count,
            available_buffers: available_count,
            total_memory: self.total_memory,
            used_memory: self.used_memory,
            buffer_size: self.buffer_size,
            utilization: if self.total_memory > 0 {
                self.used_memory as f64 / self.total_memory as f64
            } else {
                0.0
            },
        }
    }

    /// Preallocate buffers to warm up the pool
    pub fn preallocate(&mut self, count: usize) -> Result<()> {
        let effective_count = count.min(self.max_buffers);

        for _ in 0..effective_count {
            let buffer = GpuBuffer::new(self.buffer_size, self.device_id)?;
            self.used_memory += self.buffer_size * std::mem::size_of::<f32>();

            self.available_buffers
                .lock()
                .map_err(|e| anyhow!("Failed to lock available buffers: {}", e))?
                .push_back(buffer);
        }

        Ok(())
    }

    /// Clear all buffers and reset the pool
    pub fn clear(&mut self) {
        // Clear all buffers (Drop will handle GPU memory deallocation)
        self.available_buffers.lock().unwrap().clear();
        self.allocated_buffers.lock().unwrap().clear();
        self.used_memory = 0;
    }

    /// Check if pool has available capacity
    pub fn has_capacity(&self) -> bool {
        let total_buffers = self.available_buffers.lock().unwrap().len()
            + self.allocated_buffers.lock().unwrap().len();
        total_buffers < self.max_buffers
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.used_memory
    }

    /// Get memory utilization percentage
    pub fn utilization(&self) -> f64 {
        if self.total_memory > 0 {
            self.used_memory as f64 / self.total_memory as f64
        } else {
            0.0
        }
    }

    /// Defragment the pool by compacting available buffers
    pub fn defragment(&mut self) -> Result<()> {
        let start_time = Instant::now();

        // In a real implementation, this might involve more sophisticated memory management
        // For now, we'll just ensure all available buffers are contiguous in the queue
        let mut available = self
            .available_buffers
            .lock()
            .map_err(|e| anyhow!("Failed to lock available buffers: {}", e))?;

        // Sort available buffers by memory address for better locality
        let mut buffers: Vec<GpuBuffer> = available.drain(..).collect();
        buffers.sort_by_key(|b| b.ptr() as usize);

        for buffer in buffers {
            available.push_back(buffer);
        }

        // Record timing
        let elapsed = start_time.elapsed();
        self.record_operation_time("pool_defrag", elapsed);

        Ok(())
    }

    /// Detect memory leaks (buffers held for too long)
    pub fn detect_leaks(&self, threshold_secs: u64) -> Vec<MemoryLeak> {
        let mut leaks = Vec::new();
        let now = Instant::now();
        let alloc_times = self.allocation_times.lock().unwrap();

        for (ptr, alloc_time) in alloc_times.iter() {
            let duration = now.duration_since(*alloc_time);
            if duration.as_secs() > threshold_secs {
                leaks.push(MemoryLeak {
                    ptr_address: *ptr,
                    allocated_for_secs: duration.as_secs(),
                    buffer_size: self.buffer_size,
                });
            }
        }

        leaks
    }

    /// Get profiling report for memory operations
    pub fn profiling_report(&self) -> String {
        let timings = self.operation_timings.lock().unwrap();
        let mut report = String::from("GPU Memory Pool Performance Report:\n");

        for (operation, durations) in timings.iter() {
            if !durations.is_empty() {
                let total: Duration = durations.iter().sum();
                let avg = total / durations.len() as u32;
                let min = durations.iter().min().unwrap();
                let max = durations.iter().max().unwrap();

                report.push_str(&format!(
                    "  {}: {} calls, avg={:.2}µs, min={:.2}µs, max={:.2}µs\n",
                    operation,
                    durations.len(),
                    avg.as_micros(),
                    min.as_micros(),
                    max.as_micros()
                ));
            }
        }

        report
    }

    /// Get comprehensive metrics
    pub fn get_metrics(&self) -> PoolMetrics {
        PoolMetrics {
            allocation_count: self.allocation_count,
            deallocation_count: self.deallocation_count,
            peak_memory_usage: self.peak_memory_usage,
            current_memory_usage: self.used_memory,
            memory_efficiency: if self.allocation_count > 0 {
                self.deallocation_count as f64 / self.allocation_count as f64
            } else {
                0.0
            },
            active_allocations: self.allocation_times.lock().unwrap().len(),
        }
    }

    /// Adaptive buffer sizing based on usage patterns
    pub fn suggest_optimal_buffer_size(&self) -> usize {
        let metrics = self.get_metrics();

        // If we're frequently allocating/deallocating, suggest smaller buffers
        if metrics.memory_efficiency > 0.95 && self.utilization() < 0.5 {
            self.buffer_size / 2
        }
        // If we're holding memory for long periods, suggest larger buffers
        else if metrics.memory_efficiency < 0.7 && self.utilization() > 0.8 {
            self.buffer_size * 2
        } else {
            self.buffer_size
        }
    }

    /// Reset profiling statistics
    pub fn reset_profiling(&mut self) {
        if let Ok(mut timings) = self.operation_timings.lock() {
            timings.clear();
        }
    }

    /// Get average operation time for specific operation (in microseconds)
    pub fn get_avg_operation_time(&self, operation: &str) -> Option<f64> {
        let timings = self.operation_timings.lock().ok()?;
        let durations = timings.get(operation)?;

        if durations.is_empty() {
            return None;
        }

        let total: Duration = durations.iter().sum();
        let avg = total / durations.len() as u32;
        Some(avg.as_micros() as f64)
    }
}

/// Memory leak detection result
#[derive(Debug, Clone)]
pub struct MemoryLeak {
    /// Pointer address of the leaked buffer
    pub ptr_address: usize,
    /// How long the buffer has been allocated (seconds)
    pub allocated_for_secs: u64,
    /// Size of the leaked buffer
    pub buffer_size: usize,
}

impl MemoryLeak {
    /// Get formatted description of the leak
    pub fn description(&self) -> String {
        format!(
            "Memory leak at 0x{:x}: {} bytes held for {} seconds",
            self.ptr_address, self.buffer_size, self.allocated_for_secs
        )
    }
}

/// Comprehensive pool metrics for performance analysis
#[derive(Debug, Clone)]
pub struct PoolMetrics {
    /// Total number of allocations performed
    pub allocation_count: usize,
    /// Total number of deallocations performed
    pub deallocation_count: usize,
    /// Peak memory usage reached
    pub peak_memory_usage: usize,
    /// Current memory usage
    pub current_memory_usage: usize,
    /// Memory efficiency (deallocations / allocations)
    pub memory_efficiency: f64,
    /// Number of currently active allocations
    pub active_allocations: usize,
}

impl PoolMetrics {
    /// Check if there might be a memory leak
    pub fn has_potential_leak(&self) -> bool {
        self.memory_efficiency < 0.5 && self.active_allocations > 100
    }

    /// Get formatted metrics report
    pub fn report(&self) -> String {
        format!(
            "Pool Metrics:\n\
             - Allocations: {}\n\
             - Deallocations: {}\n\
             - Active: {}\n\
             - Peak memory: {:.2} MB\n\
             - Current memory: {:.2} MB\n\
             - Efficiency: {:.1}%",
            self.allocation_count,
            self.deallocation_count,
            self.active_allocations,
            self.peak_memory_usage as f64 / 1024.0 / 1024.0,
            self.current_memory_usage as f64 / 1024.0 / 1024.0,
            self.memory_efficiency * 100.0
        )
    }
}

/// Statistics about memory pool usage
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub total_buffers: usize,
    pub allocated_buffers: usize,
    pub available_buffers: usize,
    pub total_memory: usize,
    pub used_memory: usize,
    pub buffer_size: usize,
    pub utilization: f64,
}

impl MemoryPoolStats {
    /// Check if the pool is under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        self.utilization > 0.8 || self.available_buffers < 2
    }

    /// Get the number of buffers that can still be allocated
    pub fn remaining_capacity(&self) -> usize {
        if self.total_memory > self.used_memory {
            let remaining_memory = self.total_memory - self.used_memory;
            remaining_memory / (self.buffer_size * std::mem::size_of::<f32>())
        } else {
            0
        }
    }

    /// Print pool statistics
    pub fn print(&self) {
        println!("GPU Memory Pool Statistics:");
        println!("  Total buffers: {}", self.total_buffers);
        println!(
            "  Allocated: {}, Available: {}",
            self.allocated_buffers, self.available_buffers
        );
        println!(
            "  Memory usage: {:.2} MB / {:.2} MB ({:.1}%)",
            self.used_memory as f64 / 1024.0 / 1024.0,
            self.total_memory as f64 / 1024.0 / 1024.0,
            self.utilization * 100.0
        );
        println!(
            "  Buffer size: {:.2} KB",
            self.buffer_size as f64 * 4.0 / 1024.0
        );
        println!(
            "  Remaining capacity: {} buffers",
            self.remaining_capacity()
        );

        if self.is_under_pressure() {
            println!("  ⚠️  Memory pool is under pressure!");
        }
    }
}

/// Advanced memory pool with multiple buffer sizes
#[derive(Debug)]
pub struct AdvancedGpuMemoryPool {
    pools: Vec<GpuMemoryPool>,
    buffer_sizes: Vec<usize>,
    device_id: i32,
}

impl AdvancedGpuMemoryPool {
    /// Create an advanced memory pool with multiple buffer sizes
    pub fn new(config: &GpuConfig, buffer_sizes: Vec<usize>) -> Result<Self> {
        let mut pools = Vec::new();

        for &size in &buffer_sizes {
            let pool = GpuMemoryPool::new(config, size)?;
            pools.push(pool);
        }

        Ok(Self {
            pools,
            buffer_sizes: buffer_sizes.clone(),
            device_id: config.device_id,
        })
    }

    /// Get a buffer of the best fitting size
    pub fn get_buffer(&mut self, required_size: usize) -> Result<GpuBuffer> {
        // Find the smallest buffer size that can accommodate the request
        let pool_index = self
            .buffer_sizes
            .iter()
            .position(|&size| size >= required_size)
            .ok_or_else(|| anyhow!("No buffer size large enough for request"))?;

        self.pools[pool_index].get_buffer()
    }

    /// Return a buffer to the appropriate pool
    pub fn return_buffer(&mut self, buffer: GpuBuffer) -> Result<()> {
        let buffer_size = buffer.size();

        // Find the pool this buffer belongs to
        let pool_index = self
            .buffer_sizes
            .iter()
            .position(|&size| size == buffer_size)
            .ok_or_else(|| anyhow!("Buffer size does not match any pool"))?;

        self.pools[pool_index].return_buffer(buffer)
    }

    /// Get combined statistics for all pools
    pub fn combined_stats(&self) -> AdvancedMemoryPoolStats {
        let mut total_buffers = 0;
        let mut total_allocated = 0;
        let mut total_available = 0;
        let mut total_memory = 0;
        let mut total_used = 0;
        let mut pool_stats = Vec::new();

        for pool in &self.pools {
            let stats = pool.stats();
            total_buffers += stats.total_buffers;
            total_allocated += stats.allocated_buffers;
            total_available += stats.available_buffers;
            total_memory += stats.total_memory;
            total_used += stats.used_memory;
            pool_stats.push(stats);
        }

        AdvancedMemoryPoolStats {
            pool_stats,
            total_buffers,
            total_allocated,
            total_available,
            total_memory,
            total_used,
            utilization: if total_memory > 0 {
                total_used as f64 / total_memory as f64
            } else {
                0.0
            },
        }
    }

    /// Preallocate buffers in all pools
    pub fn preallocate_all(&mut self, buffers_per_pool: usize) -> Result<()> {
        for pool in &mut self.pools {
            pool.preallocate(buffers_per_pool)?;
        }
        Ok(())
    }
}

/// Statistics for advanced memory pool
#[derive(Debug, Clone)]
pub struct AdvancedMemoryPoolStats {
    pub pool_stats: Vec<MemoryPoolStats>,
    pub total_buffers: usize,
    pub total_allocated: usize,
    pub total_available: usize,
    pub total_memory: usize,
    pub total_used: usize,
    pub utilization: f64,
}

impl AdvancedMemoryPoolStats {
    /// Print detailed statistics for all pools
    pub fn print_detailed(&self) {
        println!("Advanced GPU Memory Pool Statistics:");
        println!(
            "  Overall: {} buffers, {:.1}% utilization",
            self.total_buffers,
            self.utilization * 100.0
        );
        println!(
            "  Total memory: {:.2} MB",
            self.total_memory as f64 / 1024.0 / 1024.0
        );

        for (i, stats) in self.pool_stats.iter().enumerate() {
            println!(
                "  Pool {}: {:.2} KB buffers, {} total, {:.1}% util",
                i,
                stats.buffer_size as f64 * 4.0 / 1024.0,
                stats.total_buffers,
                stats.utilization * 100.0
            );
        }
    }
}
