//! GPU memory pool management for efficient allocation and reuse

use super::{GpuBuffer, GpuConfig};
use anyhow::{anyhow, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// GPU memory pool for efficient buffer management
#[derive(Debug)]
pub struct GpuMemoryPool {
    device_id: i32,
    available_buffers: Arc<Mutex<VecDeque<GpuBuffer>>>,
    allocated_buffers: Arc<Mutex<Vec<GpuBuffer>>>,
    total_memory: usize,
    used_memory: usize,
    buffer_size: usize,
    max_buffers: usize,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool
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
        })
    }

    /// Get a buffer from the pool or allocate a new one
    pub fn get_buffer(&mut self) -> Result<GpuBuffer> {
        // Try to get a buffer from the available pool
        {
            let mut available = self
                .available_buffers
                .lock()
                .map_err(|e| anyhow!("Failed to lock available buffers: {}", e))?;

            if let Some(buffer) = available.pop_front() {
                // Move to allocated buffers
                self.allocated_buffers
                    .lock()
                    .map_err(|e| anyhow!("Failed to lock allocated buffers: {}", e))?
                    .push(buffer);

                return Ok(self.allocated_buffers.lock().unwrap().last().unwrap());
            }
        }

        // No available buffers, check if we can allocate a new one
        if self.allocated_buffers.lock().unwrap().len() >= self.max_buffers {
            return Err(anyhow!("Memory pool exhausted"));
        }

        // Allocate a new buffer
        let buffer = GpuBuffer::new(self.buffer_size, self.device_id)?;
        self.used_memory += self.buffer_size * std::mem::size_of::<f32>();

        self.allocated_buffers
            .lock()
            .map_err(|e| anyhow!("Failed to lock allocated buffers: {}", e))?
            .push(buffer);

        Ok(self.allocated_buffers.lock().unwrap().last().unwrap())
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer: GpuBuffer) -> Result<()> {
        // Remove from allocated buffers
        {
            let mut allocated = self
                .allocated_buffers
                .lock()
                .map_err(|e| anyhow!("Failed to lock allocated buffers: {}", e))?;

            // Find and remove the buffer (simplified - in practice would use better identification)
            allocated.retain(|b| b.ptr() != buffer.ptr());
        }

        // Add to available buffers
        self.available_buffers
            .lock()
            .map_err(|e| anyhow!("Failed to lock available buffers: {}", e))?
            .push_back(buffer);

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

        Ok(())
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
