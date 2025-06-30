//! GPU memory pool for efficient memory management

use super::GpuBuffer;
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// GPU memory pool for efficient memory management
pub struct GpuMemoryPool {
    device_id: i32,
    pool_size: usize,
    available_buffers: Arc<Mutex<Vec<GpuBuffer>>>,
    total_allocated: Arc<Mutex<usize>>,
    peak_usage: Arc<Mutex<usize>>,
    current_usage: usize,
    allocation_failures: u64,
}

impl GpuMemoryPool {
    /// Create new GPU memory pool
    pub fn new(pool_size: usize, device_id: i32) -> Result<Self> {
        Ok(Self {
            device_id,
            pool_size,
            available_buffers: Arc::new(Mutex::new(Vec::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            peak_usage: Arc::new(Mutex::new(0)),
            current_usage: 0,
            allocation_failures: 0,
        })
    }

    /// Get a buffer from the pool or allocate a new one
    pub fn get_buffer(&mut self, size: usize) -> Result<GpuBuffer> {
        // Try to reuse an existing buffer first
        {
            let mut buffers = self.available_buffers.lock().unwrap();
            if let Some(buffer) = buffers.pop() {
                if buffer.size() >= size {
                    return Ok(buffer);
                }
                // Buffer too small, put it back and allocate new one
                buffers.push(buffer);
            }
        }

        // Allocate new buffer
        let buffer = GpuBuffer::new(size, self.device_id)?;
        
        // Update statistics
        {
            let mut total = self.total_allocated.lock().unwrap();
            *total += size;
            let mut peak = self.peak_usage.lock().unwrap();
            if *total > *peak {
                *peak = *total;
            }
        }

        Ok(buffer)
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&mut self, buffer: GpuBuffer) {
        let mut buffers = self.available_buffers.lock().unwrap();
        buffers.push(buffer);
    }

    /// Get current pool statistics
    pub fn statistics(&self) -> PoolStatistics {
        let total = *self.total_allocated.lock().unwrap();
        let peak = *self.peak_usage.lock().unwrap();
        let available_count = self.available_buffers.lock().unwrap().len();

        PoolStatistics {
            total_allocated: total,
            peak_usage: peak,
            current_usage: self.current_usage,
            available_buffers: available_count,
            allocation_failures: self.allocation_failures,
        }
    }

    /// Clear the pool and free all buffers
    pub fn clear(&mut self) {
        let mut buffers = self.available_buffers.lock().unwrap();
        buffers.clear();
        
        let mut total = self.total_allocated.lock().unwrap();
        *total = 0;
        
        self.current_usage = 0;
    }
}

/// Statistics for GPU memory pool
#[derive(Debug)]
pub struct PoolStatistics {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub current_usage: usize,
    pub available_buffers: usize,
    pub allocation_failures: u64,
}