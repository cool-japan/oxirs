//! Memory pooling for efficient buffer reuse

use anyhow::{anyhow, Result};
use std::collections::VecDeque;

/// Memory pool for buffer reuse
pub struct MemoryPool {
    max_size: usize,
    current_usage: usize,
    buffers: VecDeque<Vec<u8>>,
}

impl MemoryPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            current_usage: 0,
            buffers: VecDeque::new(),
        }
    }

    /// Allocate a buffer from the pool
    pub fn allocate(&mut self, size: usize) -> Result<PooledBuffer> {
        // Try to find a suitable buffer in the pool
        if let Some(buffer) = self.find_suitable_buffer(size) {
            return Ok(PooledBuffer::Pooled { data: buffer });
        }

        // Check if we have room to allocate
        if self.current_usage + size > self.max_size {
            return Err(anyhow!("Pool exhausted"));
        }

        // Allocate new buffer
        let buffer = vec![0u8; size];
        self.current_usage += size;

        Ok(PooledBuffer::Pooled { data: buffer })
    }

    /// Return a buffer to the pool
    pub fn deallocate(&mut self, buffer: Vec<u8>) {
        let size = buffer.len();

        // Only keep buffers if we have room
        if self.buffers.len() < 100 {
            // Keep up to 100 buffers
            self.buffers.push_back(buffer);
        } else {
            self.current_usage = self.current_usage.saturating_sub(size);
        }
    }

    /// Find a suitable buffer from the pool
    fn find_suitable_buffer(&mut self, size: usize) -> Option<Vec<u8>> {
        // Find buffer that's at least `size` bytes
        for i in 0..self.buffers.len() {
            if self.buffers[i].len() >= size {
                return self.buffers.remove(i);
            }
        }
        None
    }

    /// Get current pool usage
    pub fn current_usage(&self) -> usize {
        self.current_usage
    }

    /// Get available space
    pub fn available(&self) -> usize {
        self.max_size.saturating_sub(self.current_usage)
    }
}

/// Pooled buffer with automatic return to pool
pub enum PooledBuffer {
    Pooled { data: Vec<u8> },
    Heap { data: Vec<u8> },
}

impl PooledBuffer {
    pub fn new_heap(size: usize) -> Result<Self> {
        Ok(Self::Heap {
            data: vec![0u8; size],
        })
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Pooled { data } => data.len(),
            Self::Heap { data } => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Pooled { data } => data,
            Self::Heap { data } => data,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Pooled { data } => data,
            Self::Heap { data } => data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_allocation() {
        let mut pool = MemoryPool::new(10240);

        let buffer = pool.allocate(1024).unwrap();
        assert_eq!(buffer.len(), 1024);
        assert_eq!(pool.current_usage(), 1024);
    }

    #[test]
    fn test_pool_reuse() {
        let mut pool = MemoryPool::new(10240);

        let buffer1 = pool.allocate(1024).unwrap();
        let data = match buffer1 {
            PooledBuffer::Pooled { data } => data,
            PooledBuffer::Heap { data } => data,
        };

        pool.deallocate(data);

        let buffer2 = pool.allocate(1024).unwrap();
        assert_eq!(buffer2.len(), 1024);
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut pool = MemoryPool::new(1024);

        let _b1 = pool.allocate(512).unwrap();
        let _b2 = pool.allocate(512).unwrap();

        // Pool should be exhausted
        let result = pool.allocate(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_available() {
        let mut pool = MemoryPool::new(10240);

        assert_eq!(pool.available(), 10240);

        let _buffer = pool.allocate(1024).unwrap();
        assert_eq!(pool.available(), 9216);
    }
}
