//! Memory protection and management for AI operations

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

/// Secure memory wrapper that zeros on drop
pub struct SecureMemory {
    data: Vec<u8>,
}

impl SecureMemory {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    pub fn from_vec(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Drop for SecureMemory {
    fn drop(&mut self) {
        // Zero memory before deallocation
        self.data.iter_mut().for_each(|b| *b = 0);
    }
}

/// Memory guard for tracking and limiting memory usage
pub struct MemoryGuard {
    max_bytes: usize,
    current_usage: Arc<RwLock<usize>>,
    allocations: Arc<RwLock<Vec<AllocationInfo>>>,
}

impl MemoryGuard {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            current_usage: Arc::new(RwLock::new(0)),
            allocations: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Check if allocation is allowed
    pub fn can_allocate(&self, bytes: usize) -> bool {
        let current = self.current_usage.read()
            .map(|usage| *usage)
            .unwrap_or(0);

        current + bytes <= self.max_bytes
    }

    /// Record allocation
    pub fn allocate(&self, bytes: usize, purpose: &str) -> Result<AllocationHandle> {
        if !self.can_allocate(bytes) {
            return Err(anyhow!(
                "Memory limit exceeded: requested {} bytes, available {} bytes",
                bytes,
                self.max_bytes - self.current_usage()
            ));
        }

        let mut current = self.current_usage.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        let mut allocations = self.allocations.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        let allocation = AllocationInfo {
            id: allocations.len(),
            bytes,
            purpose: purpose.to_string(),
            allocated_at: std::time::Instant::now(),
        };

        allocations.push(allocation.clone());
        *current += bytes;

        Ok(AllocationHandle {
            id: allocation.id,
            bytes,
            guard: self.clone_guard(),
        })
    }

    /// Record deallocation
    pub fn deallocate(&self, handle: AllocationHandle) -> Result<()> {
        let mut current = self.current_usage.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        let mut allocations = self.allocations.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        // Find and remove allocation
        if let Some(pos) = allocations.iter().position(|a| a.id == handle.id) {
            allocations.remove(pos);
            *current = current.saturating_sub(handle.bytes);
        }

        Ok(())
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage.read()
            .map(|usage| *usage)
            .unwrap_or(0)
    }

    /// Get available memory
    pub fn available_memory(&self) -> usize {
        self.max_bytes.saturating_sub(self.current_usage())
    }

    /// Get memory statistics
    pub fn statistics(&self) -> Result<MemoryStatistics> {
        let current = self.current_usage();
        let allocations = self.allocations.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        let total_allocations = allocations.len();
        let largest_allocation = allocations
            .iter()
            .map(|a| a.bytes)
            .max()
            .unwrap_or(0);

        let oldest_allocation = allocations
            .iter()
            .map(|a| a.allocated_at.elapsed().as_secs())
            .max()
            .unwrap_or(0);

        Ok(MemoryStatistics {
            current_usage: current,
            max_limit: self.max_bytes,
            available: self.available_memory(),
            utilization_percent: (current as f64 / self.max_bytes as f64) * 100.0,
            total_allocations,
            largest_allocation,
            oldest_allocation_age_secs: oldest_allocation,
        })
    }

    fn clone_guard(&self) -> Arc<RwLock<usize>> {
        Arc::clone(&self.current_usage)
    }
}

/// Allocation handle for tracking memory
#[derive(Debug)]
pub struct AllocationHandle {
    id: usize,
    bytes: usize,
    guard: Arc<RwLock<usize>>,
}

impl Drop for AllocationHandle {
    fn drop(&mut self) {
        // Auto-deallocate on drop
        if let Ok(mut current) = self.guard.write() {
            *current = current.saturating_sub(self.bytes);
        }
    }
}

/// Allocation information
#[derive(Debug, Clone)]
struct AllocationInfo {
    id: usize,
    bytes: usize,
    purpose: String,
    allocated_at: std::time::Instant,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    pub current_usage: usize,
    pub max_limit: usize,
    pub available: usize,
    pub utilization_percent: f64,
    pub total_allocations: usize,
    pub largest_allocation: usize,
    pub oldest_allocation_age_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_memory_zeros_on_drop() {
        let mem = SecureMemory::from_vec(vec![1, 2, 3, 4, 5]);
        assert_eq!(mem.as_slice(), &[1, 2, 3, 4, 5]);
        drop(mem);
        // Memory should be zeroed (can't test directly after drop)
    }

    #[test]
    fn test_memory_guard_allocation() {
        let guard = MemoryGuard::new(1024);

        assert!(guard.can_allocate(512));
        let handle = guard.allocate(512, "test").unwrap();
        assert_eq!(guard.current_usage(), 512);

        drop(handle);
        // Usage should decrease (auto-deallocate on drop)
    }

    #[test]
    fn test_memory_guard_limit() {
        let guard = MemoryGuard::new(1024);

        // Allocate up to limit
        let _h1 = guard.allocate(512, "test1").unwrap();
        let _h2 = guard.allocate(512, "test2").unwrap();

        // Should fail to allocate more
        assert!(guard.allocate(1, "test3").is_err());
    }

    #[test]
    fn test_memory_statistics() {
        let guard = MemoryGuard::new(1024);

        let _h1 = guard.allocate(256, "embedding").unwrap();
        let _h2 = guard.allocate(512, "model").unwrap();

        let stats = guard.statistics().unwrap();
        assert_eq!(stats.current_usage, 768);
        assert_eq!(stats.available, 256);
        assert_eq!(stats.total_allocations, 2);
        assert_eq!(stats.largest_allocation, 512);
    }

    #[test]
    fn test_secure_memory_operations() {
        let mut mem = SecureMemory::new(10);
        assert_eq!(mem.len(), 10);
        assert!(!mem.is_empty());

        mem.as_mut_slice()[0] = 42;
        assert_eq!(mem.as_slice()[0], 42);
    }
}
