//! Advanced Buffer Pool Management
//!
//! This module provides specialized buffer pools for frequently allocated objects
//! to reduce memory pressure and improve performance by minimizing allocations.

use crate::algebra::{Binding, Solution, Term, Variable};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Manager for multiple specialized buffer pools
pub struct BufferPoolManager {
    solution_pool: Arc<Mutex<BufferPool<Solution>>>,
    binding_pool: Arc<Mutex<BufferPool<Binding>>>,
    hashmap_pool: Arc<Mutex<BufferPool<HashMap<Variable, Term>>>>,
    vector_pool: Arc<Mutex<BufferPool<Vec<Binding>>>>,
    stats: Arc<Mutex<BufferPoolStats>>,
}

impl BufferPoolManager {
    /// Create a new buffer pool manager with default configurations
    pub fn new() -> Self {
        Self::with_capacities(1000, 2000, 500, 1000)
    }

    /// Create with custom pool capacities
    pub fn with_capacities(
        solution_capacity: usize,
        binding_capacity: usize,
        hashmap_capacity: usize,
        vector_capacity: usize,
    ) -> Self {
        Self {
            solution_pool: Arc::new(Mutex::new(BufferPool::new_for_solutions(solution_capacity))),
            binding_pool: Arc::new(Mutex::new(BufferPool::new_for_bindings(binding_capacity))),
            hashmap_pool: Arc::new(Mutex::new(BufferPool::new_for_hashmaps(hashmap_capacity))),
            vector_pool: Arc::new(Mutex::new(BufferPool::new_for_vectors(vector_capacity))),
            stats: Arc::new(Mutex::new(BufferPoolStats::default())),
        }
    }

    /// Get a solution from the pool (or create new one)
    pub fn acquire_solution(&self) -> PooledSolution {
        let solution = self.solution_pool.lock().unwrap().acquire();
        self.update_stats("solution", true);

        PooledSolution {
            solution: Some(solution),
            pool: Arc::clone(&self.solution_pool),
            manager: Some(self),
        }
    }

    /// Get a binding from the pool (or create new one)
    pub fn acquire_binding(&self) -> PooledBinding {
        let binding = self.binding_pool.lock().unwrap().acquire();
        self.update_stats("binding", true);

        PooledBinding {
            binding: Some(binding),
            pool: Arc::clone(&self.binding_pool),
            manager: Some(self),
        }
    }

    /// Get a hashmap from the pool (or create new one)
    pub fn acquire_hashmap(&self) -> PooledHashMap {
        let hashmap = self.hashmap_pool.lock().unwrap().acquire();
        self.update_stats("hashmap", true);

        PooledHashMap {
            hashmap: Some(hashmap),
            pool: Arc::clone(&self.hashmap_pool),
            manager: Some(self),
        }
    }

    /// Get a vector from the pool (or create new one)
    pub fn acquire_vector(&self) -> PooledVector {
        let vector = self.vector_pool.lock().unwrap().acquire();
        self.update_stats("vector", true);

        PooledVector {
            vector: Some(vector),
            pool: Arc::clone(&self.vector_pool),
            manager: Some(self),
        }
    }

    /// Update pool statistics
    fn update_stats(&self, pool_type: &str, is_acquire: bool) {
        let mut stats = self.stats.lock().unwrap();

        let pool_stats = match pool_type {
            "solution" => &mut stats.solution_stats,
            "binding" => &mut stats.binding_stats,
            "hashmap" => &mut stats.hashmap_stats,
            "vector" => &mut stats.vector_stats,
            _ => return,
        };

        if is_acquire {
            pool_stats.total_acquires += 1;
        } else {
            pool_stats.total_returns += 1;
        }
    }

    /// Get comprehensive statistics for all pools
    pub fn get_stats(&self) -> BufferPoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all pools (useful for memory cleanup)
    pub fn clear_all_pools(&self) {
        self.solution_pool.lock().unwrap().clear();
        self.binding_pool.lock().unwrap().clear();
        self.hashmap_pool.lock().unwrap().clear();
        self.vector_pool.lock().unwrap().clear();
    }

    /// Get total memory usage estimate for all pools
    pub fn estimate_total_memory_usage(&self) -> usize {
        let solution_mem = self.solution_pool.lock().unwrap().estimate_memory_usage()
            * std::mem::size_of::<Solution>();
        let binding_mem = self.binding_pool.lock().unwrap().estimate_memory_usage()
            * std::mem::size_of::<Binding>();
        let hashmap_mem = self.hashmap_pool.lock().unwrap().estimate_memory_usage()
            * std::mem::size_of::<HashMap<Variable, Term>>();
        let vector_mem = self.vector_pool.lock().unwrap().estimate_memory_usage()
            * std::mem::size_of::<Vec<Binding>>();

        solution_mem + binding_mem + hashmap_mem + vector_mem
    }
}

impl Default for BufferPoolManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Generic buffer pool for reusing objects
struct BufferPool<T> {
    available: VecDeque<T>,
    max_capacity: usize,
    factory: fn() -> T,
}

impl<T> BufferPool<T> {
    #[allow(dead_code)]
    fn new(max_capacity: usize) -> Self
    where
        T: Default,
    {
        Self {
            available: VecDeque::new(),
            max_capacity,
            factory: T::default,
        }
    }

    fn acquire(&mut self) -> T {
        self.available
            .pop_front()
            .unwrap_or_else(|| (self.factory)())
    }

    #[allow(dead_code)]
    fn return_object(&mut self, mut obj: T) {
        if self.available.len() < self.max_capacity {
            // Reset object to clean state before returning to pool
            self.reset_object(&mut obj);
            self.available.push_back(obj);
        }
        // If pool is full, just drop the object
    }

    fn clear(&mut self) {
        self.available.clear();
    }

    fn estimate_memory_usage(&self) -> usize {
        self.available.len()
    }

    #[allow(dead_code)]
    fn reset_object(&self, _obj: &mut T) {
        // Default implementation does nothing
        // Specialized pools will override this behavior
    }
}

// Specialized buffer pool implementations with proper reset behavior
impl BufferPool<Solution> {
    fn new_for_solutions(max_capacity: usize) -> Self {
        Self {
            available: VecDeque::new(),
            max_capacity,
            factory: Vec::new,
        }
    }

    fn return_solution(&mut self, mut solution: Solution) {
        if self.available.len() < self.max_capacity {
            solution.clear(); // Reset to clean state
            self.available.push_back(solution);
        }
    }
}

impl BufferPool<Binding> {
    fn new_for_bindings(max_capacity: usize) -> Self {
        Self {
            available: VecDeque::new(),
            max_capacity,
            factory: HashMap::new,
        }
    }

    fn return_binding(&mut self, mut binding: Binding) {
        if self.available.len() < self.max_capacity {
            binding.clear(); // Reset to clean state
            self.available.push_back(binding);
        }
    }
}

impl BufferPool<HashMap<Variable, Term>> {
    fn new_for_hashmaps(max_capacity: usize) -> Self {
        Self {
            available: VecDeque::new(),
            max_capacity,
            factory: HashMap::new,
        }
    }

    fn return_hashmap(&mut self, mut hashmap: HashMap<Variable, Term>) {
        if self.available.len() < self.max_capacity {
            hashmap.clear(); // Reset to clean state
            self.available.push_back(hashmap);
        }
    }
}

impl BufferPool<Vec<Binding>> {
    fn new_for_vectors(max_capacity: usize) -> Self {
        Self {
            available: VecDeque::new(),
            max_capacity,
            factory: Vec::new,
        }
    }

    fn return_vector(&mut self, mut vector: Vec<Binding>) {
        if self.available.len() < self.max_capacity {
            vector.clear(); // Reset to clean state
            self.available.push_back(vector);
        }
    }
}

/// RAII wrapper for pooled solutions
pub struct PooledSolution<'a> {
    solution: Option<Solution>,
    pool: Arc<Mutex<BufferPool<Solution>>>,
    manager: Option<&'a BufferPoolManager>,
}

impl<'a> PooledSolution<'a> {
    pub fn get_mut(&mut self) -> &mut Solution {
        self.solution.as_mut().unwrap()
    }

    pub fn get(&self) -> &Solution {
        self.solution.as_ref().unwrap()
    }
}

impl<'a> Drop for PooledSolution<'a> {
    fn drop(&mut self) {
        if let Some(solution) = self.solution.take() {
            self.pool.lock().unwrap().return_solution(solution);
            if let Some(manager) = self.manager {
                manager.update_stats("solution", false);
            }
        }
    }
}

/// RAII wrapper for pooled bindings
pub struct PooledBinding<'a> {
    binding: Option<Binding>,
    pool: Arc<Mutex<BufferPool<Binding>>>,
    manager: Option<&'a BufferPoolManager>,
}

impl<'a> PooledBinding<'a> {
    pub fn get_mut(&mut self) -> &mut Binding {
        self.binding.as_mut().unwrap()
    }

    pub fn get(&self) -> &Binding {
        self.binding.as_ref().unwrap()
    }
}

impl<'a> Drop for PooledBinding<'a> {
    fn drop(&mut self) {
        if let Some(binding) = self.binding.take() {
            self.pool.lock().unwrap().return_binding(binding);
            if let Some(manager) = self.manager {
                manager.update_stats("binding", false);
            }
        }
    }
}

/// RAII wrapper for pooled hashmaps
pub struct PooledHashMap<'a> {
    hashmap: Option<HashMap<Variable, Term>>,
    pool: Arc<Mutex<BufferPool<HashMap<Variable, Term>>>>,
    manager: Option<&'a BufferPoolManager>,
}

impl<'a> PooledHashMap<'a> {
    pub fn get_mut(&mut self) -> &mut HashMap<Variable, Term> {
        self.hashmap.as_mut().unwrap()
    }

    pub fn get(&self) -> &HashMap<Variable, Term> {
        self.hashmap.as_ref().unwrap()
    }
}

impl<'a> Drop for PooledHashMap<'a> {
    fn drop(&mut self) {
        if let Some(hashmap) = self.hashmap.take() {
            self.pool.lock().unwrap().return_hashmap(hashmap);
            if let Some(manager) = self.manager {
                manager.update_stats("hashmap", false);
            }
        }
    }
}

/// RAII wrapper for pooled vectors
pub struct PooledVector<'a> {
    vector: Option<Vec<Binding>>,
    pool: Arc<Mutex<BufferPool<Vec<Binding>>>>,
    manager: Option<&'a BufferPoolManager>,
}

impl<'a> PooledVector<'a> {
    pub fn get_mut(&mut self) -> &mut Vec<Binding> {
        self.vector.as_mut().unwrap()
    }

    pub fn get(&self) -> &Vec<Binding> {
        self.vector.as_ref().unwrap()
    }
}

impl<'a> Drop for PooledVector<'a> {
    fn drop(&mut self) {
        if let Some(vector) = self.vector.take() {
            self.pool.lock().unwrap().return_vector(vector);
            if let Some(manager) = self.manager {
                manager.update_stats("vector", false);
            }
        }
    }
}

/// Statistics for buffer pool usage
#[derive(Debug, Clone, Default)]
pub struct BufferPoolStats {
    pub solution_stats: PoolStats,
    pub binding_stats: PoolStats,
    pub hashmap_stats: PoolStats,
    pub vector_stats: PoolStats,
}

#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub total_acquires: usize,
    pub total_returns: usize,
    pub current_size: usize,
}

impl BufferPoolStats {
    /// Get total number of acquisitions across all pools
    pub fn total_acquisitions(&self) -> usize {
        self.solution_stats.total_acquires
            + self.binding_stats.total_acquires
            + self.hashmap_stats.total_acquires
            + self.vector_stats.total_acquires
    }

    /// Get total number of returns across all pools
    pub fn total_returns(&self) -> usize {
        self.solution_stats.total_returns
            + self.binding_stats.total_returns
            + self.hashmap_stats.total_returns
            + self.vector_stats.total_returns
    }

    /// Calculate hit rate (objects returned from pool vs newly created)
    pub fn hit_rate(&self) -> f64 {
        let total_acquires = self.total_acquisitions();
        if total_acquires == 0 {
            return 0.0;
        }

        let total_returns = self.total_returns();
        total_returns as f64 / total_acquires as f64
    }

    /// Get performance summary
    pub fn performance_summary(&self) -> String {
        format!(
            "Buffer Pool Stats: {} acquisitions, {} returns, {:.2}% hit rate",
            self.total_acquisitions(),
            self.total_returns(),
            self.hit_rate() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_manager() {
        let manager = BufferPoolManager::new();

        // Test solution pooling
        {
            let mut pooled_solution = manager.acquire_solution();
            let solution = pooled_solution.get_mut();

            // Use the solution
            let mut binding = Binding::new();
            binding.insert(
                Variable::new("x").unwrap(),
                Term::Iri(oxirs_core::model::NamedNode::new_unchecked(
                    "http://example.org/test",
                )),
            );
            solution.push(binding);

            assert_eq!(solution.len(), 1);
        } // Solution automatically returned to pool here

        // Test binding pooling
        {
            let mut pooled_binding = manager.acquire_binding();
            let binding = pooled_binding.get_mut();

            binding.insert(
                Variable::new("y").unwrap(),
                Term::Iri(oxirs_core::model::NamedNode::new_unchecked(
                    "http://example.org/test2",
                )),
            );

            assert_eq!(binding.len(), 1);
        } // Binding automatically returned to pool here

        // Check statistics
        let stats = manager.get_stats();
        assert!(stats.total_acquisitions() > 0);

        println!("Buffer pool stats: {}", stats.performance_summary());
    }

    #[test]
    fn test_buffer_pool_memory_management() {
        let manager = BufferPoolManager::with_capacities(10, 10, 10, 10);

        // First acquire and then release objects to populate the pools
        {
            let _solutions: Vec<_> = (0..5).map(|_| manager.acquire_solution()).collect();
            // Objects will be returned to pool when this scope ends
        }

        // Now check that objects were returned to pools
        let estimated_memory = manager.estimate_total_memory_usage();
        assert!(
            estimated_memory > 0,
            "Expected memory usage > 0, got {estimated_memory}"
        );

        // Clear pools to free memory
        manager.clear_all_pools();
        let memory_after_clear = manager.estimate_total_memory_usage();
        assert_eq!(memory_after_clear, 0);
    }
}
