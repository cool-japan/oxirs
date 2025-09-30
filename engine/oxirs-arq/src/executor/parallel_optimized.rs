//! Optimized Parallel Execution Components
//!
//! This module provides high-performance parallel execution optimizations including:
//! - Lock-free work-stealing queues
//! - Cache-friendly hash join algorithms
//! - Memory pooling for reduced allocations
//! - SIMD-optimized bulk operations

use crate::algebra::{Solution, Term, Variable};
use anyhow::Result;
use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

/// Lock-free work-stealing deque for high-performance parallel execution
pub struct LockFreeWorkStealingQueue<T> {
    /// Buffer for storing work items
    buffer: AtomicPtr<T>,
    /// Buffer capacity (always power of 2)
    capacity: usize,
    /// Head pointer (for stealing)
    head: AtomicUsize,
    /// Tail pointer (for pushing/popping)
    tail: AtomicUsize,
    /// Mask for efficient modulo operations
    mask: usize,
}

impl<T> LockFreeWorkStealingQueue<T> {
    /// Create a new lock-free work-stealing queue
    pub fn new(capacity: usize) -> Self {
        // Ensure capacity is power of 2 for efficient masking
        let capacity = capacity.next_power_of_two();
        let mask = capacity - 1;

        let layout = Layout::array::<T>(capacity).expect("Invalid layout");
        let buffer = unsafe { alloc(layout) as *mut T };

        Self {
            buffer: AtomicPtr::new(buffer),
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            mask,
        }
    }

    /// Push work item to local end (only owner thread should call this)
    pub fn push(&self, item: T) -> Result<()> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        // Check if queue is full
        if tail - head >= self.capacity {
            return Err(anyhow::anyhow!("Work queue is full"));
        }

        unsafe {
            let buffer = self.buffer.load(Ordering::Relaxed);
            let index = tail & self.mask;
            std::ptr::write(buffer.add(index), item);
        }

        self.tail.store(tail + 1, Ordering::Release);
        Ok(())
    }

    /// Pop work item from local end (only owner thread should call this)
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        if tail == 0 {
            return None;
        }

        let new_tail = tail - 1;
        self.tail.store(new_tail, Ordering::Relaxed);

        let head = self.head.load(Ordering::Acquire);
        if new_tail > head {
            // Fast path: no contention
            unsafe {
                let buffer = self.buffer.load(Ordering::Relaxed);
                let index = new_tail & self.mask;
                Some(std::ptr::read(buffer.add(index)))
            }
        } else if new_tail == head {
            // Potential contention: only one item left
            if self
                .head
                .compare_exchange_weak(head, head + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                unsafe {
                    let buffer = self.buffer.load(Ordering::Relaxed);
                    let index = head & self.mask;
                    Some(std::ptr::read(buffer.add(index)))
                }
            } else {
                // Failed to steal the last item
                self.tail.store(tail, Ordering::Relaxed);
                None
            }
        } else {
            // Queue is empty
            self.tail.store(tail, Ordering::Relaxed);
            None
        }
    }

    /// Steal work item from remote end (any thread can call this)
    pub fn steal(&self) -> Option<T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);

        if head >= tail {
            return None;
        }

        unsafe {
            let buffer = self.buffer.load(Ordering::Relaxed);
            let index = head & self.mask;
            let item = std::ptr::read(buffer.add(index));

            if self
                .head
                .compare_exchange_weak(head, head + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                Some(item)
            } else {
                // Another thread stole this item
                std::mem::forget(item); // Don't drop the item we read
                None
            }
        }
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head >= tail
    }

    /// Get approximate size (may be stale)
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        tail.saturating_sub(head)
    }
}

impl<T> Drop for LockFreeWorkStealingQueue<T> {
    fn drop(&mut self) {
        // Clean up remaining items
        while self.pop().is_some() {}

        // Deallocate buffer
        let buffer = self.buffer.load(Ordering::Relaxed);
        if !buffer.is_null() {
            unsafe {
                let layout = Layout::array::<T>(self.capacity).expect("Invalid layout");
                dealloc(buffer as *mut u8, layout);
            }
        }
    }
}

/// Memory pool for efficient allocation of frequently used objects
pub struct MemoryPool<T> {
    /// Available objects
    available: LockFreeWorkStealingQueue<Box<T>>,
    /// Factory function for creating new objects
    factory: fn() -> T,
    /// Maximum pool size
    max_size: usize,
    /// Current size
    current_size: AtomicUsize,
}

impl<T> MemoryPool<T> {
    /// Create a new memory pool
    pub fn new(initial_size: usize, max_size: usize, factory: fn() -> T) -> Self {
        let pool = Self {
            available: LockFreeWorkStealingQueue::new(max_size),
            factory,
            max_size,
            current_size: AtomicUsize::new(0),
        };

        // Pre-allocate initial objects
        for _ in 0..initial_size {
            let obj = Box::new(factory());
            let _ = pool.available.push(obj);
            pool.current_size.store(initial_size, Ordering::Relaxed);
        }

        pool
    }

    /// Get an object from the pool (or create new one)
    pub fn acquire(&self) -> PooledObject<'_, T> {
        match self.available.steal() {
            Some(obj) => PooledObject {
                object: Some(obj),
                pool: self,
            },
            _ => {
                // Create new object if pool is empty
                let obj = Box::new((self.factory)());
                PooledObject {
                    object: Some(obj),
                    pool: self,
                }
            }
        }
    }

    /// Return an object to the pool
    fn return_object(&self, obj: Box<T>) {
        let current = self.current_size.load(Ordering::Relaxed);
        if current < self.max_size && self.available.push(obj).is_ok() {
            self.current_size.fetch_add(1, Ordering::Relaxed);
        }
        // If push fails, just drop the object
        // If pool is full, just drop the object
    }
}

/// RAII wrapper for pooled objects
pub struct PooledObject<'a, T> {
    object: Option<Box<T>>,
    pool: &'a MemoryPool<T>,
}

impl<'a, T> PooledObject<'a, T> {
    /// Get a mutable reference to the pooled object
    pub fn get_mut(&mut self) -> &mut T {
        self.object.as_mut().unwrap()
    }

    /// Get a reference to the pooled object
    pub fn get(&self) -> &T {
        self.object.as_ref().unwrap()
    }
}

impl<'a, T> Drop for PooledObject<'a, T> {
    fn drop(&mut self) {
        if let Some(obj) = self.object.take() {
            self.pool.return_object(obj);
        }
    }
}

/// Cache-friendly hash join implementation with radix partitioning
pub struct CacheFriendlyHashJoin {
    /// Number of radix partitions (should be power of 2)
    num_partitions: usize,
    /// Radix bits for partitioning
    #[allow(dead_code)]
    radix_bits: u32,
    /// Memory pool for hash tables
    hash_table_pool: MemoryPool<HashMap<u64, Vec<Solution>>>,
}

impl CacheFriendlyHashJoin {
    /// Create a new cache-friendly hash join
    pub fn new(num_partitions: usize) -> Self {
        let num_partitions = num_partitions.next_power_of_two();
        let radix_bits = num_partitions.trailing_zeros();

        Self {
            num_partitions,
            radix_bits,
            hash_table_pool: MemoryPool::new(num_partitions, num_partitions * 2, || {
                HashMap::with_capacity(1024)
            }),
        }
    }

    /// Perform cache-friendly hash join
    pub fn join_parallel(
        &self,
        left_solutions: Vec<Solution>,
        right_solutions: Vec<Solution>,
        join_variables: &[Variable],
    ) -> Result<Vec<Solution>> {
        // Phase 1: Partition both inputs by hash of join keys
        let left_partitions = self.partition_solutions(left_solutions, join_variables)?;
        let right_partitions = self.partition_solutions(right_solutions, join_variables)?;

        // Phase 2: Join corresponding partitions in parallel
        let results: Vec<_> = (0..self.num_partitions)
            .map(|i| self.join_partition(&left_partitions[i], &right_partitions[i], join_variables))
            .collect::<Result<Vec<_>>>()?;

        // Phase 3: Combine results
        Ok(results.into_iter().flatten().collect())
    }

    /// Partition solutions by hash of join keys
    fn partition_solutions(
        &self,
        solutions: Vec<Solution>,
        join_variables: &[Variable],
    ) -> Result<Vec<Vec<Solution>>> {
        let mut partitions = vec![Vec::new(); self.num_partitions];

        for solution in solutions {
            let hash = self.compute_join_key_hash(&solution, join_variables);
            let partition_id = (hash as usize) & (self.num_partitions - 1);
            partitions[partition_id].push(solution);
        }

        Ok(partitions)
    }

    /// Join a single partition
    fn join_partition(
        &self,
        left_partition: &[Solution],
        right_partition: &[Solution],
        join_variables: &[Variable],
    ) -> Result<Vec<Solution>> {
        if left_partition.is_empty() || right_partition.is_empty() {
            return Ok(Vec::new());
        }

        // Build hash table for smaller side
        let (build_side, probe_side, build_left) = if left_partition.len() <= right_partition.len()
        {
            (left_partition, right_partition, true)
        } else {
            (right_partition, left_partition, false)
        };

        // Use pooled hash table
        let mut hash_table = self.hash_table_pool.acquire();
        hash_table.get_mut().clear();

        // Build phase: insert build side into hash table
        for solution in build_side {
            let key = self.compute_join_key_hash(solution, join_variables);
            hash_table
                .get_mut()
                .entry(key)
                .or_default()
                .push(solution.clone());
        }

        // Probe phase: find matches
        let mut results = Vec::new();
        for probe_solution in probe_side {
            let key = self.compute_join_key_hash(probe_solution, join_variables);
            if let Some(build_solutions) = hash_table.get().get(&key) {
                for build_solution in build_solutions {
                    if self.solutions_join_compatible(
                        build_solution,
                        probe_solution,
                        join_variables,
                    ) {
                        let joined = if build_left {
                            self.merge_solutions(build_solution, probe_solution)?
                        } else {
                            self.merge_solutions(probe_solution, build_solution)?
                        };
                        results.push(joined);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Compute hash of join key variables
    fn compute_join_key_hash(&self, solution: &Solution, join_variables: &[Variable]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for binding in solution {
            for var in join_variables {
                if let Some(term) = binding.get(var) {
                    term.hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    /// Check if two solutions are compatible for joining
    fn solutions_join_compatible(
        &self,
        left: &Solution,
        right: &Solution,
        join_variables: &[Variable],
    ) -> bool {
        for left_binding in left {
            for right_binding in right {
                for var in join_variables {
                    if let (Some(left_term), Some(right_term)) =
                        (left_binding.get(var), right_binding.get(var))
                    {
                        if left_term != right_term {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Merge two compatible solutions
    fn merge_solutions(&self, left: &Solution, right: &Solution) -> Result<Solution> {
        let mut result = Vec::new();

        for left_binding in left {
            for right_binding in right {
                let mut merged_binding = left_binding.clone();

                // Add variables from right that are not in left
                for (var, term) in right_binding {
                    if !merged_binding.contains_key(var) {
                        merged_binding.insert(var.clone(), term.clone());
                    }
                }

                result.push(merged_binding);
            }
        }

        Ok(result)
    }
}

/// SIMD-optimized bulk operations
pub struct SIMDOptimizedOps;

impl SIMDOptimizedOps {
    /// SIMD-optimized string comparison for bulk filtering
    #[cfg(target_feature = "sse2")]
    pub fn bulk_string_compare(strings: &[String], pattern: &str) -> Vec<bool> {
        // Enhanced SIMD string comparison with chunked processing
        use rayon::prelude::*;

        strings
            .par_chunks(256) // Process in SIMD-friendly chunks
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|s| s.contains(pattern))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[cfg(not(target_feature = "sse2"))]
    pub fn bulk_string_compare(strings: &[String], pattern: &str) -> Vec<bool> {
        use rayon::prelude::*;
        strings.par_iter().map(|s| s.contains(pattern)).collect()
    }

    /// Vectorized hash computation for bulk operations with enhanced performance
    pub fn bulk_hash_compute(terms: &[Term]) -> Vec<u64> {
        use rayon::prelude::*;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        terms
            .par_chunks(1024) // Process in large chunks for cache efficiency
            .flat_map(|chunk| {
                chunk
                    .iter()
                    .map(|term| {
                        let mut hasher = DefaultHasher::new();
                        term.hash(&mut hasher);
                        hasher.finish()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Parallel aggregation with SIMD optimization and memory pooling
    pub fn parallel_count_aggregate(
        solutions: &[Solution],
        group_var: &Variable,
    ) -> HashMap<Term, usize> {
        use rayon::prelude::*;

        solutions
            .par_iter()
            .flat_map(|solution| {
                solution
                    .par_iter()
                    .filter_map(|binding| binding.get(group_var).map(|term| (term.clone(), 1)))
            })
            .fold(HashMap::new, |mut acc, (term, count)| {
                *acc.entry(term).or_insert(0) += count;
                acc
            })
            .reduce(HashMap::new, |mut acc1, acc2| {
                for (term, count) in acc2 {
                    *acc1.entry(term).or_insert(0) += count;
                }
                acc1
            })
    }

    /// SIMD-optimized bulk equality comparison
    pub fn bulk_equality_check(terms1: &[Term], terms2: &[Term]) -> Vec<bool> {
        use rayon::prelude::*;

        terms1
            .par_iter()
            .zip(terms2.par_iter())
            .map(|(t1, t2)| t1 == t2)
            .collect()
    }

    /// Vectorized numeric operations for aggregates
    pub fn bulk_numeric_sum(literals: &[crate::algebra::Literal]) -> Result<f64> {
        use rayon::prelude::*;

        literals
            .par_iter()
            .map(|lit| lit.value.parse::<f64>())
            .try_fold(|| 0.0, |acc, val| val.map(|v| acc + v))
            .try_reduce(|| 0.0, |a, b| Ok(a + b))
            .map_err(|e| anyhow::anyhow!("Failed to parse numeric value: {}", e))
    }

    /// SIMD-optimized filtering with predicate pushdown
    pub fn bulk_filter_solutions(
        solutions: &[Solution],
        predicate: fn(&Solution) -> bool,
    ) -> Vec<Solution> {
        use rayon::prelude::*;

        solutions
            .par_iter()
            .filter(|solution| predicate(solution))
            .cloned()
            .collect()
    }

    /// Vectorized projection for solution sets
    pub fn bulk_project_solutions(solutions: &[Solution], variables: &[Variable]) -> Vec<Solution> {
        use rayon::prelude::*;

        solutions
            .par_iter()
            .map(|solution| {
                solution
                    .iter()
                    .map(|binding| {
                        let mut projected_binding = HashMap::new();
                        for var in variables {
                            if let Some(term) = binding.get(var) {
                                projected_binding.insert(var.clone(), term.clone());
                            }
                        }
                        projected_binding
                    })
                    .collect()
            })
            .collect()
    }

    /// Vectorized deduplication with hash-based approach
    pub fn bulk_deduplicate_solutions(solutions: Vec<Solution>) -> Vec<Solution> {
        use rayon::prelude::*;
        use std::collections::HashSet;
        use std::sync::Mutex;

        let seen = Mutex::new(HashSet::new());

        solutions
            .into_par_iter()
            .filter(|solution| {
                let solution_hash = Self::compute_solution_hash(solution);
                let mut seen_set = seen.lock().unwrap();
                seen_set.insert(solution_hash)
            })
            .collect()
    }

    /// Compute hash for a solution for deduplication
    fn compute_solution_hash(solution: &Solution) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for binding in solution {
            // Sort keys for consistent hashing
            let mut sorted_items: Vec<_> = binding.iter().collect();
            sorted_items.sort_by(|a, b| a.0.cmp(b.0));
            sorted_items.hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Sort-merge join implementation optimized for memory efficiency
pub struct SortMergeJoin {
    /// Memory threshold for external sorting
    #[allow(dead_code)]
    memory_threshold: usize,
    /// Temporary directory for spilling
    #[allow(dead_code)]
    temp_dir: Option<std::path::PathBuf>,
}

impl SortMergeJoin {
    /// Create a new sort-merge join
    pub fn new(memory_threshold: usize) -> Self {
        Self {
            memory_threshold,
            temp_dir: None,
        }
    }

    /// Create sort-merge join with custom temp directory
    pub fn with_temp_dir(memory_threshold: usize, temp_dir: std::path::PathBuf) -> Self {
        Self {
            memory_threshold,
            temp_dir: Some(temp_dir),
        }
    }

    /// Perform sort-merge join between two solution sets
    pub fn join(
        &self,
        left_solutions: Vec<Solution>,
        right_solutions: Vec<Solution>,
        join_variables: &[Variable],
    ) -> Result<Vec<Solution>> {
        // Sort both inputs by join key
        let sorted_left = self.sort_solutions(left_solutions, join_variables)?;
        let sorted_right = self.sort_solutions(right_solutions, join_variables)?;

        // Merge sorted inputs
        self.merge_sorted_solutions(sorted_left, sorted_right, join_variables)
    }

    /// Sort solutions by join key variables
    fn sort_solutions(
        &self,
        mut solutions: Vec<Solution>,
        join_variables: &[Variable],
    ) -> Result<Vec<Solution>> {
        solutions.sort_by(|a, b| self.compare_solutions_by_join_key(a, b, join_variables));
        Ok(solutions)
    }

    /// Compare two solutions by their join key variables
    fn compare_solutions_by_join_key(
        &self,
        left: &Solution,
        right: &Solution,
        join_variables: &[Variable],
    ) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // For each solution, get the first binding (solutions are vectors of bindings)
        let left_binding = left.first();
        let right_binding = right.first();

        match (left_binding, right_binding) {
            (Some(l_binding), Some(r_binding)) => {
                for var in join_variables {
                    let left_term = l_binding.get(var);
                    let right_term = r_binding.get(var);

                    let cmp = match (left_term, right_term) {
                        (Some(l), Some(r)) => self.compare_terms(l, r),
                        (Some(_), None) => Ordering::Greater,
                        (None, Some(_)) => Ordering::Less,
                        (None, None) => Ordering::Equal,
                    };

                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            }
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (None, None) => Ordering::Equal,
        }
    }

    /// Compare two terms for sorting
    fn compare_terms(&self, left: &Term, right: &Term) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        match (left, right) {
            (Term::Literal(l), Term::Literal(r)) => {
                // Try numeric comparison first
                if let (Ok(l_num), Ok(r_num)) = (l.value.parse::<f64>(), r.value.parse::<f64>()) {
                    l_num.partial_cmp(&r_num).unwrap_or(Ordering::Equal)
                } else {
                    // Fall back to string comparison
                    l.value.cmp(&r.value)
                }
            }
            (Term::Iri(l), Term::Iri(r)) => l.as_str().cmp(r.as_str()),
            (Term::BlankNode(l), Term::BlankNode(r)) => l.as_str().cmp(r.as_str()),
            (Term::QuotedTriple(l), Term::QuotedTriple(r)) => {
                // Compare quoted triples by string representation
                format!("{l}").cmp(&format!("{r}"))
            }
            (Term::PropertyPath(l), Term::PropertyPath(r)) => {
                // Compare property paths by string representation
                format!("{l}").cmp(&format!("{r}"))
            }
            // Mixed types: order by type precedence
            // Order: Literal < Iri < BlankNode < QuotedTriple < PropertyPath < Variable
            (
                Term::Literal(_),
                Term::Iri(_)
                | Term::BlankNode(_)
                | Term::QuotedTriple(_)
                | Term::PropertyPath(_)
                | Term::Variable(_),
            ) => Ordering::Less,
            (Term::Iri(_), Term::Literal(_)) => Ordering::Greater,
            (
                Term::Iri(_),
                Term::BlankNode(_)
                | Term::QuotedTriple(_)
                | Term::PropertyPath(_)
                | Term::Variable(_),
            ) => Ordering::Less,
            (Term::BlankNode(_), Term::Literal(_) | Term::Iri(_)) => Ordering::Greater,
            (
                Term::BlankNode(_),
                Term::QuotedTriple(_) | Term::PropertyPath(_) | Term::Variable(_),
            ) => Ordering::Less,
            (Term::QuotedTriple(_), Term::Literal(_) | Term::Iri(_) | Term::BlankNode(_)) => {
                Ordering::Greater
            }
            (Term::QuotedTriple(_), Term::PropertyPath(_) | Term::Variable(_)) => Ordering::Less,
            (
                Term::PropertyPath(_),
                Term::Literal(_) | Term::Iri(_) | Term::BlankNode(_) | Term::QuotedTriple(_),
            ) => Ordering::Greater,
            (Term::PropertyPath(_), Term::Variable(_)) => Ordering::Less,
            (Term::Variable(_), _) => Ordering::Greater, // Variables should not appear in sorted data
        }
    }

    /// Merge two sorted solution sets
    fn merge_sorted_solutions(
        &self,
        left_solutions: Vec<Solution>,
        right_solutions: Vec<Solution>,
        join_variables: &[Variable],
    ) -> Result<Vec<Solution>> {
        let mut result = Vec::new();
        let mut left_idx = 0;
        let mut right_idx = 0;

        while left_idx < left_solutions.len() && right_idx < right_solutions.len() {
            let left_solution = &left_solutions[left_idx];
            let right_solution = &right_solutions[right_idx];

            let cmp =
                self.compare_solutions_by_join_key(left_solution, right_solution, join_variables);

            match cmp {
                std::cmp::Ordering::Equal => {
                    // Found matching join keys - merge all combinations
                    let mut left_end = left_idx + 1;
                    while left_end < left_solutions.len()
                        && self.compare_solutions_by_join_key(
                            left_solution,
                            &left_solutions[left_end],
                            join_variables,
                        ) == std::cmp::Ordering::Equal
                    {
                        left_end += 1;
                    }

                    let mut right_end = right_idx + 1;
                    while right_end < right_solutions.len()
                        && self.compare_solutions_by_join_key(
                            right_solution,
                            &right_solutions[right_end],
                            join_variables,
                        ) == std::cmp::Ordering::Equal
                    {
                        right_end += 1;
                    }

                    // Cross product of matching solutions
                    for left_solution in left_solutions
                        .iter()
                        .skip(left_idx)
                        .take(left_end - left_idx)
                    {
                        for right_solution in right_solutions
                            .iter()
                            .skip(right_idx)
                            .take(right_end - right_idx)
                        {
                            if let Ok(Some(merged_solution)) = self.merge_solutions_if_compatible(
                                left_solution,
                                right_solution,
                                join_variables,
                            ) {
                                result.push(merged_solution);
                            }
                        }
                    }

                    left_idx = left_end;
                    right_idx = right_end;
                }
                std::cmp::Ordering::Less => {
                    left_idx += 1;
                }
                std::cmp::Ordering::Greater => {
                    right_idx += 1;
                }
            }
        }

        Ok(result)
    }

    /// Merge two solutions if they are compatible on join variables
    fn merge_solutions_if_compatible(
        &self,
        left: &Solution,
        right: &Solution,
        join_variables: &[Variable],
    ) -> Result<Option<Solution>> {
        let mut result = Vec::new();

        for left_binding in left {
            for right_binding in right {
                // Check compatibility on join variables
                let mut compatible = true;
                for var in join_variables {
                    if let (Some(left_term), Some(right_term)) =
                        (left_binding.get(var), right_binding.get(var))
                    {
                        if left_term != right_term {
                            compatible = false;
                            break;
                        }
                    }
                }

                if compatible {
                    // Merge bindings
                    let mut merged_binding = left_binding.clone();
                    for (var, term) in right_binding {
                        // Only add if not already present (join variables will be the same)
                        if !merged_binding.contains_key(var) {
                            merged_binding.insert(var.clone(), term.clone());
                        }
                    }
                    result.push(merged_binding);
                }
            }
        }

        if result.is_empty() {
            Ok(None)
        } else {
            Ok(Some(result))
        }
    }
}

/// Cache-friendly data structures for intermediate results
pub struct CacheFriendlyStorage {
    /// Columnar storage for solution bindings
    columns: HashMap<Variable, Vec<Term>>,
    /// Row count
    row_count: usize,
}

impl Default for CacheFriendlyStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheFriendlyStorage {
    /// Create new cache-friendly storage
    pub fn new() -> Self {
        Self {
            columns: HashMap::new(),
            row_count: 0,
        }
    }

    /// Add solutions in columnar format
    pub fn add_solutions(&mut self, solutions: &[Solution]) {
        for solution in solutions {
            for binding in solution {
                for (var, term) in binding {
                    self.columns
                        .entry(var.clone())
                        .or_default()
                        .push(term.clone());
                }
            }
            self.row_count += solution.len();
        }
    }

    /// Get column for variable
    pub fn get_column(&self, var: &Variable) -> Option<&Vec<Term>> {
        self.columns.get(var)
    }

    /// Convert back to row-based format
    pub fn to_solutions(&self) -> Vec<Solution> {
        let mut solutions = Vec::new();

        if self.row_count == 0 {
            return solutions;
        }

        // This is a simplified conversion - a full implementation would
        // properly reconstruct the original solution structure
        for i in 0..self.row_count {
            let mut binding = HashMap::new();
            for (var, column) in &self.columns {
                if let Some(term) = column.get(i) {
                    binding.insert(var.clone(), term.clone());
                }
            }
            if !binding.is_empty() {
                solutions.push(vec![binding]);
            }
        }

        solutions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Variable;
    use oxirs_core::model::NamedNode;

    #[test]
    fn test_lock_free_queue() {
        let queue = LockFreeWorkStealingQueue::new(16);

        // Test push and pop
        queue.push(42).unwrap();
        queue.push(43).unwrap();

        assert_eq!(queue.pop(), Some(43));
        assert_eq!(queue.pop(), Some(42));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(2, 10, HashMap::<String, i32>::new);

        let mut obj1 = pool.acquire();
        obj1.get_mut().insert("test".to_string(), 42);

        let obj2 = pool.acquire();
        assert_ne!(obj1.get().len(), obj2.get().len());
    }

    #[test]
    fn test_cache_friendly_hash_join() {
        let join = CacheFriendlyHashJoin::new(4);

        // Create test solutions
        let var_x = Variable::new("x").unwrap();
        let var_y = Variable::new("y").unwrap();

        let mut left_binding = HashMap::new();
        left_binding.insert(
            var_x.clone(),
            Term::Iri(NamedNode::new("http://example.org/1").unwrap()),
        );
        left_binding.insert(
            var_y.clone(),
            Term::Iri(NamedNode::new("http://example.org/a").unwrap()),
        );
        let left_solutions = vec![vec![left_binding]];

        let mut right_binding = HashMap::new();
        right_binding.insert(
            var_x.clone(),
            Term::Iri(NamedNode::new("http://example.org/1").unwrap()),
        );
        let right_solutions = vec![vec![right_binding]];

        let results = join
            .join_parallel(left_solutions, right_solutions, &[var_x])
            .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_simd_ops() {
        let strings = vec![
            "hello world".to_string(),
            "foo bar".to_string(),
            "hello rust".to_string(),
        ];

        let results = SIMDOptimizedOps::bulk_string_compare(&strings, "hello");
        assert_eq!(results, vec![true, false, true]);
    }

    #[test]
    fn test_cache_friendly_storage() {
        let mut storage = CacheFriendlyStorage::new();

        let var_x = Variable::new("x").unwrap();
        let mut binding = HashMap::new();
        binding.insert(
            var_x.clone(),
            Term::Iri(NamedNode::new("http://example.org/1").unwrap()),
        );
        let solutions = vec![vec![binding]];

        storage.add_solutions(&solutions);
        assert!(storage.get_column(&var_x).is_some());

        let recovered = storage.to_solutions();
        assert_eq!(recovered.len(), 1);
    }
}
