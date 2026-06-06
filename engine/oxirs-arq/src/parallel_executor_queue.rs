//! Parallel scan iterator and work-stealing queue.
//!
//! [`ParallelScanIterator`] performs partitioned dataset scans, while
//! [`WorkStealingQueue<T>`] provides dynamic load-balancing across worker
//! threads for the parallel query executor.

use crate::algebra::{Term as AlgebraTerm, TriplePattern};
use crate::executor::Dataset;
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Parallel scan iterator for large datasets
pub struct ParallelScanIterator<'a> {
    dataset: &'a dyn Dataset,
    pattern: TriplePattern,
    #[allow(dead_code)]
    chunk_size: usize,
}

impl<'a> ParallelScanIterator<'a> {
    pub fn new(dataset: &'a dyn Dataset, pattern: TriplePattern, chunk_size: usize) -> Self {
        Self {
            dataset,
            pattern,
            chunk_size,
        }
    }

    /// Scan dataset in parallel chunks
    pub fn par_scan(&self) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        // In a real implementation, this would partition the dataset
        // and scan chunks in parallel
        self.dataset.find_triples(&self.pattern)
    }
}

/// Work-stealing queue for dynamic load balancing
pub struct WorkStealingQueue<T: Send + Sync> {
    queues: Vec<Arc<Mutex<VecDeque<T>>>>,
    thread_count: usize,
    work_counters: Vec<AtomicUsize>,
    global_work_count: AtomicUsize,
}

impl<T: Send + Sync> WorkStealingQueue<T> {
    pub fn new(thread_count: usize) -> Self {
        let mut queues = Vec::with_capacity(thread_count);
        let mut work_counters = Vec::with_capacity(thread_count);

        for _ in 0..thread_count {
            queues.push(Arc::new(Mutex::new(VecDeque::new())));
            work_counters.push(AtomicUsize::new(0));
        }

        Self {
            queues,
            thread_count,
            work_counters,
            global_work_count: AtomicUsize::new(0),
        }
    }

    /// Push work to a specific thread's queue
    pub fn push(&self, thread_id: usize, work: T) {
        let queue_id = thread_id % self.thread_count;
        if let Some(queue) = self.queues.get(queue_id) {
            queue.lock().push_back(work);
            self.work_counters[queue_id].fetch_add(1, Ordering::Relaxed);
            self.global_work_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Push work to the least loaded queue
    pub fn push_balanced(&self, work: T) {
        let mut min_load = usize::MAX;
        let mut best_queue = 0;

        // Find the queue with minimum load
        for (i, counter) in self.work_counters.iter().enumerate() {
            let load = counter.load(Ordering::Relaxed);
            if load < min_load {
                min_load = load;
                best_queue = i;
            }
        }

        self.push(best_queue, work);
    }

    /// Try to get work, stealing if necessary
    pub fn steal(&self, thread_id: usize) -> Option<T> {
        // Try own queue first (LIFO for cache locality)
        if let Some(queue) = self.queues.get(thread_id) {
            if let Some(mut q) = queue.try_lock() {
                if let Some(work) = q.pop_back() {
                    self.work_counters[thread_id].fetch_sub(1, Ordering::Relaxed);
                    self.global_work_count.fetch_sub(1, Ordering::Relaxed);
                    return Some(work);
                }
            }
        }

        // Try to steal from others (FIFO to avoid conflicts)
        let start = (thread_id + 1) % self.thread_count;
        for i in 0..self.thread_count {
            let target = (start + i) % self.thread_count;
            if target != thread_id {
                if let Some(queue) = self.queues.get(target) {
                    if let Some(mut q) = queue.try_lock() {
                        if let Some(work) = q.pop_front() {
                            self.work_counters[target].fetch_sub(1, Ordering::Relaxed);
                            self.global_work_count.fetch_sub(1, Ordering::Relaxed);
                            return Some(work);
                        }
                    }
                }
            }
        }

        None
    }

    /// Get total pending work count
    pub fn pending_work(&self) -> usize {
        self.global_work_count.load(Ordering::Relaxed)
    }

    /// Check if all queues are empty
    pub fn is_empty(&self) -> bool {
        self.pending_work() == 0
    }

    /// Get load distribution across threads
    pub fn get_load_distribution(&self) -> Vec<usize> {
        self.work_counters
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect()
    }

    /// Drain all work from all queues
    pub fn drain_all(&self) -> Vec<T> {
        let mut all_work = Vec::new();

        for (i, queue) in self.queues.iter().enumerate() {
            {
                let mut q = queue.lock();
                while let Some(work) = q.pop_front() {
                    all_work.push(work);
                }
            }
            self.work_counters[i].store(0, Ordering::Relaxed);
        }

        self.global_work_count.store(0, Ordering::Relaxed);
        all_work
    }
}
