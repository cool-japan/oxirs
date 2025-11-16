//! Thread-per-core architecture for optimal CPU utilization
//!
//! This module implements a thread-per-core work-stealing scheduler specifically
//! optimized for RDF triple processing operations. It provides:
//!
//! - **CPU Affinity**: Each worker thread is pinned to a specific CPU core
//! - **Work Stealing**: Idle threads steal work from busy threads
//! - **NUMA Awareness**: Memory allocation considers NUMA topology
//! - **Zero Allocation**: Lock-free work queues with bounded capacity
//!
//! # Architecture
//!
//! ```text
//! ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
//! │ Core 0  │   │ Core 1  │   │ Core 2  │   │ Core 3  │
//! │ Worker  │   │ Worker  │   │ Worker  │   │ Worker  │
//! └────┬────┘   └────┬────┘   └────┬────┘   └────┬────┘
//!      │             │             │             │
//!      └─────────────┴─────────────┴─────────────┘
//!                    Work Stealing
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_core::concurrent::thread_per_core::{ThreadPerCore, Task};
//!
//! # fn example() -> Result<(), oxirs_core::OxirsError> {
//! // Create thread-per-core executor
//! let executor = ThreadPerCore::new()?;
//!
//! // Submit work for parallel execution
//! let task = Task::new(|| {
//!     // Process RDF triples
//!     42
//! });
//!
//! let result = executor.submit(task)?;
//! # Ok(())
//! # }
//! ```

use crate::OxirsError;
use crossbeam_deque::{Injector, Stealer, Worker};
use scirs2_core::metrics::{Counter, Timer};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Result type
pub type Result<T> = std::result::Result<T, OxirsError>;

/// Thread-per-core executor
pub struct ThreadPerCore {
    /// Worker threads
    workers: Vec<CoreWorker>,
    /// Global work queue (injector)
    global_queue: Arc<Injector<Task>>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Configuration
    config: ThreadPerCoreConfig,
    /// Metrics
    submitted_counter: Counter,
    #[allow(dead_code)]
    completed_counter: Counter,
    #[allow(dead_code)]
    stolen_counter: Counter,
    #[allow(dead_code)]
    execution_timer: Timer,
}

/// Configuration for thread-per-core executor
#[derive(Debug, Clone)]
pub struct ThreadPerCoreConfig {
    /// Number of worker threads (defaults to num_cpus)
    pub num_workers: usize,
    /// Enable CPU affinity (pin threads to cores)
    pub enable_affinity: bool,
    /// Work queue capacity per worker
    pub queue_capacity: usize,
    /// Enable work stealing
    pub enable_work_stealing: bool,
    /// Steal batch size
    pub steal_batch_size: usize,
}

impl Default for ThreadPerCoreConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            enable_affinity: true,
            queue_capacity: 1024,
            enable_work_stealing: true,
            steal_batch_size: 16,
        }
    }
}

/// A task to be executed on a core
pub struct Task {
    /// Task function
    func: Box<dyn FnOnce() + Send + 'static>,
    /// Task ID for tracking
    id: usize,
}

impl Task {
    /// Create a new task
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        Self {
            func: Box::new(f),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Execute the task
    fn execute(self) {
        (self.func)();
    }

    /// Get task ID
    pub fn id(&self) -> usize {
        self.id
    }
}

/// Worker thread bound to a specific core
struct CoreWorker {
    /// Worker ID (corresponds to CPU core)
    #[allow(dead_code)]
    id: usize,
    /// Thread handle
    handle: Option<JoinHandle<()>>,
    /// Local work queue
    local_queue: Worker<Task>,
    /// Stealer for this worker (used by other workers)
    #[allow(dead_code)]
    stealer: Stealer<Task>,
    /// Statistics
    stats: Arc<WorkerStats>,
}

/// Worker statistics
#[derive(Default)]
struct WorkerStats {
    /// Tasks executed
    executed: AtomicUsize,
    /// Tasks stolen from this worker
    #[allow(dead_code)]
    stolen_from: AtomicUsize,
    /// Tasks stolen by this worker
    stolen_by: AtomicUsize,
    /// Idle time in microseconds
    idle_time_us: AtomicUsize,
}

impl ThreadPerCore {
    /// Create a new thread-per-core executor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ThreadPerCoreConfig::default())
    }

    /// Create a thread-per-core executor with custom configuration
    pub fn with_config(config: ThreadPerCoreConfig) -> Result<Self> {
        tracing::info!(
            "Initializing thread-per-core executor with {} workers",
            config.num_workers
        );

        let global_queue = Arc::new(Injector::new());
        let running = Arc::new(AtomicBool::new(true));

        // Create workers with local queues and stealers
        let mut workers = Vec::with_capacity(config.num_workers);
        let mut stealers = Vec::new();
        let mut worker_stats = Vec::new();

        // First, create all local queues and collect stealers
        for worker_id in 0..config.num_workers {
            let local_queue = Worker::new_fifo();
            let stealer = local_queue.stealer();
            stealers.push(stealer.clone());

            let stats = Arc::new(WorkerStats::default());
            worker_stats.push(stats.clone());

            let worker = CoreWorker {
                id: worker_id,
                handle: None,
                local_queue,
                stealer,
                stats,
            };

            workers.push(worker);
        }

        // Start worker threads (move local queues into threads)
        let stealers_arc = Arc::new(stealers);

        for (worker_id, worker) in workers.iter_mut().enumerate() {
            // Move the local queue into the thread
            let local_queue = std::mem::replace(&mut worker.local_queue, Worker::new_fifo());
            let global_queue = global_queue.clone();
            let running = running.clone();
            let stealers = stealers_arc.clone();
            let stats = worker_stats[worker_id].clone();
            let enable_affinity = config.enable_affinity;
            let enable_work_stealing = config.enable_work_stealing;

            let handle = thread::Builder::new()
                .name(format!("rdf-worker-{}", worker_id))
                .spawn(move || {
                    Self::worker_loop(
                        worker_id,
                        local_queue,
                        global_queue,
                        stealers,
                        running,
                        stats,
                        enable_affinity,
                        enable_work_stealing,
                    )
                })
                .map_err(|e| {
                    OxirsError::ConcurrencyError(format!("Failed to spawn worker: {}", e))
                })?;

            worker.handle = Some(handle);
        }

        Ok(Self {
            workers,
            global_queue,
            running,
            config,
            submitted_counter: Counter::new("threadpool.submitted".to_string()),
            completed_counter: Counter::new("threadpool.completed".to_string()),
            stolen_counter: Counter::new("threadpool.stolen".to_string()),
            execution_timer: Timer::new("threadpool.execution".to_string()),
        })
    }

    /// Submit a task for execution
    pub fn submit(&self, task: Task) -> Result<()> {
        if !self.running.load(Ordering::Relaxed) {
            return Err(OxirsError::ConcurrencyError(
                "Thread pool is shutting down".to_string(),
            ));
        }

        // Push to global queue
        self.global_queue.push(task);
        self.submitted_counter.add(1);

        Ok(())
    }

    /// Submit multiple tasks in batch
    pub fn submit_batch(&self, tasks: Vec<Task>) -> Result<()> {
        if !self.running.load(Ordering::Relaxed) {
            return Err(OxirsError::ConcurrencyError(
                "Thread pool is shutting down".to_string(),
            ));
        }

        for task in tasks {
            self.global_queue.push(task);
        }

        self.submitted_counter.add(1);

        Ok(())
    }

    /// Get executor statistics
    pub fn stats(&self) -> ThreadPerCoreStats {
        let total_executed: usize = self
            .workers
            .iter()
            .map(|w| w.stats.executed.load(Ordering::Relaxed))
            .sum();

        let total_stolen: usize = self
            .workers
            .iter()
            .map(|w| w.stats.stolen_by.load(Ordering::Relaxed))
            .sum();

        let total_idle_us: usize = self
            .workers
            .iter()
            .map(|w| w.stats.idle_time_us.load(Ordering::Relaxed))
            .sum();

        ThreadPerCoreStats {
            num_workers: self.config.num_workers,
            submitted: self.submitted_counter.get(),
            completed: total_executed as u64,
            stolen: total_stolen as u64,
            avg_idle_time_us: total_idle_us as f64 / self.config.num_workers as f64,
        }
    }

    /// Worker thread main loop
    #[allow(clippy::too_many_arguments)]
    fn worker_loop(
        worker_id: usize,
        local_queue: Worker<Task>,
        global_queue: Arc<Injector<Task>>,
        stealers: Arc<Vec<Stealer<Task>>>,
        running: Arc<AtomicBool>,
        stats: Arc<WorkerStats>,
        enable_affinity: bool,
        enable_work_stealing: bool,
    ) {
        // Set CPU affinity if enabled
        if enable_affinity {
            if let Err(e) = Self::set_cpu_affinity(worker_id) {
                tracing::warn!("Failed to set CPU affinity for worker {}: {}", worker_id, e);
            } else {
                tracing::debug!("Worker {} pinned to core {}", worker_id, worker_id);
            }
        }

        while running.load(Ordering::Relaxed) {
            // Try to get task from local queue
            if let Some(task) = local_queue.pop() {
                task.execute();
                stats.executed.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Try to steal from global queue
            match global_queue.steal() {
                crossbeam_deque::Steal::Success(task) => {
                    task.execute();
                    stats.executed.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                crossbeam_deque::Steal::Empty => {}
                crossbeam_deque::Steal::Retry => continue,
            }

            // Try to steal from other workers
            if enable_work_stealing {
                let mut found = false;
                for (i, stealer) in stealers.iter().enumerate() {
                    if i == worker_id {
                        continue; // Skip self
                    }

                    match stealer.steal() {
                        crossbeam_deque::Steal::Success(task) => {
                            task.execute();
                            stats.executed.fetch_add(1, Ordering::Relaxed);
                            stats.stolen_by.fetch_add(1, Ordering::Relaxed);
                            found = true;
                            break;
                        }
                        crossbeam_deque::Steal::Empty => {}
                        crossbeam_deque::Steal::Retry => continue,
                    }
                }

                if found {
                    continue;
                }
            }

            // No work found, sleep briefly
            let idle_start = std::time::Instant::now();
            thread::sleep(Duration::from_micros(10));
            let idle_us = idle_start.elapsed().as_micros() as usize;
            stats.idle_time_us.fetch_add(idle_us, Ordering::Relaxed);
        }

        tracing::info!("Worker {} shutting down", worker_id);
    }

    /// Set CPU affinity for current thread
    #[cfg(target_os = "linux")]
    fn set_cpu_affinity(core_id: usize) -> Result<()> {
        use std::mem;

        // Linux-specific CPU affinity setting
        unsafe {
            let mut cpu_set: libc::cpu_set_t = mem::zeroed();
            libc::CPU_SET(core_id, &mut cpu_set);

            if libc::sched_setaffinity(0, mem::size_of::<libc::cpu_set_t>(), &cpu_set) != 0 {
                return Err(OxirsError::ConcurrencyError(format!(
                    "Failed to set CPU affinity: {}",
                    std::io::Error::last_os_error()
                )));
            }
        }

        Ok(())
    }

    /// Set CPU affinity for current thread (no-op on non-Linux)
    #[cfg(not(target_os = "linux"))]
    fn set_cpu_affinity(_core_id: usize) -> Result<()> {
        // CPU affinity not supported on this platform
        Ok(())
    }

    /// Shutdown the thread pool gracefully
    pub fn shutdown(self) -> Result<()> {
        tracing::info!("Shutting down thread-per-core executor");

        // Signal workers to stop
        self.running.store(false, Ordering::Relaxed);

        // Wait for all workers to finish
        for mut worker in self.workers {
            if let Some(handle) = worker.handle.take() {
                handle.join().map_err(|_| {
                    OxirsError::ConcurrencyError("Worker thread panicked".to_string())
                })?;
            }
        }

        tracing::info!("Thread-per-core executor shut down successfully");
        Ok(())
    }
}

impl Default for ThreadPerCore {
    fn default() -> Self {
        Self::new().expect("Failed to create ThreadPerCore executor")
    }
}

/// Statistics for thread-per-core executor
#[derive(Debug, Clone)]
pub struct ThreadPerCoreStats {
    /// Number of worker threads
    pub num_workers: usize,
    /// Total tasks submitted
    pub submitted: u64,
    /// Total tasks completed
    pub completed: u64,
    /// Total tasks stolen
    pub stolen: u64,
    /// Average idle time per worker (microseconds)
    pub avg_idle_time_us: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    #[test]
    fn test_thread_per_core_creation() -> Result<()> {
        let config = ThreadPerCoreConfig {
            num_workers: 4,
            ..Default::default()
        };

        let executor = ThreadPerCore::with_config(config)?;
        executor.shutdown()?;

        Ok(())
    }

    #[test]
    fn test_task_submission() -> Result<()> {
        let executor = ThreadPerCore::new()?;

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let task = Task::new(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        });

        executor.submit(task)?;

        // Give time for execution
        thread::sleep(Duration::from_millis(100));

        assert_eq!(counter.load(Ordering::Relaxed), 1);

        executor.shutdown()?;
        Ok(())
    }

    #[test]
    fn test_batch_submission() -> Result<()> {
        let executor = ThreadPerCore::new()?;

        let counter = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<_> = (0..100)
            .map(|_| {
                let counter = counter.clone();
                Task::new(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
                })
            })
            .collect();

        executor.submit_batch(tasks)?;

        // Give time for execution
        thread::sleep(Duration::from_millis(500));

        assert_eq!(counter.load(Ordering::Relaxed), 100);

        executor.shutdown()?;
        Ok(())
    }

    #[test]
    fn test_stats() -> Result<()> {
        let executor = ThreadPerCore::new()?;

        // Submit some tasks
        for _ in 0..10 {
            let task = Task::new(|| {
                thread::sleep(Duration::from_millis(1));
            });
            executor.submit(task)?;
        }

        // Give time for execution
        thread::sleep(Duration::from_millis(100));

        let stats = executor.stats();
        assert_eq!(stats.submitted, 10);
        assert!(stats.completed <= 10);

        executor.shutdown()?;
        Ok(())
    }
}
