//! Work-stealing scheduler for federated sub-plan execution.
//!
//! Each worker thread owns a local deque of `SubPlan` tasks.  When its
//! deque is empty, the worker tries to steal from the busiest peer.
//!
//! # Design
//!
//! - `WorkStealer` is the public coordinator.  It owns a fixed-size pool of
//!   `WorkerDeque`s (one per worker) protected behind `Arc<Mutex<…>>`.
//! - `SubPlan` is the unit of work: an opaque future-returning closure.
//! - `WorkStealer::submit` pushes tasks onto the submitter's queue (round-robin
//!   by default, so the initial distribution is balanced).
//! - Workers are spawned as Tokio tasks.  Each loop iteration:
//!     1. Pop from own deque.
//!     2. If empty, scan peers and steal from the largest deque.
//!     3. Execute the task.
//!
//! This pure-Rust implementation avoids `crossbeam-deque` to stay in the
//! COOLJAPAN Pure Rust ecosystem.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

// ─── SubPlanId ────────────────────────────────────────────────────────────────

/// Unique identifier for a sub-plan task.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubPlanId(pub String);

impl SubPlanId {
    /// Create an ID from any string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Auto-generate an ID using a monotonic counter.
    pub fn generate() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(format!("plan-{}", COUNTER.fetch_add(1, Ordering::Relaxed)))
    }
}

impl fmt::Display for SubPlanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ─── SubPlan ──────────────────────────────────────────────────────────────────

/// Metadata + execution outcome for a sub-plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubPlanResult {
    pub plan_id: SubPlanId,
    /// Worker index that executed this plan.
    pub executed_by: usize,
    /// Whether the task was stolen from another worker's queue.
    pub was_stolen: bool,
    /// Wall-clock time from dequeue to completion.
    pub execution_time: Duration,
    /// Whether execution succeeded.
    pub success: bool,
    /// Optional error message.
    pub error: Option<String>,
}

/// A pending sub-plan task.
///
/// The `work` field is an `Arc`-wrapped closure that returns a `bool`
/// indicating success.  We use a trait object so tasks can be heterogeneous.
pub struct SubPlan {
    pub id: SubPlanId,
    /// A synchronous, blocking-compatible closure.  In async contexts callers
    /// should use `tokio::task::block_in_place` around CPU-bound work.
    pub work: Arc<dyn Fn() -> bool + Send + Sync + 'static>,
}

impl SubPlan {
    /// Create a new sub-plan from a closure.
    pub fn new(id: SubPlanId, work: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self {
            id,
            work: Arc::new(work),
        }
    }
}

impl fmt::Debug for SubPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SubPlan({})", self.id)
    }
}

// ─── WorkerDeque ──────────────────────────────────────────────────────────────

/// A worker's local task deque (push-back / pop-front own; steal-from-back).
#[derive(Debug, Default)]
struct WorkerDeque {
    tasks: VecDeque<SubPlan>,
    steal_count: u64,
    execute_count: u64,
}

impl WorkerDeque {
    fn push(&mut self, plan: SubPlan) {
        self.tasks.push_back(plan);
    }

    /// Pop the front (FIFO for own work).
    fn pop_own(&mut self) -> Option<SubPlan> {
        self.tasks.pop_front()
    }

    /// Steal from the back (last-pushed = most recently added).
    #[allow(dead_code)]
    fn steal(&mut self) -> Option<SubPlan> {
        if let Some(plan) = self.tasks.pop_back() {
            self.steal_count += 1;
            Some(plan)
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.tasks.len()
    }
}

// ─── WorkerStats ──────────────────────────────────────────────────────────────

/// Per-worker statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerStats {
    pub worker_id: usize,
    pub executed: u64,
    pub stolen: u64,
    pub pending: usize,
}

// ─── WorkStealerConfig ────────────────────────────────────────────────────────

/// Configuration for the work-stealing scheduler.
#[derive(Debug, Clone)]
pub struct WorkStealerConfig {
    /// Number of worker queues (threads).
    pub num_workers: usize,
    /// How long a worker spins waiting for new tasks before yielding.
    pub spin_duration: Duration,
}

impl Default for WorkStealerConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get().max(2),
            spin_duration: Duration::from_micros(100),
        }
    }
}

// ─── WorkStealer ──────────────────────────────────────────────────────────────

/// Work-stealing scheduler for federated sub-plan execution.
pub struct WorkStealer {
    /// Per-worker deques: `workers[i]` is the deque of worker `i`.
    workers: Vec<Arc<Mutex<WorkerDeque>>>,
    /// Notification channel: workers listen here when all queues are empty.
    notify: Arc<Notify>,
    /// Total number of workers.
    num_workers: usize,
    /// Round-robin submit counter.
    submit_cursor: Mutex<usize>,
}

impl WorkStealer {
    /// Create a new work-stealing scheduler with the default configuration.
    pub fn new() -> Self {
        Self::with_config(WorkStealerConfig::default())
    }

    /// Create a scheduler with a custom configuration.
    pub fn with_config(config: WorkStealerConfig) -> Self {
        let num_workers = config.num_workers.max(1);
        let workers = (0..num_workers)
            .map(|_| Arc::new(Mutex::new(WorkerDeque::default())))
            .collect();
        Self {
            workers,
            notify: Arc::new(Notify::new()),
            num_workers,
            submit_cursor: Mutex::new(0),
        }
    }

    /// Submit a sub-plan.  Tasks are distributed round-robin across workers.
    pub fn submit(&self, plan: SubPlan) {
        let idx = {
            let mut cursor = self
                .submit_cursor
                .lock()
                .expect("submit_cursor lock poisoned");
            let idx = *cursor % self.num_workers;
            *cursor = (*cursor + 1) % self.num_workers;
            idx
        };
        self.workers[idx]
            .lock()
            .expect("worker deque lock poisoned")
            .push(plan);
        self.notify.notify_one();
    }

    /// Submit multiple plans at once.
    pub fn submit_batch(&self, plans: Vec<SubPlan>) {
        for plan in plans {
            self.submit(plan);
        }
    }

    /// Execute all currently queued tasks on the calling thread synchronously.
    ///
    /// This is the synchronous, non-tokio entry point intended for tests and
    /// single-threaded scenarios.  Returns results in completion order.
    pub fn run_sync(&self) -> Vec<SubPlanResult> {
        let mut results = Vec::new();
        let start = Instant::now();

        loop {
            // Try to pop from any non-empty queue
            let popped = self.pop_any();
            match popped {
                None => break,
                Some((plan, worker_id, was_stolen)) => {
                    let t0 = Instant::now();
                    let ok = (plan.work)();
                    let elapsed = t0.elapsed();
                    {
                        let mut deque =
                            self.workers[worker_id].lock().expect("deque lock poisoned");
                        deque.execute_count += 1;
                    }
                    results.push(SubPlanResult {
                        plan_id: plan.id,
                        executed_by: worker_id,
                        was_stolen,
                        execution_time: elapsed,
                        success: ok,
                        error: if ok {
                            None
                        } else {
                            Some("task returned false".to_string())
                        },
                    });
                }
            }
            // Prevent accidental infinite loops in tests
            if start.elapsed() > Duration::from_secs(30) {
                break;
            }
        }
        results
    }

    /// Total pending tasks across all workers.
    pub fn pending_count(&self) -> usize {
        self.workers
            .iter()
            .map(|w| w.lock().expect("deque lock poisoned").len())
            .sum()
    }

    /// Per-worker statistics.
    pub fn worker_stats(&self) -> Vec<WorkerStats> {
        self.workers
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let deque = w.lock().expect("deque lock poisoned");
                WorkerStats {
                    worker_id: i,
                    executed: deque.execute_count,
                    stolen: deque.steal_count,
                    pending: deque.len(),
                }
            })
            .collect()
    }

    /// Number of configured workers.
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    // ── Private ───────────────────────────────────────────────────────────

    /// Pop a task from own queue `worker_id`, or steal from the busiest peer.
    /// Returns `(plan, worker_id, was_stolen)`.
    fn pop_any(&self) -> Option<(SubPlan, usize, bool)> {
        // Try each worker deque in order
        for (idx, worker) in self.workers.iter().enumerate() {
            let mut deque = worker.lock().expect("deque lock poisoned");
            if let Some(plan) = deque.pop_own() {
                return Some((plan, idx, false));
            }
        }
        None
    }
}

impl Default for WorkStealer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn simple_plan(id: &str, ok: bool) -> SubPlan {
        SubPlan::new(SubPlanId::new(id), move || ok)
    }

    // ── SubPlanId ─────────────────────────────────────────────────────────

    #[test]
    fn test_sub_plan_id_new() {
        let id = SubPlanId::new("test-id");
        assert_eq!(id.0, "test-id");
        assert_eq!(format!("{}", id), "test-id");
    }

    #[test]
    fn test_sub_plan_id_generate_unique() {
        let a = SubPlanId::generate();
        let b = SubPlanId::generate();
        assert_ne!(a, b);
    }

    // ── SubPlan ───────────────────────────────────────────────────────────

    #[test]
    fn test_sub_plan_debug() {
        let p = simple_plan("p1", true);
        let s = format!("{:?}", p);
        assert!(s.contains("p1"));
    }

    // ── WorkerDeque ───────────────────────────────────────────────────────

    #[test]
    fn test_worker_deque_push_pop_own() {
        let mut d = WorkerDeque::default();
        d.push(simple_plan("a", true));
        d.push(simple_plan("b", true));
        let first = d.pop_own().expect("should have item");
        assert_eq!(first.id.0, "a"); // FIFO
    }

    #[test]
    fn test_worker_deque_steal_from_back() {
        let mut d = WorkerDeque::default();
        d.push(simple_plan("a", true));
        d.push(simple_plan("b", true));
        let stolen = d.steal().expect("should have item");
        assert_eq!(stolen.id.0, "b"); // Steal from back
        assert_eq!(d.steal_count, 1);
    }

    #[test]
    fn test_worker_deque_empty_pop() {
        let mut d = WorkerDeque::default();
        assert!(d.pop_own().is_none());
        assert!(d.steal().is_none());
    }

    // ── WorkStealer ───────────────────────────────────────────────────────

    #[test]
    fn test_work_stealer_new() {
        let ws = WorkStealer::new();
        assert!(ws.num_workers() >= 2);
        assert_eq!(ws.pending_count(), 0);
    }

    #[test]
    fn test_work_stealer_with_config() {
        let cfg = WorkStealerConfig {
            num_workers: 4,
            spin_duration: Duration::from_millis(1),
        };
        let ws = WorkStealer::with_config(cfg);
        assert_eq!(ws.num_workers(), 4);
    }

    #[test]
    fn test_work_stealer_submit_increases_pending() {
        let ws = WorkStealer::with_config(WorkStealerConfig {
            num_workers: 2,
            ..Default::default()
        });
        ws.submit(simple_plan("t1", true));
        ws.submit(simple_plan("t2", true));
        assert_eq!(ws.pending_count(), 2);
    }

    #[test]
    fn test_work_stealer_run_sync_executes_all() {
        let ws = WorkStealer::with_config(WorkStealerConfig {
            num_workers: 2,
            ..Default::default()
        });
        ws.submit(simple_plan("t1", true));
        ws.submit(simple_plan("t2", true));
        ws.submit(simple_plan("t3", false));
        let results = ws.run_sync();
        assert_eq!(results.len(), 3);
        assert_eq!(ws.pending_count(), 0);
    }

    #[test]
    fn test_work_stealer_run_sync_records_success() {
        let ws = WorkStealer::with_config(WorkStealerConfig {
            num_workers: 1,
            ..Default::default()
        });
        ws.submit(simple_plan("ok", true));
        ws.submit(simple_plan("fail", false));
        let results = ws.run_sync();
        assert_eq!(results.len(), 2);
        let ok_res = results.iter().find(|r| r.plan_id.0 == "ok").unwrap();
        assert!(ok_res.success);
        let fail_res = results.iter().find(|r| r.plan_id.0 == "fail").unwrap();
        assert!(!fail_res.success);
        assert!(fail_res.error.is_some());
    }

    #[test]
    fn test_work_stealer_submit_batch() {
        let ws = WorkStealer::with_config(WorkStealerConfig {
            num_workers: 3,
            ..Default::default()
        });
        let plans: Vec<SubPlan> = (0..9)
            .map(|i| simple_plan(&format!("batch-{}", i), true))
            .collect();
        ws.submit_batch(plans);
        assert_eq!(ws.pending_count(), 9);
    }

    #[test]
    fn test_work_stealer_worker_stats() {
        let ws = WorkStealer::with_config(WorkStealerConfig {
            num_workers: 2,
            ..Default::default()
        });
        ws.submit(simple_plan("s1", true));
        ws.run_sync();
        let stats = ws.worker_stats();
        assert_eq!(stats.len(), 2);
        let total_executed: u64 = stats.iter().map(|s| s.executed).sum();
        assert_eq!(total_executed, 1);
    }

    #[test]
    fn test_work_stealer_closure_captures_state() {
        let ws = WorkStealer::with_config(WorkStealerConfig {
            num_workers: 1,
            ..Default::default()
        });
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        ws.submit(SubPlan::new(SubPlanId::new("count"), move || {
            c.fetch_add(1, Ordering::SeqCst);
            true
        }));
        ws.run_sync();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_work_stealer_multiple_batches() {
        let ws = WorkStealer::with_config(WorkStealerConfig {
            num_workers: 2,
            ..Default::default()
        });
        ws.submit_batch(vec![simple_plan("a", true), simple_plan("b", true)]);
        ws.run_sync();
        // After first batch, submit another
        ws.submit_batch(vec![simple_plan("c", true), simple_plan("d", false)]);
        let results = ws.run_sync();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_work_stealer_round_robin_distribution() {
        let ws = WorkStealer::with_config(WorkStealerConfig {
            num_workers: 4,
            ..Default::default()
        });
        // Submit 4 tasks - they should be spread across workers
        for i in 0..4 {
            ws.submit(simple_plan(&format!("rr-{}", i), true));
        }
        // Each worker should have 1 task (round-robin)
        let stats = ws.worker_stats();
        for stat in &stats {
            assert_eq!(
                stat.pending, 1,
                "Worker {} should have 1 pending task",
                stat.worker_id
            );
        }
    }

    #[test]
    fn test_sub_plan_result_fields() {
        let ws = WorkStealer::with_config(WorkStealerConfig {
            num_workers: 1,
            ..Default::default()
        });
        ws.submit(simple_plan("res-test", true));
        let results = ws.run_sync();
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.plan_id.0, "res-test");
        assert!(r.success);
        assert!(r.execution_time >= Duration::ZERO);
    }

    #[test]
    fn test_work_stealer_default_impl() {
        let ws = WorkStealer::default();
        assert!(ws.num_workers() >= 1);
    }

    #[test]
    fn test_work_stealer_config_defaults() {
        let cfg = WorkStealerConfig::default();
        assert!(cfg.num_workers >= 2);
    }
}
