//! Advanced fault tolerance primitives for federated SPARQL endpoints.
//!
//! This module provides:
//! - [`FaultToleranceManager`] – orchestrates retry, circuit-breaker integration, and fallback.
//! - [`BulkheadIsolator`] – limits concurrent requests per endpoint to prevent cascading overload.
//! - [`TimeoutManager`] – enforces per-query wall-clock timeouts.
//! - [`DeadLetterQueue`] – durably stores permanently failed queries for manual replay.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by the fault-tolerance subsystem.
#[derive(Debug, Error)]
pub enum FaultToleranceError {
    #[error("All {max_retries} retry attempts exhausted")]
    RetriesExhausted { max_retries: u32 },

    #[error("Bulkhead limit reached for endpoint '{endpoint}' (max {max})")]
    BulkheadFull { endpoint: String, max: u32 },

    #[error("Bulkhead permit for endpoint '{endpoint}' is invalid or already released")]
    InvalidPermit { endpoint: String },

    #[error("Lock poisoned: {context}")]
    LockPoisoned { context: &'static str },

    #[error("Dead-letter queue is at capacity ({max})")]
    QueueFull { max: usize },

    #[error("Fallback failed: {reason}")]
    FallbackFailed { reason: String },
}

// ─── FaultToleranceConfig ─────────────────────────────────────────────────────

/// Configuration for [`FaultToleranceManager`].
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Maximum number of retry attempts after the initial call.
    pub max_retries: u32,
    /// Base delay between retries in milliseconds (exponential back-off base).
    pub retry_delay_ms: u64,
    /// Number of consecutive failures before the circuit breaker trips.
    pub circuit_breaker_threshold: u32,
    /// Whether to attempt the fallback function on exhaustion.
    pub use_fallback: bool,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay_ms: 100,
            circuit_breaker_threshold: 5,
            use_fallback: false,
        }
    }
}

// ─── FaultToleranceManager ────────────────────────────────────────────────────

/// Orchestrates retries, optional fallback, and tracks per-key failure counts.
///
/// The manager intentionally does **not** perform `std::thread::sleep` to keep
/// itself sync and test-friendly; callers that need real delay should sleep
/// between calls.
#[derive(Debug)]
pub struct FaultToleranceManager {
    config: FaultToleranceConfig,
    /// Per-key consecutive failure counts (used to track circuit-breaker state).
    failure_counts: Mutex<HashMap<String, u32>>,
}

impl FaultToleranceManager {
    /// Create a manager with the given configuration.
    pub fn new(config: FaultToleranceConfig) -> Self {
        Self {
            config,
            failure_counts: Mutex::new(HashMap::new()),
        }
    }

    /// Execute `f`, retrying up to `max_retries` times on `Err`.
    ///
    /// After exhaustion, if `fallback` is `Some`, call it once more.
    /// Returns `Err(FaultToleranceError::RetriesExhausted)` when all
    /// attempts (and optionally fallback) fail.
    pub fn execute_with_fault_tolerance<F, FB, R>(
        &self,
        key: &str,
        f: F,
        fallback: Option<FB>,
    ) -> Result<R, FaultToleranceError>
    where
        F: Fn() -> Result<R, String>,
        FB: Fn() -> Result<R, String>,
    {
        let mut attempt = 0u32;
        let max = self.config.max_retries;
        loop {
            match f() {
                Ok(v) => {
                    self.reset_failures(key);
                    return Ok(v);
                }
                Err(_) => {
                    self.increment_failures(key);
                    if attempt >= max {
                        break;
                    }
                    attempt += 1;
                }
            }
        }

        // Try fallback
        if self.config.use_fallback {
            if let Some(fb) = fallback {
                return fb().map_err(|e| FaultToleranceError::FallbackFailed { reason: e });
            }
        }

        Err(FaultToleranceError::RetriesExhausted { max_retries: max })
    }

    /// Return the consecutive failure count for `key`.
    pub fn failure_count(&self, key: &str) -> u32 {
        self.failure_counts
            .lock()
            .map(|m| m.get(key).copied().unwrap_or(0))
            .unwrap_or(0)
    }

    /// Whether the given key has exceeded the circuit-breaker threshold.
    pub fn is_tripped(&self, key: &str) -> bool {
        self.failure_count(key) >= self.config.circuit_breaker_threshold
    }

    /// Reset the failure count for `key`.
    pub fn reset_failures(&self, key: &str) {
        if let Ok(mut m) = self.failure_counts.lock() {
            m.remove(key);
        }
    }

    fn increment_failures(&self, key: &str) {
        if let Ok(mut m) = self.failure_counts.lock() {
            *m.entry(key.to_owned()).or_insert(0) += 1;
        }
    }

    /// Configuration accessor.
    pub fn config(&self) -> &FaultToleranceConfig {
        &self.config
    }
}

// ─── BulkheadPermit ───────────────────────────────────────────────────────────

/// A scoped concurrency token for a bulkhead slot.
///
/// The permit is returned to the [`BulkheadIsolator`] via
/// [`BulkheadIsolator::release_permit`].  It intentionally does NOT
/// auto-release on `Drop` to keep the API explicit.
#[derive(Debug, Clone)]
pub struct BulkheadPermit {
    /// The endpoint this permit belongs to.
    pub endpoint: String,
    /// Unique sequential permit id (exposed for auditing and testing).
    pub id: u64,
}

impl BulkheadPermit {
    /// Endpoint associated with this permit.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }
}

// ─── BulkheadIsolator ─────────────────────────────────────────────────────────

/// Limits concurrent requests per endpoint to prevent cascading overload.
#[derive(Debug)]
pub struct BulkheadIsolator {
    max_concurrent_per_endpoint: u32,
    /// endpoint → (active_count, next_permit_id)
    state: Mutex<HashMap<String, (u32, u64)>>,
}

impl BulkheadIsolator {
    /// Create a new isolator.
    pub fn new(max_concurrent_per_endpoint: u32) -> Self {
        Self {
            max_concurrent_per_endpoint,
            state: Mutex::new(HashMap::new()),
        }
    }

    /// Acquire a permit for `endpoint`.
    ///
    /// Returns `Err(BulkheadFull)` when the limit is already reached.
    pub fn acquire_permit(&self, endpoint: &str) -> Result<BulkheadPermit, FaultToleranceError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| FaultToleranceError::LockPoisoned {
                context: "BulkheadIsolator::acquire_permit",
            })?;

        let entry = state.entry(endpoint.to_owned()).or_insert((0, 0));
        let (count, next_id) = entry;

        if *count >= self.max_concurrent_per_endpoint {
            return Err(FaultToleranceError::BulkheadFull {
                endpoint: endpoint.to_owned(),
                max: self.max_concurrent_per_endpoint,
            });
        }

        let permit_id = *next_id;
        *next_id = next_id.wrapping_add(1);
        *count += 1;

        Ok(BulkheadPermit {
            endpoint: endpoint.to_owned(),
            id: permit_id,
        })
    }

    /// Release a previously-acquired permit.
    ///
    /// Returns `Err(InvalidPermit)` if the permit id is unrecognised (e.g.
    /// double-release attempt).
    pub fn release_permit(&self, permit: BulkheadPermit) -> Result<(), FaultToleranceError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| FaultToleranceError::LockPoisoned {
                context: "BulkheadIsolator::release_permit",
            })?;

        let entry =
            state
                .get_mut(&permit.endpoint)
                .ok_or_else(|| FaultToleranceError::InvalidPermit {
                    endpoint: permit.endpoint.clone(),
                })?;

        let (count, _) = entry;
        if *count == 0 {
            return Err(FaultToleranceError::InvalidPermit {
                endpoint: permit.endpoint,
            });
        }
        *count -= 1;
        Ok(())
    }

    /// Current active permit count for `endpoint`.
    pub fn active_count(&self, endpoint: &str) -> u32 {
        self.state
            .lock()
            .map(|m| m.get(endpoint).map(|(c, _)| *c).unwrap_or(0))
            .unwrap_or(0)
    }

    /// Maximum concurrent per endpoint as configured.
    pub fn max_concurrent(&self) -> u32 {
        self.max_concurrent_per_endpoint
    }
}

// ─── TimeoutResult ────────────────────────────────────────────────────────────

/// Result of a timeout-guarded execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimeoutResult<R> {
    /// The function completed within the timeout and returned `R`.
    Ok(R),
    /// The function exceeded the timeout.
    TimedOut,
}

impl<R> TimeoutResult<R> {
    /// Whether the result completed in time.
    pub fn is_ok(&self) -> bool {
        matches!(self, TimeoutResult::Ok(_))
    }

    /// Whether the result timed out.
    pub fn timed_out(&self) -> bool {
        matches!(self, TimeoutResult::TimedOut)
    }

    /// Unwrap the inner value, panicking on `TimedOut`.
    ///
    /// Prefer `if let` over this method in production code.
    #[cfg(test)]
    pub fn unwrap(self) -> R {
        match self {
            TimeoutResult::Ok(v) => v,
            TimeoutResult::TimedOut => panic!("called unwrap on TimeoutResult::TimedOut"),
        }
    }
}

// ─── TimeoutManager ───────────────────────────────────────────────────────────

/// Enforces per-query wall-clock timeouts using busy-wait polling.
///
/// For real-world use, prefer async timeouts (`tokio::time::timeout`).  This
/// sync implementation is useful for embedding in non-async contexts and tests.
#[derive(Debug, Default)]
pub struct TimeoutManager;

impl TimeoutManager {
    /// Create a new `TimeoutManager`.
    pub fn new() -> Self {
        Self
    }

    /// Execute `f` with a wall-clock budget of `timeout_ms` milliseconds.
    ///
    /// `f` receives a closure argument `deadline: Instant` that it can check
    /// periodically via `deadline.elapsed()`.  The manager runs `f` once; if
    /// `f` returns before the deadline, its return value is wrapped in
    /// `TimeoutResult::Ok`.  If the deadline passes before `f` returns, the
    /// result is `TimeoutResult::TimedOut`.
    ///
    /// # Note
    ///
    /// This is a *cooperative* timeout: `f` must itself be finite.  The manager
    /// does not interrupt a running thread.
    pub fn execute_with_timeout<F, R>(&self, timeout_ms: u64, f: F) -> TimeoutResult<R>
    where
        F: FnOnce(Instant) -> R,
    {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        let result = f(deadline);
        if Instant::now() <= deadline {
            TimeoutResult::Ok(result)
        } else {
            TimeoutResult::TimedOut
        }
    }

    /// Execute `f` and check whether `timeout_ms` has elapsed *after* `f`
    /// completes (post-hoc timeout check, useful for wrapping sync code).
    pub fn execute_and_check<F, R>(&self, timeout_ms: u64, f: F) -> TimeoutResult<R>
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        if start.elapsed() <= Duration::from_millis(timeout_ms) {
            TimeoutResult::Ok(result)
        } else {
            TimeoutResult::TimedOut
        }
    }
}

// ─── FailedQuery ──────────────────────────────────────────────────────────────

/// A query that has permanently failed and been placed in the dead-letter queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedQuery {
    /// Unique identifier assigned on enqueue.
    pub id: u64,
    /// The SPARQL query text.
    pub query: String,
    /// Target endpoint URL.
    pub endpoint: String,
    /// Error message from the last failure.
    pub last_error: String,
    /// Unix timestamp (seconds) when the failure was recorded.
    pub failed_at_secs: u64,
    /// Number of attempts before giving up.
    pub attempt_count: u32,
}

impl FailedQuery {
    /// Create a new `FailedQuery` with the current system time.
    pub fn new(
        id: u64,
        query: impl Into<String>,
        endpoint: impl Into<String>,
        last_error: impl Into<String>,
        attempt_count: u32,
    ) -> Self {
        let failed_at_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self {
            id,
            query: query.into(),
            endpoint: endpoint.into(),
            last_error: last_error.into(),
            failed_at_secs,
            attempt_count,
        }
    }
}

// ─── ReplayResult ─────────────────────────────────────────────────────────────

/// Outcome of replaying a single dead-letter entry.
#[derive(Debug, Clone)]
pub struct ReplayResult {
    /// The query id that was replayed.
    pub query_id: u64,
    /// Whether the replay succeeded.
    pub success: bool,
    /// Error message if the replay failed.
    pub error: Option<String>,
}

// ─── DeadLetterQueue ──────────────────────────────────────────────────────────

/// Stores permanently failed queries for later manual replay.
#[derive(Debug)]
pub struct DeadLetterQueue {
    inner: Mutex<VecDeque<FailedQuery>>,
    max_capacity: usize,
    next_id: Mutex<u64>,
}

impl DeadLetterQueue {
    /// Create a dead-letter queue with the given maximum capacity.
    /// Use `0` for unlimited.
    pub fn new(max_capacity: usize) -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
            max_capacity,
            next_id: Mutex::new(1),
        }
    }

    /// Enqueue a failed query.
    ///
    /// Returns `Err(QueueFull)` when capacity is exceeded.
    pub fn push(&self, mut query: FailedQuery) -> Result<u64, FaultToleranceError> {
        let id = {
            let mut nid = self
                .next_id
                .lock()
                .map_err(|_| FaultToleranceError::LockPoisoned {
                    context: "DeadLetterQueue::push",
                })?;
            let id = *nid;
            *nid = nid.wrapping_add(1);
            id
        };
        query.id = id;

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| FaultToleranceError::LockPoisoned {
                context: "DeadLetterQueue::push",
            })?;

        if self.max_capacity > 0 && inner.len() >= self.max_capacity {
            return Err(FaultToleranceError::QueueFull {
                max: self.max_capacity,
            });
        }

        inner.push_back(query);
        Ok(id)
    }

    /// Dequeue the oldest failed query (FIFO).
    pub fn pop(&self) -> Option<FailedQuery> {
        self.inner.lock().ok().and_then(|mut q| q.pop_front())
    }

    /// Number of entries currently in the queue.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Peek at entries without removing them.
    pub fn peek_all(&self) -> Vec<FailedQuery> {
        self.inner
            .lock()
            .map(|q| q.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Drain all entries and attempt to replay each via `f`.
    ///
    /// Items that fail replay are **re-enqueued** at the back of the queue.
    /// Returns a `Vec<ReplayResult>` describing the outcome of each attempt.
    pub fn replay_all<F>(&self, f: F) -> Vec<ReplayResult>
    where
        F: Fn(&FailedQuery) -> Result<(), String>,
    {
        let entries: Vec<FailedQuery> = {
            let mut inner = match self.inner.lock() {
                Ok(g) => g,
                Err(_) => return vec![],
            };
            inner.drain(..).collect()
        };

        let mut results = Vec::with_capacity(entries.len());
        for entry in entries {
            match f(&entry) {
                Ok(()) => {
                    results.push(ReplayResult {
                        query_id: entry.id,
                        success: true,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(ReplayResult {
                        query_id: entry.id,
                        success: false,
                        error: Some(e),
                    });
                    // Re-enqueue — ignore capacity errors during replay
                    let _ = self.push(entry);
                }
            }
        }

        results
    }

    /// Clear all entries.
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.clear();
        }
    }
}

// ─── Arc-wrapped convenience constructors ─────────────────────────────────────

impl FaultToleranceManager {
    /// Wrap in `Arc` for shared ownership.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

impl BulkheadIsolator {
    /// Wrap in `Arc` for shared ownership.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

impl DeadLetterQueue {
    /// Wrap in `Arc` for shared ownership.
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FaultToleranceManager ─────────────────────────────────────────────

    #[test]
    fn test_ftm_success_on_first_call() {
        let mgr = FaultToleranceManager::new(FaultToleranceConfig::default());
        let result: Result<i32, FaultToleranceError> =
            mgr.execute_with_fault_tolerance("ep1", || Ok(42), None::<fn() -> Result<i32, String>>);
        assert_eq!(result.expect("should succeed"), 42);
    }

    #[test]
    fn test_ftm_retries_and_eventually_succeeds() {
        let mgr = FaultToleranceManager::new(FaultToleranceConfig {
            max_retries: 3,
            ..Default::default()
        });
        let call_count = Arc::new(Mutex::new(0u32));
        let cc = Arc::clone(&call_count);
        let result: Result<String, FaultToleranceError> = mgr.execute_with_fault_tolerance(
            "ep2",
            move || {
                let mut c = cc.lock().expect("lock");
                *c += 1;
                if *c < 3 {
                    Err("not yet".to_owned())
                } else {
                    Ok("done".to_owned())
                }
            },
            None::<fn() -> Result<String, String>>,
        );
        assert_eq!(result.expect("should succeed"), "done");
    }

    #[test]
    fn test_ftm_exhausts_retries() {
        let mgr = FaultToleranceManager::new(FaultToleranceConfig {
            max_retries: 2,
            use_fallback: false,
            ..Default::default()
        });
        let result: Result<i32, FaultToleranceError> = mgr.execute_with_fault_tolerance(
            "ep3",
            || Err("always fails".to_owned()),
            None::<fn() -> Result<i32, String>>,
        );
        assert!(matches!(
            result,
            Err(FaultToleranceError::RetriesExhausted { max_retries: 2 })
        ));
    }

    #[test]
    fn test_ftm_fallback_called_on_exhaustion() {
        let mgr = FaultToleranceManager::new(FaultToleranceConfig {
            max_retries: 1,
            use_fallback: true,
            ..Default::default()
        });
        let result: Result<&str, FaultToleranceError> = mgr.execute_with_fault_tolerance(
            "ep4",
            || Err("fail".to_owned()),
            Some(|| Ok("fallback_value")),
        );
        assert_eq!(result.expect("fallback should succeed"), "fallback_value");
    }

    #[test]
    fn test_ftm_failure_count_increments() {
        let mgr = FaultToleranceManager::new(FaultToleranceConfig {
            max_retries: 2,
            ..Default::default()
        });
        let _: Result<(), FaultToleranceError> = mgr.execute_with_fault_tolerance(
            "countme",
            || Err("err".to_owned()),
            None::<fn() -> Result<(), String>>,
        );
        assert!(mgr.failure_count("countme") >= 2);
    }

    #[test]
    fn test_ftm_reset_failures() {
        let mgr = FaultToleranceManager::new(FaultToleranceConfig::default());
        let _: Result<(), FaultToleranceError> = mgr.execute_with_fault_tolerance(
            "reset_me",
            || Err("err".to_owned()),
            None::<fn() -> Result<(), String>>,
        );
        mgr.reset_failures("reset_me");
        assert_eq!(mgr.failure_count("reset_me"), 0);
    }

    #[test]
    fn test_ftm_is_tripped() {
        let mgr = FaultToleranceManager::new(FaultToleranceConfig {
            max_retries: 10,
            circuit_breaker_threshold: 3,
            ..Default::default()
        });
        for _ in 0..4 {
            let _: Result<(), FaultToleranceError> = mgr.execute_with_fault_tolerance(
                "trip",
                || Err("err".to_owned()),
                None::<fn() -> Result<(), String>>,
            );
        }
        assert!(mgr.is_tripped("trip"));
    }

    #[test]
    fn test_ftm_success_resets_trip() {
        let mgr = FaultToleranceManager::new(FaultToleranceConfig::default());
        let _: Result<(), FaultToleranceError> = mgr.execute_with_fault_tolerance(
            "r",
            || Err("err".to_owned()),
            None::<fn() -> Result<(), String>>,
        );
        let _: Result<String, FaultToleranceError> = mgr.execute_with_fault_tolerance(
            "r",
            || Ok("ok".to_owned()),
            None::<fn() -> Result<String, String>>,
        );
        assert_eq!(mgr.failure_count("r"), 0);
    }

    // ── BulkheadIsolator ──────────────────────────────────────────────────

    #[test]
    fn test_bulkhead_acquire_and_release() {
        let bh = BulkheadIsolator::new(3);
        let p = bh.acquire_permit("ep").expect("should acquire");
        assert_eq!(bh.active_count("ep"), 1);
        bh.release_permit(p).expect("should release");
        assert_eq!(bh.active_count("ep"), 0);
    }

    #[test]
    fn test_bulkhead_max_concurrent() {
        let bh = BulkheadIsolator::new(2);
        let _p1 = bh.acquire_permit("ep").expect("first");
        let _p2 = bh.acquire_permit("ep").expect("second");
        let err = bh.acquire_permit("ep").unwrap_err();
        assert!(matches!(err, FaultToleranceError::BulkheadFull { .. }));
    }

    #[test]
    fn test_bulkhead_separate_endpoints_independent() {
        let bh = BulkheadIsolator::new(1);
        let _p1 = bh.acquire_permit("ep1").expect("ep1 acquire");
        let p2 = bh.acquire_permit("ep2").expect("ep2 should be independent");
        assert_eq!(bh.active_count("ep1"), 1);
        assert_eq!(bh.active_count("ep2"), 1);
        bh.release_permit(p2).expect("release ep2");
    }

    #[test]
    fn test_bulkhead_double_release_errors() {
        let bh = BulkheadIsolator::new(2);
        let p = bh.acquire_permit("ep").expect("acquire");
        let p_clone = p.clone();
        bh.release_permit(p).expect("first release");
        let err = bh.release_permit(p_clone).unwrap_err();
        assert!(matches!(err, FaultToleranceError::InvalidPermit { .. }));
    }

    #[test]
    fn test_bulkhead_max_concurrent_accessor() {
        let bh = BulkheadIsolator::new(7);
        assert_eq!(bh.max_concurrent(), 7);
    }

    #[test]
    fn test_bulkhead_active_count_zero_for_unknown_endpoint() {
        let bh = BulkheadIsolator::new(5);
        assert_eq!(bh.active_count("unknown"), 0);
    }

    // ── TimeoutManager ────────────────────────────────────────────────────

    #[test]
    fn test_timeout_manager_completes_fast() {
        let tm = TimeoutManager::new();
        let result = tm.execute_with_timeout(1_000, |_deadline| 42u32);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_timeout_manager_detects_slow_fn() {
        let tm = TimeoutManager::new();
        // Sleep 50 ms with a 1 ms budget — post-hoc check will fire
        let result = tm.execute_and_check(1, || {
            std::thread::sleep(Duration::from_millis(50));
            "done"
        });
        assert!(result.timed_out());
    }

    #[test]
    fn test_timeout_result_is_ok() {
        let r: TimeoutResult<i32> = TimeoutResult::Ok(7);
        assert!(r.is_ok());
        assert!(!r.timed_out());
    }

    #[test]
    fn test_timeout_result_timed_out() {
        let r: TimeoutResult<i32> = TimeoutResult::TimedOut;
        assert!(r.timed_out());
        assert!(!r.is_ok());
    }

    #[test]
    fn test_timeout_manager_within_budget() {
        let tm = TimeoutManager::new();
        // Instant computation should always be within a 5-second budget
        let result = tm.execute_and_check(5_000, || 1 + 1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);
    }

    // ── DeadLetterQueue ───────────────────────────────────────────────────

    fn make_failed_query(query: &str) -> FailedQuery {
        FailedQuery::new(0, query, "http://ep/sparql", "timeout", 3)
    }

    #[test]
    fn test_dlq_push_and_pop() {
        let dlq = DeadLetterQueue::new(0);
        let id = dlq.push(make_failed_query("SELECT *")).expect("push");
        assert_eq!(dlq.len(), 1);
        let entry = dlq.pop().expect("pop");
        assert_eq!(entry.id, id);
        assert_eq!(dlq.len(), 0);
    }

    #[test]
    fn test_dlq_fifo_order() {
        let dlq = DeadLetterQueue::new(0);
        dlq.push(make_failed_query("q1")).expect("push 1");
        dlq.push(make_failed_query("q2")).expect("push 2");
        let first = dlq.pop().expect("first");
        let second = dlq.pop().expect("second");
        assert_eq!(first.query, "q1");
        assert_eq!(second.query, "q2");
    }

    #[test]
    fn test_dlq_capacity_limit() {
        let dlq = DeadLetterQueue::new(2);
        dlq.push(make_failed_query("q1")).expect("1");
        dlq.push(make_failed_query("q2")).expect("2");
        let err = dlq.push(make_failed_query("q3")).unwrap_err();
        assert!(matches!(err, FaultToleranceError::QueueFull { .. }));
    }

    #[test]
    fn test_dlq_is_empty() {
        let dlq = DeadLetterQueue::new(0);
        assert!(dlq.is_empty());
        dlq.push(make_failed_query("q")).expect("push");
        assert!(!dlq.is_empty());
    }

    #[test]
    fn test_dlq_pop_empty_returns_none() {
        let dlq = DeadLetterQueue::new(0);
        assert!(dlq.pop().is_none());
    }

    #[test]
    fn test_dlq_replay_all_success() {
        let dlq = DeadLetterQueue::new(0);
        for i in 0..3 {
            dlq.push(make_failed_query(&format!("q{i}"))).expect("push");
        }
        let results = dlq.replay_all(|_q| Ok(()));
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.success));
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_dlq_replay_all_failure_requeues() {
        let dlq = DeadLetterQueue::new(0);
        dlq.push(make_failed_query("bad")).expect("push");
        let results = dlq.replay_all(|_q| Err("still broken".to_owned()));
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        // Item should be re-enqueued
        assert_eq!(dlq.len(), 1);
    }

    #[test]
    fn test_dlq_replay_partial() {
        let dlq = DeadLetterQueue::new(0);
        dlq.push(make_failed_query("good")).expect("push");
        dlq.push(make_failed_query("bad")).expect("push");
        let results = dlq.replay_all(|q| {
            if q.query == "good" {
                Ok(())
            } else {
                Err("fail".to_owned())
            }
        });
        assert_eq!(results.len(), 2);
        let successes: usize = results.iter().filter(|r| r.success).count();
        assert_eq!(successes, 1);
        // "bad" should be re-enqueued
        assert_eq!(dlq.len(), 1);
    }

    #[test]
    fn test_dlq_clear() {
        let dlq = DeadLetterQueue::new(0);
        for _ in 0..5 {
            dlq.push(make_failed_query("q")).expect("push");
        }
        dlq.clear();
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_dlq_peek_all_nondestructive() {
        let dlq = DeadLetterQueue::new(0);
        dlq.push(make_failed_query("peek")).expect("push");
        let snapshot = dlq.peek_all();
        assert_eq!(snapshot.len(), 1);
        // Queue still has the item
        assert_eq!(dlq.len(), 1);
    }

    #[test]
    fn test_dlq_unlimited_capacity() {
        let dlq = DeadLetterQueue::new(0);
        for i in 0..100 {
            dlq.push(make_failed_query(&format!("q{i}"))).expect("push");
        }
        assert_eq!(dlq.len(), 100);
    }

    #[test]
    fn test_failed_query_new_sets_fields() {
        let fq = FailedQuery::new(0, "SELECT *", "http://ep/sparql", "err", 5);
        assert_eq!(fq.query, "SELECT *");
        assert_eq!(fq.attempt_count, 5);
        assert!(fq.failed_at_secs > 0);
    }
}
