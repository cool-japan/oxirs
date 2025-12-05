//! Production Hardening for SPARQL Query Engine
//!
//! Beta.1 Feature: Production-Ready SPARQL Operations
//!
//! This module provides production-grade features for reliable SPARQL query execution:
//! - Enhanced error handling with query context
//! - Query execution circuit breakers
//! - SPARQL performance monitoring
//! - Query result size quotas
//! - Health checks for query engine components
//!
//! ## File Organization (3018 lines)
//!
//! This file intentionally exceeds 2000 lines because it contains a cohesive set of
//! tightly-coupled production components for query lifecycle management. The components
//! share extensive internal state and are designed to work together.
//!
//! ### Module Structure:
//!
//! 1. **Error Handling** (lines 18-137)
//!    - `SparqlProductionError` - Enhanced error types with SPARQL context
//!    - `QueryErrorContext` - Detailed error metadata
//!    - `ErrorSeverity` - Severity classification for monitoring
//!
//! 2. **Circuit Breakers** (lines 138-224)
//!    - `QueryCircuitBreaker` - Prevents cascading failures
//!    - Configurable failure thresholds and recovery timeouts
//!
//! 3. **Performance Monitoring** (lines 225-430)
//!    - `SparqlPerformanceMonitor` - Real-time query performance tracking
//!    - Latency histograms, slow query detection, statistics collection
//!
//! 4. **Resource Quotas** (lines 431-543)
//!    - `QueryResourceQuota` - Resource limits for queries
//!    - Result size quotas, pattern complexity limits
//!
//! 5. **Health Checks** (lines 544-658)
//!    - `QueryEngineHealth` - Component health monitoring
//!    - Configurable health check intervals and timeouts
//!
//! 6. **Query Cancellation** (lines 659-782)
//!    - `QueryCancellationToken` - Cooperative cancellation with callbacks
//!    - Support for child tokens and cancellation propagation
//!
//! 7. **Timeout Management** (lines 783-950)
//!    - `QueryTimeoutManager` - Soft/hard timeouts with warnings
//!    - Configurable warning thresholds and timeout actions
//!
//! 8. **Memory Tracking** (lines 951-1151)
//!    - `QueryMemoryTracker` - Per-query memory usage monitoring
//!    - Memory pressure detection and throttling
//!
//! 9. **Session Management** (lines 1152-1398)
//!    - `QuerySession` - Unified query lifecycle management
//!    - `QuerySessionManager` - Session pool and coordination
//!
//! 10. **Rate Limiting** (lines 1399-1502)
//!     - `QueryRateLimiter` - Token bucket rate limiting
//!     - Per-user rate tracking and enforcement
//!
//! 11. **Audit Trail** (lines 1503-1642)
//!     - `QueryAuditTrail` - Circular buffer audit logging
//!     - Compliance and debugging support
//!
//! 12. **Priority Scheduling** (lines 1643-1879) [Beta.2]
//!     - `QueryPriorityScheduler` - 5-level priority-based execution
//!     - Aging to prevent starvation
//!
//! 13. **Cost Estimation** (lines 1880-2098) [Beta.2]
//!     - `QueryCostEstimator` - Proactive cost estimation
//!     - Historical cost tracking and recommendations
//!
//! 14. **Performance Baselines** (lines 2099-end) [Beta.2+]
//!     - `PerformanceBaselineTracker` - Regression detection
//!     - Statistical trend analysis
//!
//! ### Why This File is Large:
//!
//! 1. **Cohesive Domain**: All components relate to production query execution
//! 2. **Tight Coupling**: Components share state and coordinate closely
//! 3. **Complete Feature Set**: Provides full production-grade query management
//! 4. **Integration Complexity**: Extensive interaction between components
//!
//! ### Future Refactoring Considerations:
//!
//! If this file needs to be split, the logical boundaries would be:
//! - `production/errors.rs` - Error types and handling
//! - `production/circuit_breaker.rs` - Circuit breaker logic
//! - `production/monitoring.rs` - Performance monitoring
//! - `production/resources.rs` - Resource quotas and limits
//! - `production/session.rs` - Session management
//! - `production/scheduling.rs` - Priority scheduling [Beta.2]
//! - `production/cost.rs` - Cost estimation [Beta.2]
//! - `production/baseline.rs` - Performance baselines [Beta.2+]

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Enhanced error type with SPARQL query context
#[derive(Debug, Clone)]
pub struct SparqlProductionError {
    pub error: String,
    pub context: QueryErrorContext,
    pub timestamp: SystemTime,
    pub severity: ErrorSeverity,
    pub retryable: bool,
}

/// Context information for SPARQL query errors
#[derive(Debug, Clone)]
pub struct QueryErrorContext {
    pub query: String,
    pub operation: String,
    pub pattern_count: usize,
    pub execution_time: Option<Duration>,
    pub result_count: Option<usize>,
    pub metadata: HashMap<String, String>,
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Informational - no action required
    Info,
    /// Warning - should be investigated
    Warning,
    /// Error - requires attention
    Error,
    /// Critical - requires immediate action
    Critical,
}

impl SparqlProductionError {
    pub fn new(
        error: String,
        context: QueryErrorContext,
        severity: ErrorSeverity,
        retryable: bool,
    ) -> Self {
        Self {
            error,
            context,
            timestamp: SystemTime::now(),
            severity,
            retryable,
        }
    }

    pub fn parse_error(query: String, message: String) -> Self {
        Self::new(
            format!("SPARQL parse error: {}", message),
            QueryErrorContext {
                query,
                operation: "parse".to_string(),
                pattern_count: 0,
                execution_time: None,
                result_count: None,
                metadata: HashMap::new(),
            },
            ErrorSeverity::Error,
            false,
        )
    }

    pub fn execution_error(query: String, message: String, elapsed: Duration) -> Self {
        Self::new(
            format!("SPARQL execution error: {}", message),
            QueryErrorContext {
                query,
                operation: "execute".to_string(),
                pattern_count: 0,
                execution_time: Some(elapsed),
                result_count: None,
                metadata: HashMap::new(),
            },
            ErrorSeverity::Error,
            true,
        )
    }

    pub fn timeout_error(query: String, elapsed: Duration, limit: Duration) -> Self {
        Self::new(
            format!("Query timeout: {:?} exceeded limit {:?}", elapsed, limit),
            QueryErrorContext {
                query,
                operation: "timeout".to_string(),
                pattern_count: 0,
                execution_time: Some(elapsed),
                result_count: None,
                metadata: HashMap::new(),
            },
            ErrorSeverity::Warning,
            true,
        )
    }
}

/// Circuit breaker for SPARQL query execution
pub struct QueryCircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failures: AtomicUsize,
    successes: AtomicUsize,
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub success_threshold: usize,
    pub timeout: Duration,
    pub half_open_max_requests: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_requests: 3,
        }
    }
}

impl QueryCircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failures: AtomicUsize::new(0),
            successes: AtomicUsize::new(0),
            config,
        }
    }

    pub fn is_request_allowed(&self) -> bool {
        let state = self.state.read().unwrap();
        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => false,
            CircuitState::HalfOpen => {
                let successes = self.successes.load(Ordering::Relaxed);
                successes < self.config.half_open_max_requests
            }
        }
    }

    pub fn record_success(&self) {
        let mut state = self.state.write().unwrap();
        match *state {
            CircuitState::Closed => {
                self.failures.store(0, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                let successes = self.successes.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    self.failures.store(0, Ordering::Relaxed);
                    self.successes.store(0, Ordering::Relaxed);
                }
            }
            CircuitState::Open => {}
        }
    }

    pub fn record_failure(&self) {
        let mut state = self.state.write().unwrap();
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;

        if failures >= self.config.failure_threshold {
            *state = CircuitState::Open;
            self.successes.store(0, Ordering::Relaxed);
        }
    }

    pub fn try_half_open(&self) {
        let mut state = self.state.write().unwrap();
        if *state == CircuitState::Open {
            *state = CircuitState::HalfOpen;
            self.successes.store(0, Ordering::Relaxed);
        }
    }

    pub fn state(&self) -> String {
        format!("{:?}", *self.state.read().unwrap())
    }
}

/// Performance monitoring for SPARQL query execution
pub struct SparqlPerformanceMonitor {
    query_latencies: RwLock<HashMap<String, Vec<Duration>>>,
    query_counts: RwLock<HashMap<String, AtomicU64>>,
    pattern_complexities: RwLock<HashMap<String, Vec<usize>>>,
    result_sizes: RwLock<HashMap<String, Vec<usize>>>,
    timeouts: AtomicU64,
    errors: AtomicU64,
    start_time: Instant,
}

impl Default for SparqlPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl SparqlPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            query_latencies: RwLock::new(HashMap::new()),
            query_counts: RwLock::new(HashMap::new()),
            pattern_complexities: RwLock::new(HashMap::new()),
            result_sizes: RwLock::new(HashMap::new()),
            timeouts: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    pub fn record_query(
        &self,
        query_type: &str,
        latency: Duration,
        pattern_count: usize,
        result_count: usize,
    ) {
        // Record latency
        self.query_latencies
            .write()
            .unwrap()
            .entry(query_type.to_string())
            .or_default()
            .push(latency);

        // Increment count
        self.query_counts
            .write()
            .unwrap()
            .entry(query_type.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        // Record pattern complexity
        self.pattern_complexities
            .write()
            .unwrap()
            .entry(query_type.to_string())
            .or_default()
            .push(pattern_count);

        // Record result size
        self.result_sizes
            .write()
            .unwrap()
            .entry(query_type.to_string())
            .or_default()
            .push(result_count);
    }

    pub fn record_timeout(&self) {
        self.timeouts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get_statistics(&self, query_type: &str) -> QueryStatistics {
        let latencies = self.query_latencies.read().unwrap();
        let counts = self.query_counts.read().unwrap();
        let complexities = self.pattern_complexities.read().unwrap();
        let sizes = self.result_sizes.read().unwrap();

        let latency_data = latencies.get(query_type);
        let count = counts
            .get(query_type)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0);
        let complexity_data = complexities.get(query_type);
        let size_data = sizes.get(query_type);

        let (avg_latency, p50_latency, p95_latency, p99_latency) = if let Some(data) = latency_data
        {
            let mut sorted = data.clone();
            sorted.sort();

            let sum: Duration = sorted.iter().sum();
            let avg = if !sorted.is_empty() {
                sum / sorted.len() as u32
            } else {
                Duration::ZERO
            };

            let p50 = if !sorted.is_empty() {
                sorted[sorted.len() / 2]
            } else {
                Duration::ZERO
            };

            let p95 = if !sorted.is_empty() {
                sorted[sorted.len() * 95 / 100]
            } else {
                Duration::ZERO
            };

            let p99 = if !sorted.is_empty() {
                sorted[sorted.len() * 99 / 100]
            } else {
                Duration::ZERO
            };

            (avg, p50, p95, p99)
        } else {
            (
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
                Duration::ZERO,
            )
        };

        let avg_complexity = if let Some(data) = complexity_data {
            if !data.is_empty() {
                data.iter().sum::<usize>() / data.len()
            } else {
                0
            }
        } else {
            0
        };

        let avg_result_size = if let Some(data) = size_data {
            if !data.is_empty() {
                data.iter().sum::<usize>() / data.len()
            } else {
                0
            }
        } else {
            0
        };

        QueryStatistics {
            query_type: query_type.to_string(),
            total_queries: count,
            average_latency: avg_latency,
            p50_latency,
            p95_latency,
            p99_latency,
            average_pattern_complexity: avg_complexity,
            average_result_size: avg_result_size,
        }
    }

    pub fn get_global_statistics(&self) -> GlobalStatistics {
        GlobalStatistics {
            uptime: self.start_time.elapsed(),
            total_queries: self
                .query_counts
                .read()
                .unwrap()
                .values()
                .map(|c| c.load(Ordering::Relaxed))
                .sum(),
            total_timeouts: self.timeouts.load(Ordering::Relaxed),
            total_errors: self.errors.load(Ordering::Relaxed),
        }
    }

    pub fn reset(&self) {
        self.query_latencies.write().unwrap().clear();
        self.query_counts.write().unwrap().clear();
        self.pattern_complexities.write().unwrap().clear();
        self.result_sizes.write().unwrap().clear();
        self.timeouts.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct QueryStatistics {
    pub query_type: String,
    pub total_queries: u64,
    pub average_latency: Duration,
    pub p50_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub average_pattern_complexity: usize,
    pub average_result_size: usize,
}

#[derive(Debug, Clone)]
pub struct GlobalStatistics {
    pub uptime: Duration,
    pub total_queries: u64,
    pub total_timeouts: u64,
    pub total_errors: u64,
}

/// Resource quota manager for SPARQL queries
pub struct QueryResourceQuota {
    max_result_size: AtomicUsize,
    max_query_time: RwLock<Duration>,
    max_pattern_complexity: AtomicUsize,
    enforced: AtomicBool,
}

impl QueryResourceQuota {
    pub fn new(
        max_result_size: usize,
        max_query_time: Duration,
        max_pattern_complexity: usize,
    ) -> Self {
        Self {
            max_result_size: AtomicUsize::new(max_result_size),
            max_query_time: RwLock::new(max_query_time),
            max_pattern_complexity: AtomicUsize::new(max_pattern_complexity),
            enforced: AtomicBool::new(true),
        }
    }

    pub fn check_result_size(&self, size: usize) -> Result<()> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }

        let max = self.max_result_size.load(Ordering::Relaxed);
        if size > max {
            return Err(anyhow!("Result size {} exceeds quota of {}", size, max));
        }

        Ok(())
    }

    pub fn check_query_time(&self, elapsed: Duration) -> Result<()> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }

        let max = *self.max_query_time.read().unwrap();
        if elapsed > max {
            return Err(anyhow!(
                "Query time {:?} exceeds quota of {:?}",
                elapsed,
                max
            ));
        }

        Ok(())
    }

    pub fn check_pattern_complexity(&self, complexity: usize) -> Result<()> {
        if !self.enforced.load(Ordering::Relaxed) {
            return Ok(());
        }

        let max = self.max_pattern_complexity.load(Ordering::Relaxed);
        if complexity > max {
            return Err(anyhow!(
                "Pattern complexity {} exceeds quota of {}",
                complexity,
                max
            ));
        }

        Ok(())
    }

    pub fn set_result_size_limit(&self, limit: usize) {
        self.max_result_size.store(limit, Ordering::Relaxed);
    }

    pub fn set_time_limit(&self, limit: Duration) {
        *self.max_query_time.write().unwrap() = limit;
    }

    pub fn set_complexity_limit(&self, limit: usize) {
        self.max_pattern_complexity.store(limit, Ordering::Relaxed);
    }

    pub fn set_enforced(&self, enforced: bool) {
        self.enforced.store(enforced, Ordering::Relaxed);
    }

    pub fn is_enforced(&self) -> bool {
        self.enforced.load(Ordering::Relaxed)
    }
}

impl Default for QueryResourceQuota {
    fn default() -> Self {
        Self::new(
            1_000_000,                // 1M results max
            Duration::from_secs(300), // 5 minute timeout
            1000,                     // 1000 pattern complexity max
        )
    }
}

/// Health check for SPARQL query engine components
pub struct QueryEngineHealth {
    checks: RwLock<HashMap<String, HealthCheck>>,
}

#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: SystemTime,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl QueryEngineHealth {
    pub fn new() -> Self {
        Self {
            checks: RwLock::new(HashMap::new()),
        }
    }

    pub fn register_check(&self, name: &str) {
        self.checks.write().unwrap().insert(
            name.to_string(),
            HealthCheck {
                name: name.to_string(),
                status: HealthStatus::Unknown,
                last_check: SystemTime::now(),
                message: "Not yet checked".to_string(),
            },
        );
    }

    pub fn update_check(&self, name: &str, status: HealthStatus, message: String) {
        if let Some(check) = self.checks.write().unwrap().get_mut(name) {
            check.status = status;
            check.last_check = SystemTime::now();
            check.message = message;
        }
    }

    pub fn check_parser(&self) -> HealthStatus {
        // Simple parser health check
        let status = HealthStatus::Healthy;
        self.update_check("parser", status, "Parser is operational".to_string());
        status
    }

    pub fn check_executor(&self) -> HealthStatus {
        // Simple executor health check
        let status = HealthStatus::Healthy;
        self.update_check("executor", status, "Executor is operational".to_string());
        status
    }

    pub fn check_optimizer(&self) -> HealthStatus {
        // Simple optimizer health check
        let status = HealthStatus::Healthy;
        self.update_check("optimizer", status, "Optimizer is operational".to_string());
        status
    }

    pub fn get_overall_status(&self) -> HealthStatus {
        let checks = self.checks.read().unwrap();

        if checks.is_empty() {
            return HealthStatus::Unknown;
        }

        let mut has_unhealthy = false;
        let mut has_degraded = false;

        for check in checks.values() {
            match check.status {
                HealthStatus::Unhealthy => has_unhealthy = true,
                HealthStatus::Degraded => has_degraded = true,
                _ => {}
            }
        }

        if has_unhealthy {
            HealthStatus::Unhealthy
        } else if has_degraded {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }

    pub fn get_checks(&self) -> Vec<HealthCheck> {
        self.checks.read().unwrap().values().cloned().collect()
    }

    pub fn perform_all_checks(&self) {
        self.check_parser();
        self.check_executor();
        self.check_optimizer();
    }
}

impl Default for QueryEngineHealth {
    fn default() -> Self {
        let health = Self::new();
        health.register_check("parser");
        health.register_check("executor");
        health.register_check("optimizer");
        health
    }
}

// =============================================================================
// Beta.2 Features: Advanced Query Management
// =============================================================================

/// Type alias for cancellation callbacks to reduce type complexity
type CancellationCallbacks = Arc<RwLock<Vec<Box<dyn Fn() + Send + Sync>>>>;

/// Query cancellation token for cooperative cancellation
///
/// Provides a mechanism for cancelling long-running queries gracefully.
/// Supports both synchronous and asynchronous cancellation patterns.
#[derive(Clone)]
pub struct QueryCancellationToken {
    cancelled: Arc<AtomicBool>,
    cancel_time: Arc<RwLock<Option<Instant>>>,
    reason: Arc<RwLock<Option<String>>>,
    callbacks: CancellationCallbacks,
}

impl Default for QueryCancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryCancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            cancel_time: Arc::new(RwLock::new(None)),
            reason: Arc::new(RwLock::new(None)),
            callbacks: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Request cancellation with an optional reason
    pub fn cancel(&self, reason: Option<String>) {
        if !self.cancelled.swap(true, Ordering::SeqCst) {
            *self.cancel_time.write().unwrap() = Some(Instant::now());
            *self.reason.write().unwrap() = reason;

            // Execute callbacks
            let callbacks = self.callbacks.read().unwrap();
            for callback in callbacks.iter() {
                callback();
            }
        }
    }

    /// Get the cancellation reason if available
    pub fn get_reason(&self) -> Option<String> {
        self.reason.read().unwrap().clone()
    }

    /// Get the time when cancellation was requested
    pub fn cancel_time(&self) -> Option<Instant> {
        *self.cancel_time.read().unwrap()
    }

    /// Register a callback to be executed when cancelled
    pub fn on_cancel<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.callbacks.write().unwrap().push(Box::new(callback));
    }

    /// Check for cancellation and return an error if cancelled
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            let reason = self
                .get_reason()
                .unwrap_or_else(|| "No reason provided".to_string());
            Err(anyhow!("Query cancelled: {}", reason))
        } else {
            Ok(())
        }
    }

    /// Create a child token that inherits parent cancellation
    pub fn child(&self) -> Self {
        let child = Self::new();
        let parent_cancelled = self.cancelled.clone();
        let child_cancelled = child.cancelled.clone();

        // Check parent periodically
        self.on_cancel(move || {
            if parent_cancelled.load(Ordering::Relaxed) {
                child_cancelled.store(true, Ordering::Relaxed);
            }
        });

        child
    }
}

impl std::fmt::Debug for QueryCancellationToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryCancellationToken")
            .field("cancelled", &self.is_cancelled())
            .field("reason", &self.get_reason())
            .finish()
    }
}

/// Advanced timeout management for queries
///
/// Provides configurable soft and hard timeouts with warning callbacks.
pub struct QueryTimeoutManager {
    soft_timeout: RwLock<Duration>,
    hard_timeout: RwLock<Duration>,
    warning_thresholds: RwLock<Vec<f64>>,
    active_queries: RwLock<HashMap<u64, QueryTimeoutState>>,
    next_query_id: AtomicU64,
    timeout_action: RwLock<TimeoutAction>,
}

/// State tracking for an active query's timeout
#[derive(Debug, Clone)]
pub struct QueryTimeoutState {
    pub query_id: u64,
    pub start_time: Instant,
    pub query_snippet: String,
    pub soft_timeout_triggered: bool,
    pub warnings_triggered: Vec<f64>,
}

/// Action to take when timeout is exceeded
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutAction {
    /// Log warning but allow query to continue
    Warn,
    /// Cancel the query immediately
    Cancel,
    /// Throttle resources but continue
    Throttle,
}

impl Default for QueryTimeoutManager {
    fn default() -> Self {
        Self {
            soft_timeout: RwLock::new(Duration::from_secs(30)),
            hard_timeout: RwLock::new(Duration::from_secs(300)),
            warning_thresholds: RwLock::new(vec![0.5, 0.75, 0.9]),
            active_queries: RwLock::new(HashMap::new()),
            next_query_id: AtomicU64::new(1),
            timeout_action: RwLock::new(TimeoutAction::Cancel),
        }
    }
}

impl QueryTimeoutManager {
    pub fn new(soft_timeout: Duration, hard_timeout: Duration) -> Self {
        Self {
            soft_timeout: RwLock::new(soft_timeout),
            hard_timeout: RwLock::new(hard_timeout),
            ..Default::default()
        }
    }

    /// Start tracking a query, returns a query ID
    pub fn start_query(&self, query_snippet: &str) -> u64 {
        let query_id = self.next_query_id.fetch_add(1, Ordering::Relaxed);
        let state = QueryTimeoutState {
            query_id,
            start_time: Instant::now(),
            query_snippet: query_snippet.chars().take(100).collect(),
            soft_timeout_triggered: false,
            warnings_triggered: Vec::new(),
        };

        self.active_queries.write().unwrap().insert(query_id, state);
        query_id
    }

    /// End tracking for a query
    pub fn end_query(&self, query_id: u64) -> Option<Duration> {
        self.active_queries
            .write()
            .unwrap()
            .remove(&query_id)
            .map(|state| state.start_time.elapsed())
    }

    /// Check if a query has exceeded its timeout
    pub fn check_timeout(&self, query_id: u64) -> TimeoutCheckResult {
        let hard = *self.hard_timeout.read().unwrap();
        let soft = *self.soft_timeout.read().unwrap();
        let thresholds = self.warning_thresholds.read().unwrap().clone();

        let mut queries = self.active_queries.write().unwrap();

        if let Some(state) = queries.get_mut(&query_id) {
            let elapsed = state.start_time.elapsed();

            // Check hard timeout
            if elapsed > hard {
                return TimeoutCheckResult::HardTimeout {
                    elapsed,
                    limit: hard,
                };
            }

            // Check soft timeout
            if elapsed > soft && !state.soft_timeout_triggered {
                state.soft_timeout_triggered = true;
                return TimeoutCheckResult::SoftTimeout {
                    elapsed,
                    limit: soft,
                };
            }

            // Check warning thresholds
            let progress = elapsed.as_secs_f64() / hard.as_secs_f64();
            for threshold in thresholds {
                if progress >= threshold && !state.warnings_triggered.contains(&threshold) {
                    state.warnings_triggered.push(threshold);
                    return TimeoutCheckResult::Warning {
                        elapsed,
                        threshold,
                        remaining: hard.saturating_sub(elapsed),
                    };
                }
            }

            TimeoutCheckResult::Ok { elapsed }
        } else {
            TimeoutCheckResult::QueryNotFound
        }
    }

    /// Get remaining time for a query
    pub fn remaining_time(&self, query_id: u64) -> Option<Duration> {
        let hard = *self.hard_timeout.read().unwrap();

        self.active_queries
            .read()
            .unwrap()
            .get(&query_id)
            .map(|state| hard.saturating_sub(state.start_time.elapsed()))
    }

    /// Set timeouts
    pub fn set_soft_timeout(&self, timeout: Duration) {
        *self.soft_timeout.write().unwrap() = timeout;
    }

    pub fn set_hard_timeout(&self, timeout: Duration) {
        *self.hard_timeout.write().unwrap() = timeout;
    }

    pub fn set_warning_thresholds(&self, thresholds: Vec<f64>) {
        *self.warning_thresholds.write().unwrap() = thresholds;
    }

    pub fn set_timeout_action(&self, action: TimeoutAction) {
        *self.timeout_action.write().unwrap() = action;
    }

    /// Get count of active queries
    pub fn active_query_count(&self) -> usize {
        self.active_queries.read().unwrap().len()
    }

    /// Get all active query states
    pub fn get_active_queries(&self) -> Vec<QueryTimeoutState> {
        self.active_queries
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }
}

/// Result of a timeout check
#[derive(Debug, Clone)]
pub enum TimeoutCheckResult {
    /// Query is within timeout limits
    Ok { elapsed: Duration },
    /// Soft timeout exceeded (warning level)
    SoftTimeout { elapsed: Duration, limit: Duration },
    /// Hard timeout exceeded (error level)
    HardTimeout { elapsed: Duration, limit: Duration },
    /// Warning threshold reached
    Warning {
        elapsed: Duration,
        threshold: f64,
        remaining: Duration,
    },
    /// Query not found in tracker
    QueryNotFound,
}

/// Memory usage tracker for queries
///
/// Tracks memory allocation and provides pressure-based throttling.
pub struct QueryMemoryTracker {
    current_usage: AtomicUsize,
    peak_usage: AtomicUsize,
    memory_limit: AtomicUsize,
    per_query_limit: AtomicUsize,
    query_allocations: RwLock<HashMap<u64, usize>>,
    pressure_threshold: RwLock<f64>,
}

impl Default for QueryMemoryTracker {
    fn default() -> Self {
        Self {
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
            memory_limit: AtomicUsize::new(1 << 30), // 1GB default
            per_query_limit: AtomicUsize::new(256 << 20), // 256MB default
            query_allocations: RwLock::new(HashMap::new()),
            pressure_threshold: RwLock::new(0.8),
        }
    }
}

impl QueryMemoryTracker {
    pub fn new(memory_limit: usize, per_query_limit: usize) -> Self {
        Self {
            memory_limit: AtomicUsize::new(memory_limit),
            per_query_limit: AtomicUsize::new(per_query_limit),
            ..Default::default()
        }
    }

    /// Allocate memory for a query
    pub fn allocate(&self, query_id: u64, bytes: usize) -> Result<()> {
        let per_query_limit = self.per_query_limit.load(Ordering::Relaxed);
        let memory_limit = self.memory_limit.load(Ordering::Relaxed);

        let mut allocations = self.query_allocations.write().unwrap();
        let current_query_usage = allocations.get(&query_id).copied().unwrap_or(0);

        // Check per-query limit
        if current_query_usage + bytes > per_query_limit {
            return Err(anyhow!(
                "Query {} memory allocation {} would exceed per-query limit of {}",
                query_id,
                current_query_usage + bytes,
                per_query_limit
            ));
        }

        // Check global limit
        let current = self.current_usage.load(Ordering::Relaxed);
        if current + bytes > memory_limit {
            return Err(anyhow!(
                "Global memory limit {} would be exceeded (current: {}, requested: {})",
                memory_limit,
                current,
                bytes
            ));
        }

        // Perform allocation
        *allocations.entry(query_id).or_insert(0) += bytes;
        let new_usage = self.current_usage.fetch_add(bytes, Ordering::SeqCst) + bytes;

        // Update peak
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while new_usage > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                new_usage,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }

        Ok(())
    }

    /// Deallocate memory for a query
    pub fn deallocate(&self, query_id: u64, bytes: usize) {
        let mut allocations = self.query_allocations.write().unwrap();

        if let Some(query_usage) = allocations.get_mut(&query_id) {
            let to_free = bytes.min(*query_usage);
            *query_usage -= to_free;
            self.current_usage.fetch_sub(to_free, Ordering::SeqCst);

            if *query_usage == 0 {
                allocations.remove(&query_id);
            }
        }
    }

    /// Free all memory for a completed query
    pub fn free_query(&self, query_id: u64) -> usize {
        let mut allocations = self.query_allocations.write().unwrap();

        if let Some(freed) = allocations.remove(&query_id) {
            self.current_usage.fetch_sub(freed, Ordering::SeqCst);
            freed
        } else {
            0
        }
    }

    /// Check if under memory pressure
    pub fn is_under_pressure(&self) -> bool {
        let current = self.current_usage.load(Ordering::Relaxed);
        let limit = self.memory_limit.load(Ordering::Relaxed);
        let threshold = *self.pressure_threshold.read().unwrap();

        (current as f64 / limit as f64) > threshold
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage.load(Ordering::Relaxed)
    }

    /// Get usage for a specific query
    pub fn query_usage(&self, query_id: u64) -> usize {
        self.query_allocations
            .read()
            .unwrap()
            .get(&query_id)
            .copied()
            .unwrap_or(0)
    }

    /// Get memory pressure as a percentage
    pub fn pressure_percentage(&self) -> f64 {
        let current = self.current_usage.load(Ordering::Relaxed);
        let limit = self.memory_limit.load(Ordering::Relaxed);

        if limit == 0 {
            0.0
        } else {
            (current as f64 / limit as f64) * 100.0
        }
    }

    /// Set memory limits
    pub fn set_memory_limit(&self, limit: usize) {
        self.memory_limit.store(limit, Ordering::Relaxed);
    }

    pub fn set_per_query_limit(&self, limit: usize) {
        self.per_query_limit.store(limit, Ordering::Relaxed);
    }

    pub fn set_pressure_threshold(&self, threshold: f64) {
        *self.pressure_threshold.write().unwrap() = threshold.clamp(0.0, 1.0);
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.peak_usage.store(
            self.current_usage.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    /// Get detailed memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage: self.current_usage.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            memory_limit: self.memory_limit.load(Ordering::Relaxed),
            active_queries: self.query_allocations.read().unwrap().len(),
            pressure_percentage: self.pressure_percentage(),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub memory_limit: usize,
    pub active_queries: usize,
    pub pressure_percentage: f64,
}

// =============================================================================
// Query Session Management - Unified Interface
// =============================================================================

/// Unified query session that integrates cancellation, timeout, and memory tracking
///
/// This provides a single entry point for managing query lifecycle with all
/// production features enabled.
pub struct QuerySession {
    pub session_id: u64,
    pub query_id: u64,
    pub cancellation_token: QueryCancellationToken,
    pub start_time: Instant,
    query: String,
    user_id: Option<String>,
    metadata: HashMap<String, String>,
}

impl QuerySession {
    /// Create a new query session
    pub fn new(session_id: u64, query_id: u64, query: String) -> Self {
        Self {
            session_id,
            query_id,
            cancellation_token: QueryCancellationToken::new(),
            start_time: Instant::now(),
            query,
            user_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Set user ID for this session
    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Add metadata to this session
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get elapsed time for this session
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get query snippet (first 100 chars)
    pub fn query_snippet(&self) -> String {
        self.query.chars().take(100).collect()
    }

    /// Cancel this session
    pub fn cancel(&self, reason: Option<String>) {
        self.cancellation_token.cancel(reason);
    }

    /// Check if session is cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token.is_cancelled()
    }
}

/// Query session manager for coordinating multiple concurrent sessions
pub struct QuerySessionManager {
    sessions: RwLock<HashMap<u64, Arc<QuerySession>>>,
    timeout_manager: QueryTimeoutManager,
    memory_tracker: QueryMemoryTracker,
    rate_limiter: QueryRateLimiter,
    audit_trail: QueryAuditTrail,
    next_session_id: AtomicU64,
}

impl Default for QuerySessionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl QuerySessionManager {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            timeout_manager: QueryTimeoutManager::default(),
            memory_tracker: QueryMemoryTracker::default(),
            rate_limiter: QueryRateLimiter::default(),
            audit_trail: QueryAuditTrail::new(1000),
            next_session_id: AtomicU64::new(1),
        }
    }

    /// Start a new query session
    pub fn start_session(&self, query: &str, user_id: Option<&str>) -> Result<Arc<QuerySession>> {
        // Check rate limit
        let user_key = user_id.unwrap_or("anonymous");
        if !self.rate_limiter.check_rate_limit(user_key) {
            return Err(anyhow!("Rate limit exceeded for user: {}", user_key));
        }

        // Check memory pressure
        if self.memory_tracker.is_under_pressure() {
            return Err(anyhow!("Server under memory pressure, try again later"));
        }

        let session_id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        let query_id = self.timeout_manager.start_query(query);

        let mut session = QuerySession::new(session_id, query_id, query.to_string());
        if let Some(uid) = user_id {
            session = session.with_user(uid.to_string());
        }

        let session = Arc::new(session);
        self.sessions
            .write()
            .unwrap()
            .insert(session_id, session.clone());

        // Log audit event
        self.audit_trail.log(QueryAuditEvent {
            timestamp: SystemTime::now(),
            session_id,
            query_id,
            event_type: AuditEventType::QueryStarted,
            user_id: user_id.map(|s| s.to_string()),
            query_snippet: query.chars().take(100).collect(),
            duration: None,
            result_count: None,
            error: None,
        });

        Ok(session)
    }

    /// Complete a query session
    pub fn complete_session(&self, session_id: u64, result_count: usize) -> Result<Duration> {
        let session = self.sessions.write().unwrap().remove(&session_id);

        if let Some(session) = session {
            let duration = self.timeout_manager.end_query(session.query_id);
            self.memory_tracker.free_query(session.query_id);

            // Log audit event
            self.audit_trail.log(QueryAuditEvent {
                timestamp: SystemTime::now(),
                session_id,
                query_id: session.query_id,
                event_type: AuditEventType::QueryCompleted,
                user_id: session.user_id.clone(),
                query_snippet: session.query_snippet(),
                duration,
                result_count: Some(result_count),
                error: None,
            });

            Ok(duration.unwrap_or_else(|| session.elapsed()))
        } else {
            Err(anyhow!("Session {} not found", session_id))
        }
    }

    /// Fail a query session
    pub fn fail_session(&self, session_id: u64, error: &str) -> Result<()> {
        let session = self.sessions.write().unwrap().remove(&session_id);

        if let Some(session) = session {
            let duration = self.timeout_manager.end_query(session.query_id);
            self.memory_tracker.free_query(session.query_id);

            // Log audit event
            self.audit_trail.log(QueryAuditEvent {
                timestamp: SystemTime::now(),
                session_id,
                query_id: session.query_id,
                event_type: AuditEventType::QueryFailed,
                user_id: session.user_id.clone(),
                query_snippet: session.query_snippet(),
                duration,
                result_count: None,
                error: Some(error.to_string()),
            });

            Ok(())
        } else {
            Err(anyhow!("Session {} not found", session_id))
        }
    }

    /// Check session timeout
    pub fn check_timeout(&self, session_id: u64) -> Result<TimeoutCheckResult> {
        let sessions = self.sessions.read().unwrap();
        if let Some(session) = sessions.get(&session_id) {
            Ok(self.timeout_manager.check_timeout(session.query_id))
        } else {
            Err(anyhow!("Session {} not found", session_id))
        }
    }

    /// Allocate memory for a session
    pub fn allocate_memory(&self, session_id: u64, bytes: usize) -> Result<()> {
        let sessions = self.sessions.read().unwrap();
        if let Some(session) = sessions.get(&session_id) {
            self.memory_tracker.allocate(session.query_id, bytes)
        } else {
            Err(anyhow!("Session {} not found", session_id))
        }
    }

    /// Get session by ID
    pub fn get_session(&self, session_id: u64) -> Option<Arc<QuerySession>> {
        self.sessions.read().unwrap().get(&session_id).cloned()
    }

    /// Get active session count
    pub fn active_session_count(&self) -> usize {
        self.sessions.read().unwrap().len()
    }

    /// Get memory stats
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory_tracker.get_stats()
    }

    /// Get audit events
    pub fn get_audit_events(&self, limit: usize) -> Vec<QueryAuditEvent> {
        self.audit_trail.get_recent(limit)
    }

    /// Configure rate limiter
    pub fn configure_rate_limit(&self, requests_per_second: u32, burst_size: u32) {
        self.rate_limiter.configure(requests_per_second, burst_size);
    }

    /// Configure timeouts
    pub fn configure_timeouts(&self, soft: Duration, hard: Duration) {
        self.timeout_manager.set_soft_timeout(soft);
        self.timeout_manager.set_hard_timeout(hard);
    }

    /// Configure memory limits
    pub fn configure_memory(&self, global_limit: usize, per_query_limit: usize) {
        self.memory_tracker.set_memory_limit(global_limit);
        self.memory_tracker.set_per_query_limit(per_query_limit);
    }
}

// =============================================================================
// Query Rate Limiting
// =============================================================================

/// Token bucket rate limiter for query requests
pub struct QueryRateLimiter {
    buckets: RwLock<HashMap<String, TokenBucket>>,
    requests_per_second: AtomicU32,
    burst_size: AtomicU32,
    enabled: AtomicBool,
}

struct TokenBucket {
    tokens: f64,
    last_update: Instant,
}

impl Default for QueryRateLimiter {
    fn default() -> Self {
        Self {
            buckets: RwLock::new(HashMap::new()),
            requests_per_second: AtomicU32::new(100),
            burst_size: AtomicU32::new(200),
            enabled: AtomicBool::new(true),
        }
    }
}

impl QueryRateLimiter {
    /// Check if request is allowed for the given key
    pub fn check_rate_limit(&self, key: &str) -> bool {
        if !self.enabled.load(Ordering::Relaxed) {
            return true;
        }

        let rate = self.requests_per_second.load(Ordering::Relaxed) as f64;
        let burst = self.burst_size.load(Ordering::Relaxed) as f64;

        let mut buckets = self.buckets.write().unwrap();
        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| TokenBucket {
                tokens: burst,
                last_update: Instant::now(),
            });

        // Refill tokens based on elapsed time
        let now = Instant::now();
        let elapsed = now.duration_since(bucket.last_update).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * rate).min(burst);
        bucket.last_update = now;

        // Check if we have tokens
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Configure rate limit parameters
    pub fn configure(&self, requests_per_second: u32, burst_size: u32) {
        self.requests_per_second
            .store(requests_per_second, Ordering::Relaxed);
        self.burst_size.store(burst_size, Ordering::Relaxed);
    }

    /// Enable/disable rate limiting
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get current rate limit stats for a key
    pub fn get_stats(&self, key: &str) -> Option<f64> {
        self.buckets.read().unwrap().get(key).map(|b| b.tokens)
    }

    /// Clear rate limit state for all keys
    pub fn clear(&self) {
        self.buckets.write().unwrap().clear();
    }
}

// =============================================================================
// Query Audit Trail
// =============================================================================

/// Audit trail for query execution compliance and debugging
pub struct QueryAuditTrail {
    events: RwLock<Vec<QueryAuditEvent>>,
    max_events: usize,
    enabled: AtomicBool,
}

/// Audit event for query execution
#[derive(Debug, Clone)]
pub struct QueryAuditEvent {
    pub timestamp: SystemTime,
    pub session_id: u64,
    pub query_id: u64,
    pub event_type: AuditEventType,
    pub user_id: Option<String>,
    pub query_snippet: String,
    pub duration: Option<Duration>,
    pub result_count: Option<usize>,
    pub error: Option<String>,
}

/// Types of audit events
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditEventType {
    QueryStarted,
    QueryCompleted,
    QueryFailed,
    QueryCancelled,
    TimeoutWarning,
    MemoryWarning,
    RateLimitExceeded,
}

impl QueryAuditTrail {
    pub fn new(max_events: usize) -> Self {
        Self {
            events: RwLock::new(Vec::with_capacity(max_events)),
            max_events,
            enabled: AtomicBool::new(true),
        }
    }

    /// Log an audit event
    pub fn log(&self, event: QueryAuditEvent) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let mut events = self.events.write().unwrap();

        // Circular buffer behavior
        if events.len() >= self.max_events {
            events.remove(0);
        }

        events.push(event);
    }

    /// Get recent events
    pub fn get_recent(&self, limit: usize) -> Vec<QueryAuditEvent> {
        let events = self.events.read().unwrap();
        let start = if events.len() > limit {
            events.len() - limit
        } else {
            0
        };
        events[start..].to_vec()
    }

    /// Get events for a specific user
    pub fn get_by_user(&self, user_id: &str, limit: usize) -> Vec<QueryAuditEvent> {
        self.events
            .read()
            .unwrap()
            .iter()
            .filter(|e| e.user_id.as_deref() == Some(user_id))
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get events by type
    pub fn get_by_type(&self, event_type: AuditEventType, limit: usize) -> Vec<QueryAuditEvent> {
        self.events
            .read()
            .unwrap()
            .iter()
            .filter(|e| e.event_type == event_type)
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get failed queries
    pub fn get_failures(&self, limit: usize) -> Vec<QueryAuditEvent> {
        self.get_by_type(AuditEventType::QueryFailed, limit)
    }

    /// Get total event count
    pub fn event_count(&self) -> usize {
        self.events.read().unwrap().len()
    }

    /// Clear all events
    pub fn clear(&self) {
        self.events.write().unwrap().clear();
    }

    /// Enable/disable audit logging
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

// =============================================================================
// Beta.2 Enhancement: Query Priority System
// =============================================================================

/// Query priority levels for resource allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum QueryPriority {
    /// Critical queries that must complete quickly
    Critical = 4,
    /// High priority queries (e.g., user-facing)
    High = 3,
    /// Normal priority queries
    #[default]
    Normal = 2,
    /// Low priority queries (e.g., background tasks)
    Low = 1,
    /// Batch queries that can be delayed
    Batch = 0,
}

/// Query with priority and metadata
#[derive(Debug, Clone)]
pub struct PrioritizedQuery {
    pub query_id: u64,
    pub priority: QueryPriority,
    pub submitted_at: SystemTime,
    pub query_text: String,
    pub user_id: Option<String>,
    pub estimated_cost: Option<f64>,
}

/// Priority-based query scheduler
pub struct QueryPriorityScheduler {
    queues: Arc<RwLock<HashMap<QueryPriority, Vec<PrioritizedQuery>>>>,
    active_queries: Arc<RwLock<HashMap<u64, QueryPriority>>>,
    next_query_id: Arc<AtomicU64>,
    config: PrioritySchedulerConfig,
}

/// Configuration for priority scheduler
#[derive(Debug, Clone)]
pub struct PrioritySchedulerConfig {
    /// Maximum queries per priority level
    pub max_per_priority: usize,
    /// Maximum total queued queries
    pub max_total_queued: usize,
    /// Maximum concurrent queries per priority
    pub max_concurrent_per_priority: HashMap<QueryPriority, usize>,
    /// Enable priority boosting for aged queries
    pub enable_aging: bool,
    /// Age threshold for priority boost (seconds)
    pub aging_threshold: Duration,
}

impl Default for PrioritySchedulerConfig {
    fn default() -> Self {
        let mut max_concurrent = HashMap::new();
        max_concurrent.insert(QueryPriority::Critical, 10);
        max_concurrent.insert(QueryPriority::High, 8);
        max_concurrent.insert(QueryPriority::Normal, 5);
        max_concurrent.insert(QueryPriority::Low, 3);
        max_concurrent.insert(QueryPriority::Batch, 2);

        Self {
            max_per_priority: 100,
            max_total_queued: 500,
            max_concurrent_per_priority: max_concurrent,
            enable_aging: true,
            aging_threshold: Duration::from_secs(30),
        }
    }
}

impl QueryPriorityScheduler {
    pub fn new(config: PrioritySchedulerConfig) -> Self {
        let mut queues = HashMap::new();
        for priority in [
            QueryPriority::Critical,
            QueryPriority::High,
            QueryPriority::Normal,
            QueryPriority::Low,
            QueryPriority::Batch,
        ] {
            queues.insert(priority, Vec::new());
        }

        Self {
            queues: Arc::new(RwLock::new(queues)),
            active_queries: Arc::new(RwLock::new(HashMap::new())),
            next_query_id: Arc::new(AtomicU64::new(1)),
            config,
        }
    }

    /// Submit a query to the scheduler
    pub fn submit_query(
        &self,
        query: String,
        priority: QueryPriority,
        user_id: Option<String>,
        estimated_cost: Option<f64>,
    ) -> Result<u64> {
        let query_id = self.next_query_id.fetch_add(1, Ordering::SeqCst);

        let prioritized = PrioritizedQuery {
            query_id,
            priority,
            submitted_at: SystemTime::now(),
            query_text: query,
            user_id,
            estimated_cost,
        };

        let mut queues = self.queues.write().unwrap();
        let total_queued: usize = queues.values().map(|q| q.len()).sum();

        if total_queued >= self.config.max_total_queued {
            return Err(anyhow!("Query queue is full"));
        }

        let priority_queue = queues.get_mut(&priority).unwrap();
        if priority_queue.len() >= self.config.max_per_priority {
            return Err(anyhow!("Priority queue {:?} is full", priority));
        }

        priority_queue.push(prioritized);
        Ok(query_id)
    }

    /// Get next query to execute based on priority
    pub fn next_query(&self) -> Option<PrioritizedQuery> {
        let mut queues = self.queues.write().unwrap();

        // Process aging if enabled
        if self.config.enable_aging {
            self.process_aging(&mut queues);
        }

        // Try each priority level from highest to lowest
        for priority in [
            QueryPriority::Critical,
            QueryPriority::High,
            QueryPriority::Normal,
            QueryPriority::Low,
            QueryPriority::Batch,
        ] {
            let active = self.active_queries.read().unwrap();
            let concurrent_count = active.values().filter(|&&p| p == priority).count();

            if let Some(&max_concurrent) = self.config.max_concurrent_per_priority.get(&priority) {
                if concurrent_count >= max_concurrent {
                    continue;
                }
            }

            if let Some(queue) = queues.get_mut(&priority) {
                if !queue.is_empty() {
                    let query = queue.remove(0);
                    drop(active); // Release read lock before acquiring write lock
                    self.active_queries
                        .write()
                        .unwrap()
                        .insert(query.query_id, priority);
                    return Some(query);
                }
            }
        }

        None
    }

    /// Process query aging - boost priority of old queries
    fn process_aging(&self, queues: &mut HashMap<QueryPriority, Vec<PrioritizedQuery>>) {
        let now = SystemTime::now();
        let threshold = self.config.aging_threshold;

        for priority in [
            QueryPriority::Batch,
            QueryPriority::Low,
            QueryPriority::Normal,
            QueryPriority::High,
        ] {
            if let Some(queue) = queues.get_mut(&priority) {
                let mut to_boost = Vec::new();

                queue.retain(|query| {
                    if let Ok(age) = now.duration_since(query.submitted_at) {
                        if age > threshold {
                            to_boost.push(query.clone());
                            return false;
                        }
                    }
                    true
                });

                // Boost to next priority level
                if !to_boost.is_empty() {
                    let next_priority = match priority {
                        QueryPriority::Batch => QueryPriority::Low,
                        QueryPriority::Low => QueryPriority::Normal,
                        QueryPriority::Normal => QueryPriority::High,
                        QueryPriority::High => QueryPriority::Critical,
                        QueryPriority::Critical => QueryPriority::Critical,
                    };

                    if let Some(next_queue) = queues.get_mut(&next_priority) {
                        for mut query in to_boost {
                            query.priority = next_priority;
                            next_queue.push(query);
                        }
                    }
                }
            }
        }
    }

    /// Mark query as completed
    pub fn complete_query(&self, query_id: u64) {
        self.active_queries.write().unwrap().remove(&query_id);
    }

    /// Cancel a queued query
    pub fn cancel_query(&self, query_id: u64) -> bool {
        let mut queues = self.queues.write().unwrap();
        for queue in queues.values_mut() {
            if let Some(pos) = queue.iter().position(|q| q.query_id == query_id) {
                queue.remove(pos);
                return true;
            }
        }
        false
    }

    /// Get queue statistics
    pub fn get_stats(&self) -> PrioritySchedulerStats {
        let queues = self.queues.read().unwrap();
        let active = self.active_queries.read().unwrap();

        let mut queued_per_priority = HashMap::new();
        for (priority, queue) in queues.iter() {
            queued_per_priority.insert(*priority, queue.len());
        }

        let mut active_per_priority = HashMap::new();
        for priority in active.values() {
            *active_per_priority.entry(*priority).or_insert(0) += 1;
        }

        PrioritySchedulerStats {
            total_queued: queues.values().map(|q| q.len()).sum(),
            total_active: active.len(),
            queued_per_priority,
            active_per_priority,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrioritySchedulerStats {
    pub total_queued: usize,
    pub total_active: usize,
    pub queued_per_priority: HashMap<QueryPriority, usize>,
    pub active_per_priority: HashMap<QueryPriority, usize>,
}

// =============================================================================
// Beta.2 Enhancement: Query Cost Estimator
// =============================================================================

/// Estimates query execution cost for resource planning
pub struct QueryCostEstimator {
    stats_collector: Arc<RwLock<CostStatistics>>,
    config: CostEstimatorConfig,
}

#[derive(Debug, Clone)]
pub struct CostEstimatorConfig {
    /// Cost weight for number of triple patterns
    pub pattern_weight: f64,
    /// Cost weight for number of joins
    pub join_weight: f64,
    /// Cost weight for filter complexity
    pub filter_weight: f64,
    /// Cost weight for aggregations
    pub aggregate_weight: f64,
    /// Cost weight for property paths
    pub path_weight: f64,
    /// Enable machine learning cost prediction
    pub enable_ml_prediction: bool,
}

impl Default for CostEstimatorConfig {
    fn default() -> Self {
        Self {
            pattern_weight: 10.0,
            join_weight: 50.0,
            filter_weight: 20.0,
            aggregate_weight: 30.0,
            path_weight: 100.0,
            enable_ml_prediction: false,
        }
    }
}

#[derive(Debug, Clone)]
struct CostStatistics {
    historical_costs: Vec<(QueryFeatures, f64)>,
    max_samples: usize,
}

impl CostStatistics {
    fn new(max_samples: usize) -> Self {
        Self {
            historical_costs: Vec::new(),
            max_samples,
        }
    }

    fn add_sample(&mut self, features: QueryFeatures, actual_cost: f64) {
        self.historical_costs.push((features, actual_cost));
        if self.historical_costs.len() > self.max_samples {
            self.historical_costs.remove(0);
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct QueryFeatures {
    pub pattern_count: usize,
    pub join_count: usize,
    pub filter_count: usize,
    pub aggregate_count: usize,
    pub path_count: usize,
    pub optional_count: usize,
    pub union_count: usize,
    pub distinct: bool,
    pub order_by: bool,
    pub group_by: bool,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct QueryCostEstimate {
    pub estimated_cost: f64,
    pub estimated_duration_ms: f64,
    pub estimated_memory_mb: f64,
    pub complexity_score: f64,
    pub recommendation: CostRecommendation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CostRecommendation {
    /// Query is lightweight and should execute quickly
    Lightweight,
    /// Query is moderate, normal execution
    Moderate,
    /// Query is expensive, consider optimization
    Expensive,
    /// Query is very expensive, strongly recommend optimization
    VeryExpensive,
}

impl QueryCostEstimator {
    pub fn new(config: CostEstimatorConfig) -> Self {
        Self {
            stats_collector: Arc::new(RwLock::new(CostStatistics::new(1000))),
            config,
        }
    }

    /// Estimate cost for query features
    pub fn estimate_cost(&self, features: &QueryFeatures) -> QueryCostEstimate {
        let mut cost = 0.0;

        // Pattern-based cost
        cost += features.pattern_count as f64 * self.config.pattern_weight;

        // Join-based cost (exponential)
        cost += (features.join_count as f64).powi(2) * self.config.join_weight;

        // Filter-based cost
        cost += features.filter_count as f64 * self.config.filter_weight;

        // Aggregate-based cost
        cost += features.aggregate_count as f64 * self.config.aggregate_weight;

        // Property path cost (expensive)
        cost += features.path_count as f64 * self.config.path_weight;

        // OPTIONAL multiplier
        if features.optional_count > 0 {
            cost *= 1.0 + (features.optional_count as f64 * 0.3);
        }

        // UNION multiplier
        if features.union_count > 0 {
            cost *= 1.0 + (features.union_count as f64 * 0.5);
        }

        // DISTINCT overhead
        if features.distinct {
            cost *= 1.5;
        }

        // ORDER BY overhead
        if features.order_by {
            cost *= 1.3;
        }

        // GROUP BY overhead
        if features.group_by {
            cost *= 1.4;
        }

        // LIMIT benefit
        if let Some(limit) = features.limit {
            if limit < 100 {
                cost *= 0.5;
            } else if limit < 1000 {
                cost *= 0.7;
            }
        }

        let complexity_score = cost / 100.0;
        let estimated_duration_ms = cost * 0.1; // Rough estimate
        let estimated_memory_mb = cost * 0.01; // Rough estimate

        let recommendation = if cost < 100.0 {
            CostRecommendation::Lightweight
        } else if cost < 500.0 {
            CostRecommendation::Moderate
        } else if cost < 2000.0 {
            CostRecommendation::Expensive
        } else {
            CostRecommendation::VeryExpensive
        };

        QueryCostEstimate {
            estimated_cost: cost,
            estimated_duration_ms,
            estimated_memory_mb,
            complexity_score,
            recommendation,
        }
    }

    /// Record actual query cost for learning
    pub fn record_actual_cost(&self, features: QueryFeatures, actual_duration_ms: f64) {
        let mut stats = self.stats_collector.write().unwrap();
        stats.add_sample(features, actual_duration_ms);
    }

    /// Get historical statistics
    pub fn get_statistics(&self) -> CostEstimatorStatistics {
        let stats = self.stats_collector.read().unwrap();
        let sample_count = stats.historical_costs.len();

        if sample_count == 0 {
            return CostEstimatorStatistics {
                sample_count: 0,
                avg_cost: 0.0,
                min_cost: 0.0,
                max_cost: 0.0,
            };
        }

        let costs: Vec<f64> = stats.historical_costs.iter().map(|(_, c)| *c).collect();
        let avg_cost = costs.iter().sum::<f64>() / costs.len() as f64;
        let min_cost = costs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_cost = costs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        CostEstimatorStatistics {
            sample_count,
            avg_cost,
            min_cost,
            max_cost,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CostEstimatorStatistics {
    pub sample_count: usize,
    pub avg_cost: f64,
    pub min_cost: f64,
    pub max_cost: f64,
}

// =============================================================================
// Beta.2 Enhancement: Performance Baseline Tracker
// =============================================================================

/// Tracks performance baselines for detecting regressions
pub struct PerformanceBaselineTracker {
    baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
    config: BaselineTrackerConfig,
}

#[derive(Debug, Clone)]
pub struct BaselineTrackerConfig {
    /// Window size for rolling average (number of samples)
    pub window_size: usize,
    /// Threshold for regression detection (percentage)
    pub regression_threshold: f64,
    /// Minimum samples before baseline is established
    pub min_samples: usize,
    /// Enable automatic baseline updates
    pub auto_update_baseline: bool,
}

impl Default for BaselineTrackerConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            regression_threshold: 0.2, // 20% degradation
            min_samples: 10,
            auto_update_baseline: true,
        }
    }
}

#[derive(Debug, Clone)]
struct PerformanceBaseline {
    #[allow(dead_code)] // Used for debugging and future extensions
    query_pattern: String,
    samples: Vec<PerformanceSample>,
    baseline_duration_ms: f64,
    baseline_memory_mb: f64,
    last_updated: SystemTime,
}

#[derive(Debug, Clone)]
struct PerformanceSample {
    #[allow(dead_code)] // Used for debugging and future extensions
    timestamp: SystemTime,
    duration_ms: f64,
    memory_mb: f64,
    #[allow(dead_code)] // Used for debugging and future extensions
    result_count: usize,
}

impl PerformanceBaselineTracker {
    pub fn new(config: BaselineTrackerConfig) -> Self {
        Self {
            baselines: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Record a query execution for baseline tracking
    pub fn record_execution(
        &self,
        query_pattern: String,
        duration_ms: f64,
        memory_mb: f64,
        result_count: usize,
    ) {
        let mut baselines = self.baselines.write().unwrap();

        let baseline =
            baselines
                .entry(query_pattern.clone())
                .or_insert_with(|| PerformanceBaseline {
                    query_pattern,
                    samples: Vec::new(),
                    baseline_duration_ms: duration_ms,
                    baseline_memory_mb: memory_mb,
                    last_updated: SystemTime::now(),
                });

        baseline.samples.push(PerformanceSample {
            timestamp: SystemTime::now(),
            duration_ms,
            memory_mb,
            result_count,
        });

        // Keep only recent samples
        if baseline.samples.len() > self.config.window_size {
            baseline.samples.remove(0);
        }

        // Update baseline if auto-update is enabled
        if self.config.auto_update_baseline && baseline.samples.len() >= self.config.min_samples {
            let avg_duration: f64 = baseline.samples.iter().map(|s| s.duration_ms).sum::<f64>()
                / baseline.samples.len() as f64;
            let avg_memory: f64 = baseline.samples.iter().map(|s| s.memory_mb).sum::<f64>()
                / baseline.samples.len() as f64;

            baseline.baseline_duration_ms = avg_duration;
            baseline.baseline_memory_mb = avg_memory;
            baseline.last_updated = SystemTime::now();
        }
    }

    /// Check for performance regression
    pub fn check_regression(
        &self,
        query_pattern: &str,
        current_duration_ms: f64,
    ) -> Option<RegressionReport> {
        let baselines = self.baselines.read().unwrap();

        if let Some(baseline) = baselines.get(query_pattern) {
            if baseline.samples.len() < self.config.min_samples {
                return None;
            }

            let baseline_duration = baseline.baseline_duration_ms;
            let degradation = (current_duration_ms - baseline_duration) / baseline_duration;

            if degradation > self.config.regression_threshold {
                return Some(RegressionReport {
                    query_pattern: query_pattern.to_string(),
                    baseline_duration_ms: baseline_duration,
                    current_duration_ms,
                    degradation_percentage: degradation * 100.0,
                    sample_count: baseline.samples.len(),
                    severity: if degradation > 0.5 {
                        RegressionSeverity::Critical
                    } else if degradation > 0.3 {
                        RegressionSeverity::High
                    } else {
                        RegressionSeverity::Moderate
                    },
                });
            }
        }

        None
    }

    /// Get performance trend for a query pattern
    pub fn get_trend(&self, query_pattern: &str) -> Option<PerformanceTrend> {
        let baselines = self.baselines.read().unwrap();

        if let Some(baseline) = baselines.get(query_pattern) {
            if baseline.samples.is_empty() {
                return None;
            }

            let durations: Vec<f64> = baseline.samples.iter().map(|s| s.duration_ms).collect();
            let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
            let min_duration = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_duration = durations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Calculate variance
            let variance = durations
                .iter()
                .map(|d| (d - avg_duration).powi(2))
                .sum::<f64>()
                / durations.len() as f64;
            let std_dev = variance.sqrt();

            return Some(PerformanceTrend {
                query_pattern: query_pattern.to_string(),
                sample_count: baseline.samples.len(),
                avg_duration_ms: avg_duration,
                min_duration_ms: min_duration,
                max_duration_ms: max_duration,
                std_dev_ms: std_dev,
                baseline_duration_ms: baseline.baseline_duration_ms,
                last_updated: baseline.last_updated,
            });
        }

        None
    }

    /// Get all tracked patterns
    pub fn get_tracked_patterns(&self) -> Vec<String> {
        self.baselines.read().unwrap().keys().cloned().collect()
    }

    /// Clear baselines
    pub fn clear(&self) {
        self.baselines.write().unwrap().clear();
    }
}

#[derive(Debug, Clone)]
pub struct RegressionReport {
    pub query_pattern: String,
    pub baseline_duration_ms: f64,
    pub current_duration_ms: f64,
    pub degradation_percentage: f64,
    pub sample_count: usize,
    pub severity: RegressionSeverity,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegressionSeverity {
    Moderate,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub query_pattern: String,
    pub sample_count: usize,
    pub avg_duration_ms: f64,
    pub min_duration_ms: f64,
    pub max_duration_ms: f64,
    pub std_dev_ms: f64,
    pub baseline_duration_ms: f64,
    pub last_updated: SystemTime,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_token() {
        let token = QueryCancellationToken::new();

        assert!(!token.is_cancelled());
        assert!(token.check().is_ok());

        token.cancel(Some("User requested".to_string()));

        assert!(token.is_cancelled());
        assert!(token.check().is_err());
        assert_eq!(token.get_reason(), Some("User requested".to_string()));
        assert!(token.cancel_time().is_some());
    }

    #[test]
    fn test_timeout_manager() {
        let manager =
            QueryTimeoutManager::new(Duration::from_millis(100), Duration::from_millis(200));

        let query_id = manager.start_query("SELECT * WHERE { ?s ?p ?o }");

        // Initially OK
        match manager.check_timeout(query_id) {
            TimeoutCheckResult::Ok { .. } => {}
            _ => panic!("Expected Ok result"),
        }

        // Simulate time passing
        std::thread::sleep(Duration::from_millis(110));

        match manager.check_timeout(query_id) {
            TimeoutCheckResult::SoftTimeout { .. } => {}
            _ => panic!("Expected SoftTimeout result"),
        }

        let elapsed = manager.end_query(query_id);
        assert!(elapsed.is_some());
    }

    #[test]
    fn test_memory_tracker() {
        let tracker = QueryMemoryTracker::new(
            1024, // 1KB limit
            512,  // 512B per query
        );

        // Allocate within limits
        assert!(tracker.allocate(1, 256).is_ok());
        assert_eq!(tracker.query_usage(1), 256);
        assert_eq!(tracker.current_usage(), 256);

        // Allocate more
        assert!(tracker.allocate(1, 128).is_ok());
        assert_eq!(tracker.query_usage(1), 384);

        // Exceed per-query limit
        assert!(tracker.allocate(1, 256).is_err());

        // Allocate for another query
        assert!(tracker.allocate(2, 400).is_ok());
        assert_eq!(tracker.current_usage(), 784);

        // Free query
        let freed = tracker.free_query(1);
        assert_eq!(freed, 384);
        assert_eq!(tracker.current_usage(), 400);
    }

    #[test]
    fn test_memory_pressure() {
        let tracker = QueryMemoryTracker::new(1000, 500);
        tracker.set_pressure_threshold(0.8);

        assert!(!tracker.is_under_pressure());

        tracker.allocate(1, 400).unwrap();
        assert!(!tracker.is_under_pressure());

        tracker.allocate(2, 450).unwrap();
        assert!(tracker.is_under_pressure());

        assert!(tracker.pressure_percentage() > 80.0);
    }

    #[test]
    fn test_circuit_breaker_closed_to_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_requests: 2,
        };

        let breaker = QueryCircuitBreaker::new(config);

        assert!(breaker.is_request_allowed());

        // Record failures
        breaker.record_failure();
        assert!(breaker.is_request_allowed());

        breaker.record_failure();
        assert!(breaker.is_request_allowed());

        breaker.record_failure();
        assert!(!breaker.is_request_allowed()); // Should open
    }

    #[test]
    fn test_circuit_breaker_half_open_to_closed() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            half_open_max_requests: 3,
        };

        let breaker = QueryCircuitBreaker::new(config);

        // Open the circuit
        breaker.record_failure();
        breaker.record_failure();
        assert!(!breaker.is_request_allowed());

        // Try half-open
        breaker.try_half_open();
        assert!(breaker.is_request_allowed());

        // Record successes to close
        breaker.record_success();
        breaker.record_success();

        assert!(breaker.is_request_allowed());
        assert_eq!(breaker.state(), "Closed");
    }

    #[test]
    fn test_performance_monitor() {
        let monitor = SparqlPerformanceMonitor::new();

        monitor.record_query("SELECT", Duration::from_millis(100), 5, 1000);
        monitor.record_query("SELECT", Duration::from_millis(200), 10, 2000);
        monitor.record_query("SELECT", Duration::from_millis(150), 7, 1500);

        let stats = monitor.get_statistics("SELECT");
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.average_pattern_complexity, 7);
        assert_eq!(stats.average_result_size, 1500);

        let global = monitor.get_global_statistics();
        assert_eq!(global.total_queries, 3);
    }

    #[test]
    fn test_resource_quota() {
        let quota = QueryResourceQuota::new(1000, Duration::from_secs(10), 50);

        // Within limits
        assert!(quota.check_result_size(500).is_ok());
        assert!(quota.check_query_time(Duration::from_secs(5)).is_ok());
        assert!(quota.check_pattern_complexity(30).is_ok());

        // Exceeding limits
        assert!(quota.check_result_size(2000).is_err());
        assert!(quota.check_query_time(Duration::from_secs(20)).is_err());
        assert!(quota.check_pattern_complexity(100).is_err());

        // Disable enforcement
        quota.set_enforced(false);
        assert!(quota.check_result_size(2000).is_ok());
    }

    #[test]
    fn test_health_checks() {
        let health = QueryEngineHealth::default();

        health.perform_all_checks();

        assert_eq!(health.get_overall_status(), HealthStatus::Healthy);

        let checks = health.get_checks();
        assert_eq!(checks.len(), 3);
        assert!(checks.iter().all(|c| c.status == HealthStatus::Healthy));
    }

    #[test]
    fn test_production_error() {
        let error = SparqlProductionError::parse_error(
            "SELECT ?s WHERE { ?s ?p ?o".to_string(),
            "Missing closing brace".to_string(),
        );

        assert_eq!(error.severity, ErrorSeverity::Error);
        assert!(!error.retryable);
        assert_eq!(error.context.operation, "parse");
    }

    #[test]
    fn test_session_manager() {
        let manager = QuerySessionManager::new();

        // Start a session
        let session = manager
            .start_session("SELECT * WHERE { ?s ?p ?o }", Some("user1"))
            .unwrap();

        assert_eq!(manager.active_session_count(), 1);
        assert!(!session.is_cancelled());

        // Complete the session
        let duration = manager.complete_session(session.session_id, 100).unwrap();
        assert!(duration.as_nanos() > 0);
        assert_eq!(manager.active_session_count(), 0);

        // Check audit trail
        let events = manager.get_audit_events(10);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, AuditEventType::QueryStarted);
        assert_eq!(events[1].event_type, AuditEventType::QueryCompleted);
    }

    #[test]
    fn test_session_failure() {
        let manager = QuerySessionManager::new();

        let session = manager
            .start_session("SELECT * WHERE { ?s ?p ?o }", None)
            .unwrap();

        manager
            .fail_session(session.session_id, "Test error")
            .unwrap();

        let events = manager.get_audit_events(10);
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].event_type, AuditEventType::QueryFailed);
        assert_eq!(events[1].error, Some("Test error".to_string()));
    }

    #[test]
    fn test_rate_limiter() {
        let limiter = QueryRateLimiter::default();
        limiter.configure(10, 5); // 10 req/s, burst of 5

        // Should allow burst
        for _ in 0..5 {
            assert!(limiter.check_rate_limit("user1"));
        }

        // Should be rate limited after burst
        assert!(!limiter.check_rate_limit("user1"));

        // Different user should have separate bucket
        assert!(limiter.check_rate_limit("user2"));

        // Disable rate limiting
        limiter.set_enabled(false);
        assert!(limiter.check_rate_limit("user1"));
    }

    #[test]
    fn test_audit_trail() {
        let trail = QueryAuditTrail::new(100);

        // Log some events
        trail.log(QueryAuditEvent {
            timestamp: SystemTime::now(),
            session_id: 1,
            query_id: 1,
            event_type: AuditEventType::QueryStarted,
            user_id: Some("user1".to_string()),
            query_snippet: "SELECT *".to_string(),
            duration: None,
            result_count: None,
            error: None,
        });

        trail.log(QueryAuditEvent {
            timestamp: SystemTime::now(),
            session_id: 1,
            query_id: 1,
            event_type: AuditEventType::QueryCompleted,
            user_id: Some("user1".to_string()),
            query_snippet: "SELECT *".to_string(),
            duration: Some(Duration::from_millis(100)),
            result_count: Some(50),
            error: None,
        });

        assert_eq!(trail.event_count(), 2);

        // Get by user
        let user_events = trail.get_by_user("user1", 10);
        assert_eq!(user_events.len(), 2);

        // Get by type
        let completed = trail.get_by_type(AuditEventType::QueryCompleted, 10);
        assert_eq!(completed.len(), 1);

        // Clear and disable
        trail.clear();
        assert_eq!(trail.event_count(), 0);

        trail.set_enabled(false);
        trail.log(QueryAuditEvent {
            timestamp: SystemTime::now(),
            session_id: 2,
            query_id: 2,
            event_type: AuditEventType::QueryStarted,
            user_id: None,
            query_snippet: "ASK".to_string(),
            duration: None,
            result_count: None,
            error: None,
        });
        assert_eq!(trail.event_count(), 0); // Not logged when disabled
    }

    #[test]
    fn test_audit_trail_circular_buffer() {
        let trail = QueryAuditTrail::new(3);

        for i in 0..5 {
            trail.log(QueryAuditEvent {
                timestamp: SystemTime::now(),
                session_id: i,
                query_id: i,
                event_type: AuditEventType::QueryStarted,
                user_id: None,
                query_snippet: format!("Query {}", i),
                duration: None,
                result_count: None,
                error: None,
            });
        }

        assert_eq!(trail.event_count(), 3);

        let events = trail.get_recent(10);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].session_id, 2); // First two were evicted
        assert_eq!(events[1].session_id, 3);
        assert_eq!(events[2].session_id, 4);
    }

    #[test]
    fn test_session_memory_allocation() {
        let manager = QuerySessionManager::new();
        manager.configure_memory(1000, 500);

        let session = manager
            .start_session("SELECT * WHERE { ?s ?p ?o }", None)
            .unwrap();

        // Allocate memory
        assert!(manager.allocate_memory(session.session_id, 200).is_ok());
        assert!(manager.allocate_memory(session.session_id, 200).is_ok());

        // Exceed per-query limit
        assert!(manager.allocate_memory(session.session_id, 200).is_err());

        manager.complete_session(session.session_id, 0).unwrap();
    }

    // Beta.2 Enhancement Tests

    #[test]
    fn test_query_priority_scheduler_basic() {
        let scheduler = QueryPriorityScheduler::new(PrioritySchedulerConfig::default());

        // Submit queries with different priorities
        let critical_id = scheduler
            .submit_query(
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                QueryPriority::Critical,
                Some("user1".to_string()),
                Some(100.0),
            )
            .unwrap();

        let normal_id = scheduler
            .submit_query(
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                QueryPriority::Normal,
                Some("user2".to_string()),
                Some(50.0),
            )
            .unwrap();

        let batch_id = scheduler
            .submit_query(
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                QueryPriority::Batch,
                None,
                None,
            )
            .unwrap();

        // Stats should show 3 queued queries
        let stats = scheduler.get_stats();
        assert_eq!(stats.total_queued, 3);

        // Next query should be critical (highest priority)
        let next = scheduler.next_query().unwrap();
        assert_eq!(next.query_id, critical_id);
        assert_eq!(next.priority, QueryPriority::Critical);

        // Then normal
        let next = scheduler.next_query().unwrap();
        assert_eq!(next.query_id, normal_id);

        // Then batch
        let next = scheduler.next_query().unwrap();
        assert_eq!(next.query_id, batch_id);

        // Complete queries
        scheduler.complete_query(critical_id);
        scheduler.complete_query(normal_id);
        scheduler.complete_query(batch_id);

        let stats = scheduler.get_stats();
        assert_eq!(stats.total_active, 0);
        assert_eq!(stats.total_queued, 0);
    }

    #[test]
    fn test_query_priority_aging() {
        let config = PrioritySchedulerConfig {
            enable_aging: true,
            aging_threshold: Duration::from_millis(10),
            ..Default::default()
        };

        let scheduler = QueryPriorityScheduler::new(config);

        // Submit low priority query
        let _low_id = scheduler
            .submit_query(
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                QueryPriority::Low,
                None,
                None,
            )
            .unwrap();

        // Wait for aging
        std::thread::sleep(Duration::from_millis(20));

        // Submit normal priority after delay
        scheduler
            .submit_query(
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                QueryPriority::Normal,
                None,
                None,
            )
            .unwrap();

        // The aged low priority query should now be normal priority
        // and compete equally
        let next = scheduler.next_query();
        assert!(next.is_some());

        scheduler.complete_query(next.unwrap().query_id);
    }

    #[test]
    fn test_query_priority_cancel() {
        let scheduler = QueryPriorityScheduler::new(PrioritySchedulerConfig::default());

        let query_id = scheduler
            .submit_query(
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                QueryPriority::Normal,
                None,
                None,
            )
            .unwrap();

        assert_eq!(scheduler.get_stats().total_queued, 1);

        // Cancel the query
        assert!(scheduler.cancel_query(query_id));
        assert_eq!(scheduler.get_stats().total_queued, 0);

        // Cancelling again should fail
        assert!(!scheduler.cancel_query(query_id));
    }

    #[test]
    fn test_query_cost_estimator_lightweight() {
        let estimator = QueryCostEstimator::new(CostEstimatorConfig::default());

        let features = QueryFeatures {
            pattern_count: 2,
            join_count: 1,
            filter_count: 1,
            limit: Some(10),
            ..Default::default()
        };

        let estimate = estimator.estimate_cost(&features);

        assert_eq!(estimate.recommendation, CostRecommendation::Lightweight);
        assert!(estimate.estimated_cost < 100.0);
        assert!(estimate.complexity_score < 1.0);
    }

    #[test]
    fn test_query_cost_estimator_expensive() {
        let estimator = QueryCostEstimator::new(CostEstimatorConfig::default());

        let features = QueryFeatures {
            pattern_count: 10,
            join_count: 5,
            filter_count: 3,
            aggregate_count: 2,
            path_count: 1,
            optional_count: 2,
            distinct: true,
            order_by: true,
            group_by: true,
            ..Default::default()
        };

        let estimate = estimator.estimate_cost(&features);

        assert!(
            estimate.recommendation == CostRecommendation::Expensive
                || estimate.recommendation == CostRecommendation::VeryExpensive
        );
        assert!(estimate.estimated_cost > 500.0);
        assert!(estimate.complexity_score > 5.0);
    }

    #[test]
    fn test_query_cost_estimator_learning() {
        let estimator = QueryCostEstimator::new(CostEstimatorConfig::default());

        let features = QueryFeatures {
            pattern_count: 3,
            join_count: 2,
            ..Default::default()
        };

        // Record some actual costs
        for i in 1..=10 {
            estimator.record_actual_cost(features.clone(), i as f64 * 10.0);
        }

        let stats = estimator.get_statistics();
        assert_eq!(stats.sample_count, 10);
        assert!(stats.avg_cost > 0.0);
        assert_eq!(stats.min_cost, 10.0);
        assert_eq!(stats.max_cost, 100.0);
    }

    #[test]
    fn test_performance_baseline_tracker_basic() {
        let tracker = PerformanceBaselineTracker::new(BaselineTrackerConfig::default());

        // Record some executions
        tracker.record_execution("SELECT ?s ?p ?o".to_string(), 100.0, 10.0, 100);

        tracker.record_execution("SELECT ?s ?p ?o".to_string(), 110.0, 11.0, 105);

        tracker.record_execution("SELECT ?s ?p ?o".to_string(), 105.0, 10.5, 102);

        let patterns = tracker.get_tracked_patterns();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0], "SELECT ?s ?p ?o");
    }

    #[test]
    fn test_performance_baseline_regression_detection() {
        let config = BaselineTrackerConfig {
            regression_threshold: 0.2, // 20% degradation
            min_samples: 3,
            auto_update_baseline: false, // Fixed baseline for testing
            ..Default::default()
        };

        let tracker = PerformanceBaselineTracker::new(config);

        let pattern = "SELECT ?s ?p ?o";

        // Record baseline samples (~100ms)
        for _ in 0..5 {
            tracker.record_execution(pattern.to_string(), 100.0, 10.0, 100);
        }

        // Check for regression with normal performance (110ms = 10% slower)
        let regression = tracker.check_regression(pattern, 110.0);
        assert!(regression.is_none()); // Within threshold

        // Check for regression with degraded performance (130ms = 30% slower)
        let regression = tracker.check_regression(pattern, 130.0);
        assert!(regression.is_some());

        let report = regression.unwrap();
        assert!(report.degradation_percentage > 20.0);
        assert_eq!(report.severity, RegressionSeverity::Moderate);
    }

    #[test]
    fn test_performance_baseline_trend() {
        let tracker = PerformanceBaselineTracker::new(BaselineTrackerConfig::default());

        let pattern = "SELECT ?s ?p ?o";

        // Record samples with varying performance
        let durations = vec![100.0, 110.0, 105.0, 95.0, 120.0, 100.0, 105.0, 110.0];
        for duration in durations {
            tracker.record_execution(pattern.to_string(), duration, 10.0, 100);
        }

        let trend = tracker.get_trend(pattern);
        assert!(trend.is_some());

        let trend = trend.unwrap();
        assert_eq!(trend.sample_count, 8);
        assert!(trend.avg_duration_ms > 0.0);
        assert_eq!(trend.min_duration_ms, 95.0);
        assert_eq!(trend.max_duration_ms, 120.0);
        assert!(trend.std_dev_ms > 0.0);
    }

    #[test]
    fn test_performance_baseline_auto_update() {
        let config = BaselineTrackerConfig {
            auto_update_baseline: true,
            min_samples: 3,
            ..Default::default()
        };

        let tracker = PerformanceBaselineTracker::new(config);

        let pattern = "SELECT ?s ?p ?o";

        // Record initial samples
        for _ in 0..5 {
            tracker.record_execution(pattern.to_string(), 100.0, 10.0, 100);
        }

        // Get initial trend
        let trend1 = tracker.get_trend(pattern).unwrap();
        let baseline1 = trend1.baseline_duration_ms;

        // Record new samples with different performance
        for _ in 0..5 {
            tracker.record_execution(pattern.to_string(), 150.0, 15.0, 100);
        }

        // Baseline should have updated
        let trend2 = tracker.get_trend(pattern).unwrap();
        let baseline2 = trend2.baseline_duration_ms;

        assert!(baseline2 > baseline1); // Baseline increased
    }

    #[test]
    fn test_cost_estimator_limit_optimization() {
        let estimator = QueryCostEstimator::new(CostEstimatorConfig::default());

        // Same features, different limits
        let features_no_limit = QueryFeatures {
            pattern_count: 10,
            join_count: 3,
            filter_count: 2,
            ..Default::default()
        };

        let mut features_small_limit = features_no_limit.clone();
        features_small_limit.limit = Some(10);

        let mut features_large_limit = features_no_limit.clone();
        features_large_limit.limit = Some(500);

        let estimate_no_limit = estimator.estimate_cost(&features_no_limit);
        let estimate_small = estimator.estimate_cost(&features_small_limit);
        let estimate_large = estimator.estimate_cost(&features_large_limit);

        // Small limit should have lowest cost
        assert!(estimate_small.estimated_cost < estimate_large.estimated_cost);
        assert!(estimate_small.estimated_cost < estimate_no_limit.estimated_cost);
    }

    #[test]
    fn test_priority_scheduler_queue_limits() {
        let config = PrioritySchedulerConfig {
            max_per_priority: 2,
            max_total_queued: 5,
            ..Default::default()
        };

        let scheduler = QueryPriorityScheduler::new(config);

        // Fill up normal priority queue
        assert!(scheduler
            .submit_query("Query 1".to_string(), QueryPriority::Normal, None, None)
            .is_ok());
        assert!(scheduler
            .submit_query("Query 2".to_string(), QueryPriority::Normal, None, None)
            .is_ok());

        // Third normal should fail (exceeds max_per_priority)
        assert!(scheduler
            .submit_query("Query 3".to_string(), QueryPriority::Normal, None, None)
            .is_err());

        // But high priority should work
        assert!(scheduler
            .submit_query("Query 4".to_string(), QueryPriority::High, None, None)
            .is_ok());
    }
}
