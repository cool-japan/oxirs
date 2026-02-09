//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use super::functions::CancellationCallbacks;
use super::querymemorytracker_type::QueryMemoryTracker;

/// Estimates query execution cost for resource planning
pub struct QueryCostEstimator {
    stats_collector: Arc<RwLock<CostStatistics>>,
    config: CostEstimatorConfig,
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
        cost += features.pattern_count as f64 * self.config.pattern_weight;
        cost += (features.join_count as f64).powi(2) * self.config.join_weight;
        cost += features.filter_count as f64 * self.config.filter_weight;
        cost += features.aggregate_count as f64 * self.config.aggregate_weight;
        cost += features.path_count as f64 * self.config.path_weight;
        if features.optional_count > 0 {
            cost *= 1.0 + (features.optional_count as f64 * 0.3);
        }
        if features.union_count > 0 {
            cost *= 1.0 + (features.union_count as f64 * 0.5);
        }
        if features.distinct {
            cost *= 1.5;
        }
        if features.order_by {
            cost *= 1.3;
        }
        if features.group_by {
            cost *= 1.4;
        }
        if let Some(limit) = features.limit {
            if limit < 100 {
                cost *= 0.5;
            } else if limit < 1000 {
                cost *= 0.7;
            }
        }
        let complexity_score = cost / 100.0;
        let estimated_duration_ms = cost * 0.1;
        let estimated_memory_mb = cost * 0.01;
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
        let mut stats = self.stats_collector.write().expect("lock poisoned");
        stats.add_sample(features, actual_duration_ms);
    }
    /// Get historical statistics
    pub fn get_statistics(&self) -> CostEstimatorStatistics {
        let stats = self.stats_collector.read().expect("lock poisoned");
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}
#[derive(Debug, Clone)]
pub struct QueryCostEstimate {
    pub estimated_cost: f64,
    pub estimated_duration_ms: f64,
    pub estimated_memory_mb: f64,
    pub complexity_score: f64,
    pub recommendation: CostRecommendation,
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
/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub memory_limit: usize,
    pub active_queries: usize,
    pub pressure_percentage: f64,
}
/// Circuit breaker for SPARQL query execution
pub struct QueryCircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failures: AtomicUsize,
    successes: AtomicUsize,
    config: CircuitBreakerConfig,
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
        let state = self.state.read().expect("lock poisoned");
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
        let mut state = self.state.write().expect("lock poisoned");
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
        let mut state = self.state.write().expect("lock poisoned");
        let failures = self.failures.fetch_add(1, Ordering::Relaxed) + 1;
        if failures >= self.config.failure_threshold {
            *state = CircuitState::Open;
            self.successes.store(0, Ordering::Relaxed);
        }
    }
    pub fn try_half_open(&self) {
        let mut state = self.state.write().expect("lock poisoned");
        if *state == CircuitState::Open {
            *state = CircuitState::HalfOpen;
            self.successes.store(0, Ordering::Relaxed);
        }
    }
    pub fn state(&self) -> String {
        format!("{:?}", *self.state.read().expect("lock poisoned"))
    }
}
/// Enhanced error type with SPARQL query context
#[derive(Debug, Clone)]
pub struct SparqlProductionError {
    pub error: String,
    pub context: QueryErrorContext,
    pub timestamp: SystemTime,
    pub severity: ErrorSeverity,
    pub retryable: bool,
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
pub(super) struct TokenBucket {
    pub(super) tokens: f64,
    pub(super) last_update: Instant,
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
/// Query session manager for coordinating multiple concurrent sessions
pub struct QuerySessionManager {
    sessions: RwLock<HashMap<u64, Arc<QuerySession>>>,
    timeout_manager: QueryTimeoutManager,
    memory_tracker: QueryMemoryTracker,
    rate_limiter: QueryRateLimiter,
    audit_trail: QueryAuditTrail,
    next_session_id: AtomicU64,
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
        let user_key = user_id.unwrap_or("anonymous");
        if !self.rate_limiter.check_rate_limit(user_key) {
            return Err(anyhow!("Rate limit exceeded for user: {}", user_key));
        }
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
            .expect("lock poisoned")
            .insert(session_id, session.clone());
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
        let session = self
            .sessions
            .write()
            .expect("lock poisoned")
            .remove(&session_id);
        if let Some(session) = session {
            let duration = self.timeout_manager.end_query(session.query_id);
            self.memory_tracker.free_query(session.query_id);
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
        let session = self
            .sessions
            .write()
            .expect("lock poisoned")
            .remove(&session_id);
        if let Some(session) = session {
            let duration = self.timeout_manager.end_query(session.query_id);
            self.memory_tracker.free_query(session.query_id);
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
        let sessions = self.sessions.read().expect("lock poisoned");
        if let Some(session) = sessions.get(&session_id) {
            Ok(self.timeout_manager.check_timeout(session.query_id))
        } else {
            Err(anyhow!("Session {} not found", session_id))
        }
    }
    /// Allocate memory for a session
    pub fn allocate_memory(&self, session_id: u64, bytes: usize) -> Result<()> {
        let sessions = self.sessions.read().expect("lock poisoned");
        if let Some(session) = sessions.get(&session_id) {
            self.memory_tracker.allocate(session.query_id, bytes)
        } else {
            Err(anyhow!("Session {} not found", session_id))
        }
    }
    /// Get session by ID
    pub fn get_session(&self, session_id: u64) -> Option<Arc<QuerySession>> {
        self.sessions
            .read()
            .expect("lock poisoned")
            .get(&session_id)
            .cloned()
    }
    /// Get active session count
    pub fn active_session_count(&self) -> usize {
        self.sessions.read().expect("lock poisoned").len()
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
/// Audit trail for query execution compliance and debugging
pub struct QueryAuditTrail {
    events: RwLock<Vec<QueryAuditEvent>>,
    max_events: usize,
    enabled: AtomicBool,
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
        let mut events = self.events.write().expect("lock poisoned");
        if events.len() >= self.max_events {
            events.remove(0);
        }
        events.push(event);
    }
    /// Get recent events
    pub fn get_recent(&self, limit: usize) -> Vec<QueryAuditEvent> {
        let events = self.events.read().expect("lock poisoned");
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
            .expect("lock poisoned")
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
            .expect("lock poisoned")
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
        self.events.read().expect("lock poisoned").len()
    }
    /// Clear all events
    pub fn clear(&self) {
        self.events.write().expect("lock poisoned").clear();
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
#[derive(Debug, Clone)]
pub struct RegressionReport {
    pub query_pattern: String,
    pub baseline_duration_ms: f64,
    pub current_duration_ms: f64,
    pub degradation_percentage: f64,
    pub sample_count: usize,
    pub severity: RegressionSeverity,
}
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub success_threshold: usize,
    pub timeout: Duration,
    pub half_open_max_requests: usize,
}
/// Health check for SPARQL query engine components
pub struct QueryEngineHealth {
    checks: RwLock<HashMap<String, HealthCheck>>,
}
impl QueryEngineHealth {
    pub fn new() -> Self {
        Self {
            checks: RwLock::new(HashMap::new()),
        }
    }
    pub fn register_check(&self, name: &str) {
        self.checks.write().expect("lock poisoned").insert(
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
        if let Some(check) = self.checks.write().expect("lock poisoned").get_mut(name) {
            check.status = status;
            check.last_check = SystemTime::now();
            check.message = message;
        }
    }
    pub fn check_parser(&self) -> HealthStatus {
        let status = HealthStatus::Healthy;
        self.update_check("parser", status, "Parser is operational".to_string());
        status
    }
    pub fn check_executor(&self) -> HealthStatus {
        let status = HealthStatus::Healthy;
        self.update_check("executor", status, "Executor is operational".to_string());
        status
    }
    pub fn check_optimizer(&self) -> HealthStatus {
        let status = HealthStatus::Healthy;
        self.update_check("optimizer", status, "Optimizer is operational".to_string());
        status
    }
    pub fn get_overall_status(&self) -> HealthStatus {
        let checks = self.checks.read().expect("lock poisoned");
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
        self.checks
            .read()
            .expect("lock poisoned")
            .values()
            .cloned()
            .collect()
    }
    pub fn perform_all_checks(&self) {
        self.check_parser();
        self.check_executor();
        self.check_optimizer();
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegressionSeverity {
    Moderate,
    High,
    Critical,
}
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
#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: SystemTime,
    pub message: String,
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
/// Priority-based query scheduler
pub struct QueryPriorityScheduler {
    queues: Arc<RwLock<HashMap<QueryPriority, Vec<PrioritizedQuery>>>>,
    active_queries: Arc<RwLock<HashMap<u64, QueryPriority>>>,
    next_query_id: Arc<AtomicU64>,
    config: PrioritySchedulerConfig,
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
        let mut queues = self.queues.write().expect("lock poisoned");
        let total_queued: usize = queues.values().map(|q| q.len()).sum();
        if total_queued >= self.config.max_total_queued {
            return Err(anyhow!("Query queue is full"));
        }
        let priority_queue = queues
            .get_mut(&priority)
            .expect("priority queue should exist for all priority levels");
        if priority_queue.len() >= self.config.max_per_priority {
            return Err(anyhow!("Priority queue {:?} is full", priority));
        }
        priority_queue.push(prioritized);
        Ok(query_id)
    }
    /// Get next query to execute based on priority
    pub fn next_query(&self) -> Option<PrioritizedQuery> {
        let mut queues = self.queues.write().expect("lock poisoned");
        if self.config.enable_aging {
            self.process_aging(&mut queues);
        }
        for priority in [
            QueryPriority::Critical,
            QueryPriority::High,
            QueryPriority::Normal,
            QueryPriority::Low,
            QueryPriority::Batch,
        ] {
            let active = self.active_queries.read().expect("lock poisoned");
            let concurrent_count = active.values().filter(|&&p| p == priority).count();
            if let Some(&max_concurrent) = self.config.max_concurrent_per_priority.get(&priority) {
                if concurrent_count >= max_concurrent {
                    continue;
                }
            }
            if let Some(queue) = queues.get_mut(&priority) {
                if !queue.is_empty() {
                    let query = queue.remove(0);
                    drop(active);
                    self.active_queries
                        .write()
                        .expect("lock poisoned")
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
        self.active_queries
            .write()
            .expect("lock poisoned")
            .remove(&query_id);
    }
    /// Cancel a queued query
    pub fn cancel_query(&self, query_id: u64) -> bool {
        let mut queues = self.queues.write().expect("lock poisoned");
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
        let queues = self.queues.read().expect("lock poisoned");
        let active = self.active_queries.read().expect("lock poisoned");
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
        self.query_latencies
            .write()
            .expect("lock poisoned")
            .entry(query_type.to_string())
            .or_default()
            .push(latency);
        self.query_counts
            .write()
            .expect("lock poisoned")
            .entry(query_type.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
        self.pattern_complexities
            .write()
            .expect("lock poisoned")
            .entry(query_type.to_string())
            .or_default()
            .push(pattern_count);
        self.result_sizes
            .write()
            .expect("lock poisoned")
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
        let latencies = self.query_latencies.read().expect("lock poisoned");
        let counts = self.query_counts.read().expect("lock poisoned");
        let complexities = self.pattern_complexities.read().expect("lock poisoned");
        let sizes = self.result_sizes.read().expect("lock poisoned");
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
                .expect("lock poisoned")
                .values()
                .map(|c| c.load(Ordering::Relaxed))
                .sum(),
            total_timeouts: self.timeouts.load(Ordering::Relaxed),
            total_errors: self.errors.load(Ordering::Relaxed),
        }
    }
    pub fn reset(&self) {
        self.query_latencies.write().expect("lock poisoned").clear();
        self.query_counts.write().expect("lock poisoned").clear();
        self.pattern_complexities
            .write()
            .expect("lock poisoned")
            .clear();
        self.result_sizes.write().expect("lock poisoned").clear();
        self.timeouts.store(0, Ordering::Relaxed);
        self.errors.store(0, Ordering::Relaxed);
    }
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
/// Token bucket rate limiter for query requests
pub struct QueryRateLimiter {
    pub(super) buckets: RwLock<HashMap<String, TokenBucket>>,
    pub(super) requests_per_second: AtomicU32,
    pub(super) burst_size: AtomicU32,
    pub(super) enabled: AtomicBool,
}
impl QueryRateLimiter {
    /// Check if request is allowed for the given key
    pub fn check_rate_limit(&self, key: &str) -> bool {
        if !self.enabled.load(Ordering::Relaxed) {
            return true;
        }
        let rate = self.requests_per_second.load(Ordering::Relaxed) as f64;
        let burst = self.burst_size.load(Ordering::Relaxed) as f64;
        let mut buckets = self.buckets.write().expect("lock poisoned");
        let bucket = buckets
            .entry(key.to_string())
            .or_insert_with(|| TokenBucket {
                tokens: burst,
                last_update: Instant::now(),
            });
        let now = Instant::now();
        let elapsed = now.duration_since(bucket.last_update).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * rate).min(burst);
        bucket.last_update = now;
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
        self.buckets
            .read()
            .expect("lock poisoned")
            .get(key)
            .map(|b| b.tokens)
    }
    /// Clear rate limit state for all keys
    pub fn clear(&self) {
        self.buckets.write().expect("lock poisoned").clear();
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
            *self.cancel_time.write().expect("lock poisoned") = Some(Instant::now());
            *self.reason.write().expect("lock poisoned") = reason;
            let callbacks = self.callbacks.read().expect("lock poisoned");
            for callback in callbacks.iter() {
                callback();
            }
        }
    }
    /// Get the cancellation reason if available
    pub fn get_reason(&self) -> Option<String> {
        self.reason.read().expect("lock poisoned").clone()
    }
    /// Get the time when cancellation was requested
    pub fn cancel_time(&self) -> Option<Instant> {
        *self.cancel_time.read().expect("lock poisoned")
    }
    /// Register a callback to be executed when cancelled
    pub fn on_cancel<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.callbacks
            .write()
            .expect("lock poisoned")
            .push(Box::new(callback));
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
        self.on_cancel(move || {
            if parent_cancelled.load(Ordering::Relaxed) {
                child_cancelled.store(true, Ordering::Relaxed);
            }
        });
        child
    }
}
#[derive(Debug, Clone)]
pub struct GlobalStatistics {
    pub uptime: Duration,
    pub total_queries: u64,
    pub total_timeouts: u64,
    pub total_errors: u64,
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
struct PerformanceBaseline {
    #[allow(dead_code)]
    query_pattern: String,
    samples: Vec<PerformanceSample>,
    baseline_duration_ms: f64,
    baseline_memory_mb: f64,
    last_updated: SystemTime,
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
#[derive(Debug, Clone)]
struct PerformanceSample {
    #[allow(dead_code)]
    timestamp: SystemTime,
    duration_ms: f64,
    memory_mb: f64,
    #[allow(dead_code)]
    result_count: usize,
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
        let max = *self.max_query_time.read().expect("lock poisoned");
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
        *self.max_query_time.write().expect("lock poisoned") = limit;
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
/// Tracks performance baselines for detecting regressions
pub struct PerformanceBaselineTracker {
    baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
    config: BaselineTrackerConfig,
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
        let mut baselines = self.baselines.write().expect("lock poisoned");
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
        if baseline.samples.len() > self.config.window_size {
            baseline.samples.remove(0);
        }
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
        let baselines = self.baselines.read().expect("lock poisoned");
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
        let baselines = self.baselines.read().expect("lock poisoned");
        if let Some(baseline) = baselines.get(query_pattern) {
            if baseline.samples.is_empty() {
                return None;
            }
            let durations: Vec<f64> = baseline.samples.iter().map(|s| s.duration_ms).collect();
            let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
            let min_duration = durations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_duration = durations.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
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
        self.baselines
            .read()
            .expect("lock poisoned")
            .keys()
            .cloned()
            .collect()
    }
    /// Clear baselines
    pub fn clear(&self) {
        self.baselines.write().expect("lock poisoned").clear();
    }
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
/// Advanced timeout management for queries
///
/// Provides configurable soft and hard timeouts with warning callbacks.
pub struct QueryTimeoutManager {
    pub(super) soft_timeout: RwLock<Duration>,
    pub(super) hard_timeout: RwLock<Duration>,
    pub(super) warning_thresholds: RwLock<Vec<f64>>,
    pub(super) active_queries: RwLock<HashMap<u64, QueryTimeoutState>>,
    pub(super) next_query_id: AtomicU64,
    pub(super) timeout_action: RwLock<TimeoutAction>,
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
        self.active_queries
            .write()
            .expect("lock poisoned")
            .insert(query_id, state);
        query_id
    }
    /// End tracking for a query
    pub fn end_query(&self, query_id: u64) -> Option<Duration> {
        self.active_queries
            .write()
            .expect("lock poisoned")
            .remove(&query_id)
            .map(|state| state.start_time.elapsed())
    }
    /// Check if a query has exceeded its timeout
    pub fn check_timeout(&self, query_id: u64) -> TimeoutCheckResult {
        let hard = *self.hard_timeout.read().expect("lock poisoned");
        let soft = *self.soft_timeout.read().expect("lock poisoned");
        let thresholds = self
            .warning_thresholds
            .read()
            .expect("lock poisoned")
            .clone();
        let mut queries = self.active_queries.write().expect("lock poisoned");
        if let Some(state) = queries.get_mut(&query_id) {
            let elapsed = state.start_time.elapsed();
            if elapsed > hard {
                return TimeoutCheckResult::HardTimeout {
                    elapsed,
                    limit: hard,
                };
            }
            if elapsed > soft && !state.soft_timeout_triggered {
                state.soft_timeout_triggered = true;
                return TimeoutCheckResult::SoftTimeout {
                    elapsed,
                    limit: soft,
                };
            }
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
        let hard = *self.hard_timeout.read().expect("lock poisoned");
        self.active_queries
            .read()
            .expect("lock poisoned")
            .get(&query_id)
            .map(|state| hard.saturating_sub(state.start_time.elapsed()))
    }
    /// Set timeouts
    pub fn set_soft_timeout(&self, timeout: Duration) {
        *self.soft_timeout.write().expect("lock poisoned") = timeout;
    }
    pub fn set_hard_timeout(&self, timeout: Duration) {
        *self.hard_timeout.write().expect("lock poisoned") = timeout;
    }
    pub fn set_warning_thresholds(&self, thresholds: Vec<f64>) {
        *self.warning_thresholds.write().expect("lock poisoned") = thresholds;
    }
    pub fn set_timeout_action(&self, action: TimeoutAction) {
        *self.timeout_action.write().expect("lock poisoned") = action;
    }
    /// Get count of active queries
    pub fn active_query_count(&self) -> usize {
        self.active_queries.read().expect("lock poisoned").len()
    }
    /// Get all active query states
    pub fn get_active_queries(&self) -> Vec<QueryTimeoutState> {
        self.active_queries
            .read()
            .expect("lock poisoned")
            .values()
            .cloned()
            .collect()
    }
}
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
#[derive(Debug, Clone)]
pub struct PrioritySchedulerStats {
    pub total_queued: usize,
    pub total_active: usize,
    pub queued_per_priority: HashMap<QueryPriority, usize>,
    pub active_per_priority: HashMap<QueryPriority, usize>,
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
