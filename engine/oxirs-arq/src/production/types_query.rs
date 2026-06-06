use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use super::functions::CancellationCallbacks;
use super::querymemorytracker_type::QueryMemoryTracker;
use super::types_monitor::TimeoutAction;
use super::types_resource::MemoryStats;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum QueryPriority {
    Critical = 4,
    High = 3,
    #[default]
    Normal = 2,
    Low = 1,
    Batch = 0,
}

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

#[derive(Debug, Clone, PartialEq)]
pub enum CostRecommendation {
    Lightweight,
    Moderate,
    Expensive,
    VeryExpensive,
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
pub struct CostEstimatorConfig {
    pub pattern_weight: f64,
    pub join_weight: f64,
    pub filter_weight: f64,
    pub aggregate_weight: f64,
    pub path_weight: f64,
    pub enable_ml_prediction: bool,
}

#[derive(Debug, Clone)]
pub struct CostEstimatorStatistics {
    pub sample_count: usize,
    pub avg_cost: f64,
    pub min_cost: f64,
    pub max_cost: f64,
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
    pub fn record_actual_cost(&self, features: QueryFeatures, actual_duration_ms: f64) {
        let mut stats = self.stats_collector.write().expect("lock poisoned");
        stats.add_sample(features, actual_duration_ms);
    }
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

pub(super) struct TokenBucket {
    pub(super) tokens: f64,
    pub(super) last_update: Instant,
}

pub struct QueryRateLimiter {
    pub(super) buckets: RwLock<HashMap<String, TokenBucket>>,
    pub(super) requests_per_second: AtomicU32,
    pub(super) burst_size: AtomicU32,
    pub(super) enabled: AtomicBool,
}

impl QueryRateLimiter {
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
    pub fn configure(&self, requests_per_second: u32, burst_size: u32) {
        self.requests_per_second
            .store(requests_per_second, Ordering::Relaxed);
        self.burst_size.store(burst_size, Ordering::Relaxed);
    }
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
    pub fn get_stats(&self, key: &str) -> Option<f64> {
        self.buckets
            .read()
            .expect("lock poisoned")
            .get(key)
            .map(|b| b.tokens)
    }
    pub fn clear(&self) {
        self.buckets.write().expect("lock poisoned").clear();
    }
}

#[derive(Debug, Clone)]
pub enum TimeoutCheckResult {
    Ok {
        elapsed: Duration,
    },
    SoftTimeout {
        elapsed: Duration,
        limit: Duration,
    },
    HardTimeout {
        elapsed: Duration,
        limit: Duration,
    },
    Warning {
        elapsed: Duration,
        threshold: f64,
        remaining: Duration,
    },
    QueryNotFound,
}

#[derive(Debug, Clone)]
pub struct QueryTimeoutState {
    pub query_id: u64,
    pub start_time: Instant,
    pub query_snippet: String,
    pub soft_timeout_triggered: bool,
    pub warnings_triggered: Vec<f64>,
}

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
    pub fn end_query(&self, query_id: u64) -> Option<Duration> {
        self.active_queries
            .write()
            .expect("lock poisoned")
            .remove(&query_id)
            .map(|state| state.start_time.elapsed())
    }
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
    pub fn remaining_time(&self, query_id: u64) -> Option<Duration> {
        let hard = *self.hard_timeout.read().expect("lock poisoned");
        self.active_queries
            .read()
            .expect("lock poisoned")
            .get(&query_id)
            .map(|state| hard.saturating_sub(state.start_time.elapsed()))
    }
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
    pub fn active_query_count(&self) -> usize {
        self.active_queries.read().expect("lock poisoned").len()
    }
    pub fn get_active_queries(&self) -> Vec<QueryTimeoutState> {
        self.active_queries
            .read()
            .expect("lock poisoned")
            .values()
            .cloned()
            .collect()
    }
}

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
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
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
    pub fn get_reason(&self) -> Option<String> {
        self.reason.read().expect("lock poisoned").clone()
    }
    pub fn cancel_time(&self) -> Option<Instant> {
        *self.cancel_time.read().expect("lock poisoned")
    }
    pub fn on_cancel<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.callbacks
            .write()
            .expect("lock poisoned")
            .push(Box::new(callback));
    }
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

pub struct QuerySession {
    pub session_id: u64,
    pub query_id: u64,
    pub cancellation_token: QueryCancellationToken,
    pub start_time: Instant,
    query: String,
    pub(super) user_id: Option<String>,
    metadata: HashMap<String, String>,
}

impl QuerySession {
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
    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    pub fn query_snippet(&self) -> String {
        self.query.chars().take(100).collect()
    }
    pub fn cancel(&self, reason: Option<String>) {
        self.cancellation_token.cancel(reason);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token.is_cancelled()
    }
}

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
    pub fn get_recent(&self, limit: usize) -> Vec<QueryAuditEvent> {
        let events = self.events.read().expect("lock poisoned");
        let start = if events.len() > limit {
            events.len() - limit
        } else {
            0
        };
        events[start..].to_vec()
    }
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
    pub fn get_failures(&self, limit: usize) -> Vec<QueryAuditEvent> {
        self.get_by_type(AuditEventType::QueryFailed, limit)
    }
    pub fn event_count(&self) -> usize {
        self.events.read().expect("lock poisoned").len()
    }
    pub fn clear(&self) {
        self.events.write().expect("lock poisoned").clear();
    }
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
pub struct PrioritySchedulerConfig {
    pub max_per_priority: usize,
    pub max_total_queued: usize,
    pub max_concurrent_per_priority: HashMap<QueryPriority, usize>,
    pub enable_aging: bool,
    pub aging_threshold: Duration,
}

#[derive(Debug, Clone)]
pub struct PrioritySchedulerStats {
    pub total_queued: usize,
    pub total_active: usize,
    pub queued_per_priority: HashMap<QueryPriority, usize>,
    pub active_per_priority: HashMap<QueryPriority, usize>,
}

#[derive(Debug, Clone)]
pub struct PrioritizedQuery {
    pub query_id: u64,
    pub priority: QueryPriority,
    pub submitted_at: SystemTime,
    pub query_text: String,
    pub user_id: Option<String>,
    pub estimated_cost: Option<f64>,
}

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
    pub fn complete_query(&self, query_id: u64) {
        self.active_queries
            .write()
            .expect("lock poisoned")
            .remove(&query_id);
    }
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

pub struct SparqlProductionError {
    pub error: String,
    pub context: QueryErrorContext,
    pub timestamp: SystemTime,
    pub severity: super::types_monitor::ErrorSeverity,
    pub retryable: bool,
}

impl SparqlProductionError {
    pub fn new(
        error: String,
        context: QueryErrorContext,
        severity: super::types_monitor::ErrorSeverity,
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
            super::types_monitor::ErrorSeverity::Error,
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
            super::types_monitor::ErrorSeverity::Error,
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
            super::types_monitor::ErrorSeverity::Warning,
            true,
        )
    }
}

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
    pub fn check_timeout(&self, session_id: u64) -> Result<TimeoutCheckResult> {
        let sessions = self.sessions.read().expect("lock poisoned");
        if let Some(session) = sessions.get(&session_id) {
            Ok(self.timeout_manager.check_timeout(session.query_id))
        } else {
            Err(anyhow!("Session {} not found", session_id))
        }
    }
    pub fn allocate_memory(&self, session_id: u64, bytes: usize) -> Result<()> {
        let sessions = self.sessions.read().expect("lock poisoned");
        if let Some(session) = sessions.get(&session_id) {
            self.memory_tracker.allocate(session.query_id, bytes)
        } else {
            Err(anyhow!("Session {} not found", session_id))
        }
    }
    pub fn get_session(&self, session_id: u64) -> Option<Arc<QuerySession>> {
        self.sessions
            .read()
            .expect("lock poisoned")
            .get(&session_id)
            .cloned()
    }
    pub fn active_session_count(&self) -> usize {
        self.sessions.read().expect("lock poisoned").len()
    }
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory_tracker.get_stats()
    }
    pub fn get_audit_events(&self, limit: usize) -> Vec<QueryAuditEvent> {
        self.audit_trail.get_recent(limit)
    }
    pub fn configure_rate_limit(&self, requests_per_second: u32, burst_size: u32) {
        self.rate_limiter.configure(requests_per_second, burst_size);
    }
    pub fn configure_timeouts(&self, soft: Duration, hard: Duration) {
        self.timeout_manager.set_soft_timeout(soft);
        self.timeout_manager.set_hard_timeout(hard);
    }
    pub fn configure_memory(&self, global_limit: usize, per_query_limit: usize) {
        self.memory_tracker.set_memory_limit(global_limit);
        self.memory_tracker.set_per_query_limit(per_query_limit);
    }
}
