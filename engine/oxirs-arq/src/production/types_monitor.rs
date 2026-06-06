use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutAction {
    Warn,
    Cancel,
    Throttle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegressionSeverity {
    Moderate,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub last_check: SystemTime,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub success_threshold: usize,
    pub timeout: Duration,
    pub half_open_max_requests: usize,
}

#[derive(Debug, Clone)]
pub struct BaselineTrackerConfig {
    pub window_size: usize,
    pub regression_threshold: f64,
    pub min_samples: usize,
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
struct PerformanceSample {
    #[allow(dead_code)]
    timestamp: SystemTime,
    duration_ms: f64,
    memory_mb: f64,
    #[allow(dead_code)]
    result_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

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
    pub fn get_statistics(&self, query_type: &str) -> super::types_query::QueryStatistics {
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
        super::types_query::QueryStatistics {
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
    pub fn get_global_statistics(&self) -> super::types_resource::GlobalStatistics {
        super::types_resource::GlobalStatistics {
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
    pub fn get_tracked_patterns(&self) -> Vec<String> {
        self.baselines
            .read()
            .expect("lock poisoned")
            .keys()
            .cloned()
            .collect()
    }
    pub fn clear(&self) {
        self.baselines.write().expect("lock poisoned").clear();
    }
}
