//! # Advanced Fault Tolerance for Stream Processing
//!
//! Provides production-grade fault tolerance primitives for stream processing:
//! health monitoring, bulkhead isolation, configurable retry policies, and
//! worker supervision.
//!
//! ## Components
//!
//! - [`StreamHealthMonitor`]: Monitors stream health with configurable thresholds
//! - [`BulkheadIsolator`]: Isolates stream failures using the bulkhead pattern
//! - [`StreamRetryPolicy`]: Configurable retry with exponential backoff
//! - [`StreamSupervisor`]: Supervises stream workers and restarts on failure

pub mod checkpoint_recovery;
pub use checkpoint_recovery::*;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

// ─── Error Types ─────────────────────────────────────────────────────────────

/// Errors in fault tolerance operations
#[derive(Error, Debug, Clone)]
pub enum FaultToleranceError {
    #[error("Bulkhead full: compartment {compartment} has reached capacity {capacity}")]
    BulkheadFull {
        compartment: String,
        capacity: usize,
    },

    #[error("Max retries exceeded: {attempts} attempts for operation {operation}")]
    MaxRetriesExceeded { attempts: u32, operation: String },

    #[error("Worker {worker_id} failed to restart after {attempts} attempts")]
    SupervisorRestartFailed { worker_id: String, attempts: u32 },

    #[error("Health check failed: metric {metric} value {value} exceeds threshold {threshold}")]
    HealthCheckFailed {
        metric: String,
        value: f64,
        threshold: f64,
    },

    #[error("Operation timeout after {elapsed_ms}ms (limit {timeout_ms}ms)")]
    OperationTimeout { elapsed_ms: u64, timeout_ms: u64 },
}

/// Result type for fault tolerance operations
pub type FaultResult<T> = Result<T, FaultToleranceError>;

// ─── Stream Health Monitor ────────────────────────────────────────────────────

/// A threshold rule that can trigger a health alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthThreshold {
    /// Name of the metric (e.g. "error_rate", "latency_p99_ms")
    pub metric_name: String,
    /// Value above which the health is considered degraded
    pub warn_threshold: f64,
    /// Value above which the health is considered critical
    pub critical_threshold: f64,
}

/// Severity of a health alert
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthAlertSeverity {
    /// Metric crossed the warning threshold
    Warning,
    /// Metric crossed the critical threshold
    Critical,
    /// Metric has recovered below warning threshold
    Recovered,
}

/// A health alert emitted by the monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    /// Metric that triggered the alert
    pub metric_name: String,
    /// Current metric value
    pub current_value: f64,
    /// Applicable threshold that was crossed
    pub threshold: f64,
    /// Alert severity
    pub severity: HealthAlertSeverity,
    /// When the alert was raised
    pub raised_at: SystemTime,
}

/// Overall health status of the stream
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StreamHealthStatus {
    /// All metrics are within normal thresholds
    Healthy,
    /// One or more metrics have crossed warning thresholds
    Degraded,
    /// One or more metrics have crossed critical thresholds
    Critical,
    /// Health data is too old to be reliable
    Unknown,
}

/// A snapshot of all health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSnapshot {
    /// Current health status
    pub status: StreamHealthStatus,
    /// All current metric values
    pub metrics: HashMap<String, f64>,
    /// Active alerts
    pub active_alerts: Vec<HealthAlert>,
    /// Timestamp of this snapshot
    pub snapshot_time: SystemTime,
}

/// Configuration for the stream health monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMonitorConfig {
    /// Health thresholds per metric
    pub thresholds: Vec<HealthThreshold>,
    /// Maximum age of a metric before it is considered stale
    pub metric_staleness: Duration,
    /// How often to evaluate thresholds
    pub check_interval: Duration,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            thresholds: vec![
                HealthThreshold {
                    metric_name: "error_rate".to_string(),
                    warn_threshold: 0.01,
                    critical_threshold: 0.05,
                },
                HealthThreshold {
                    metric_name: "latency_p99_ms".to_string(),
                    warn_threshold: 100.0,
                    critical_threshold: 500.0,
                },
                HealthThreshold {
                    metric_name: "backpressure_ratio".to_string(),
                    warn_threshold: 0.5,
                    critical_threshold: 0.9,
                },
            ],
            metric_staleness: Duration::from_secs(60),
            check_interval: Duration::from_secs(5),
        }
    }
}

/// Monitors stream health with configurable thresholds.
///
/// Accepts metric updates from the stream pipeline and evaluates them against
/// configured thresholds to produce health alerts.
pub struct StreamHealthMonitor {
    config: HealthMonitorConfig,
    /// Current metric values with timestamps
    metrics: Arc<RwLock<HashMap<String, (f64, Instant)>>>,
    /// Currently active alerts
    active_alerts: Arc<RwLock<Vec<HealthAlert>>>,
    /// Alert history (capped to last 1000 alerts)
    alert_history: Arc<RwLock<Vec<HealthAlert>>>,
    /// Total alerts raised since creation
    total_alerts_raised: Arc<RwLock<u64>>,
}

impl StreamHealthMonitor {
    /// Creates a new health monitor with the given configuration.
    pub fn new(config: HealthMonitorConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(Vec::new())),
            alert_history: Arc::new(RwLock::new(Vec::new())),
            total_alerts_raised: Arc::new(RwLock::new(0)),
        }
    }

    /// Records a new metric value and evaluates thresholds.
    ///
    /// Returns any new alerts that were raised.
    pub fn record_metric(&self, metric_name: &str, value: f64) -> Vec<HealthAlert> {
        self.metrics
            .write()
            .insert(metric_name.to_string(), (value, Instant::now()));
        self.evaluate_thresholds(metric_name, value)
    }

    /// Returns the current health snapshot.
    pub fn snapshot(&self) -> HealthSnapshot {
        let metrics = self.metrics.read();
        let now = Instant::now();
        let stale_limit = self.config.metric_staleness;

        // Check for stale metrics
        let all_fresh = metrics
            .values()
            .all(|(_, ts)| now.duration_since(*ts) < stale_limit);

        let metric_values: HashMap<String, f64> =
            metrics.iter().map(|(k, (v, _))| (k.clone(), *v)).collect();

        let active_alerts = self.active_alerts.read().clone();

        let status = if !all_fresh || metric_values.is_empty() {
            StreamHealthStatus::Unknown
        } else if active_alerts
            .iter()
            .any(|a| a.severity == HealthAlertSeverity::Critical)
        {
            StreamHealthStatus::Critical
        } else if active_alerts
            .iter()
            .any(|a| a.severity == HealthAlertSeverity::Warning)
        {
            StreamHealthStatus::Degraded
        } else {
            StreamHealthStatus::Healthy
        };

        HealthSnapshot {
            status,
            metrics: metric_values,
            active_alerts,
            snapshot_time: SystemTime::now(),
        }
    }

    /// Returns current metric value for a named metric, if present.
    pub fn current_metric(&self, name: &str) -> Option<f64> {
        self.metrics.read().get(name).map(|(v, _)| *v)
    }

    /// Returns the total number of alerts ever raised.
    pub fn total_alerts_raised(&self) -> u64 {
        *self.total_alerts_raised.read()
    }

    fn evaluate_thresholds(&self, metric_name: &str, value: f64) -> Vec<HealthAlert> {
        let mut new_alerts = Vec::new();
        let thresholds = self.config.thresholds.clone();

        for threshold in &thresholds {
            if threshold.metric_name != metric_name {
                continue;
            }
            let severity = if value >= threshold.critical_threshold {
                Some(HealthAlertSeverity::Critical)
            } else if value >= threshold.warn_threshold {
                Some(HealthAlertSeverity::Warning)
            } else {
                // Potentially recovered — remove existing alert for this metric
                let mut active = self.active_alerts.write();
                active.retain(|a| a.metric_name != metric_name);
                None
            };

            if let Some(sev) = severity {
                let threshold_val = if sev == HealthAlertSeverity::Critical {
                    threshold.critical_threshold
                } else {
                    threshold.warn_threshold
                };
                let alert = HealthAlert {
                    metric_name: metric_name.to_string(),
                    current_value: value,
                    threshold: threshold_val,
                    severity: sev,
                    raised_at: SystemTime::now(),
                };
                // Upsert active alert for this metric
                let mut active = self.active_alerts.write();
                active.retain(|a| a.metric_name != metric_name);
                active.push(alert.clone());
                drop(active);

                // Append to history (cap at 1000)
                let mut history = self.alert_history.write();
                if history.len() >= 1000 {
                    history.remove(0);
                }
                history.push(alert.clone());

                *self.total_alerts_raised.write() += 1;
                new_alerts.push(alert);
                debug!("Health alert raised for metric {}: {}", metric_name, value);
            }
        }
        new_alerts
    }
}

// ─── Bulkhead Isolator ────────────────────────────────────────────────────────

/// Statistics for a single bulkhead compartment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompartmentStats {
    /// Compartment identifier
    pub compartment_id: String,
    /// Maximum concurrent operations
    pub capacity: usize,
    /// Currently active operations
    pub active: usize,
    /// Total operations rejected due to full compartment
    pub rejected: u64,
    /// Total operations accepted
    pub accepted: u64,
}

/// Configuration for the bulkhead isolator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkheadConfig {
    /// Capacity per named compartment
    pub compartment_capacities: HashMap<String, usize>,
    /// Default capacity for compartments not explicitly listed
    pub default_capacity: usize,
}

impl Default for BulkheadConfig {
    fn default() -> Self {
        let mut compartments = HashMap::new();
        compartments.insert("critical".to_string(), 100);
        compartments.insert("standard".to_string(), 50);
        compartments.insert("background".to_string(), 20);
        Self {
            compartment_capacities: compartments,
            default_capacity: 30,
        }
    }
}

/// A permit acquired from the bulkhead; releases on drop.
pub struct BulkheadPermit {
    compartment_id: String,
    active_counter: Arc<RwLock<usize>>,
}

impl Drop for BulkheadPermit {
    fn drop(&mut self) {
        let mut active = self.active_counter.write();
        if *active > 0 {
            *active -= 1;
        }
        debug!(
            "Bulkhead permit released for compartment {}",
            self.compartment_id
        );
    }
}

/// Internal state for a single compartment
struct Compartment {
    capacity: usize,
    active: Arc<RwLock<usize>>,
    rejected: Arc<RwLock<u64>>,
    accepted: Arc<RwLock<u64>>,
}

/// Isolates stream failures using the bulkhead pattern.
///
/// Divides the system into isolated compartments with independent concurrency
/// limits; a failure or overload in one compartment does not affect others.
pub struct BulkheadIsolator {
    compartments: Arc<RwLock<HashMap<String, Compartment>>>,
    default_capacity: usize,
}

impl BulkheadIsolator {
    /// Creates a new bulkhead isolator with the given configuration.
    pub fn new(config: BulkheadConfig) -> Self {
        let mut compartments = HashMap::new();
        for (id, capacity) in &config.compartment_capacities {
            compartments.insert(
                id.clone(),
                Compartment {
                    capacity: *capacity,
                    active: Arc::new(RwLock::new(0)),
                    rejected: Arc::new(RwLock::new(0)),
                    accepted: Arc::new(RwLock::new(0)),
                },
            );
        }
        Self {
            compartments: Arc::new(RwLock::new(compartments)),
            default_capacity: config.default_capacity,
        }
    }

    /// Attempts to acquire a permit for the named compartment.
    ///
    /// Returns `Ok(BulkheadPermit)` if capacity is available; `Err` if full.
    pub fn acquire(&self, compartment_id: &str) -> FaultResult<BulkheadPermit> {
        let mut compartments = self.compartments.write();
        // Auto-create compartment with default capacity if unknown
        let compartment = compartments
            .entry(compartment_id.to_string())
            .or_insert_with(|| Compartment {
                capacity: self.default_capacity,
                active: Arc::new(RwLock::new(0)),
                rejected: Arc::new(RwLock::new(0)),
                accepted: Arc::new(RwLock::new(0)),
            });

        let current = *compartment.active.read();
        if current >= compartment.capacity {
            *compartment.rejected.write() += 1;
            return Err(FaultToleranceError::BulkheadFull {
                compartment: compartment_id.to_string(),
                capacity: compartment.capacity,
            });
        }
        *compartment.active.write() += 1;
        *compartment.accepted.write() += 1;
        debug!(
            "Bulkhead permit acquired for compartment {} ({}/{})",
            compartment_id,
            current + 1,
            compartment.capacity
        );

        Ok(BulkheadPermit {
            compartment_id: compartment_id.to_string(),
            active_counter: Arc::clone(&compartment.active),
        })
    }

    /// Returns statistics for all compartments.
    pub fn stats(&self) -> Vec<CompartmentStats> {
        self.compartments
            .read()
            .iter()
            .map(|(id, c)| CompartmentStats {
                compartment_id: id.clone(),
                capacity: c.capacity,
                active: *c.active.read(),
                rejected: *c.rejected.read(),
                accepted: *c.accepted.read(),
            })
            .collect()
    }

    /// Returns statistics for a specific compartment.
    pub fn compartment_stats(&self, compartment_id: &str) -> Option<CompartmentStats> {
        self.compartments
            .read()
            .get(compartment_id)
            .map(|c| CompartmentStats {
                compartment_id: compartment_id.to_string(),
                capacity: c.capacity,
                active: *c.active.read(),
                rejected: *c.rejected.read(),
                accepted: *c.accepted.read(),
            })
    }
}

// ─── Stream Retry Policy ──────────────────────────────────────────────────────

/// Configures when and how many times an operation should be retried
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamRetryPolicy {
    /// Maximum number of retry attempts (0 = no retries)
    pub max_attempts: u32,
    /// Initial delay before first retry
    pub initial_delay: Duration,
    /// Multiplier applied to delay after each retry (exponential backoff)
    pub backoff_multiplier: f64,
    /// Maximum delay cap (prevents unbounded backoff)
    pub max_delay: Duration,
    /// Whether to add random jitter (fraction of delay) to prevent thundering herd
    pub jitter: bool,
}

impl Default for StreamRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(30),
            jitter: true,
        }
    }
}

impl StreamRetryPolicy {
    /// Returns the delay before the nth retry attempt (0-indexed).
    ///
    /// Incorporates exponential backoff and an optional pseudo-random jitter
    /// derived deterministically from the attempt number (no rand crate needed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let factor = self.backoff_multiplier.powi(attempt as i32);
        let base_ms = self.initial_delay.as_millis() as f64 * factor;
        let capped_ms = base_ms.min(self.max_delay.as_millis() as f64);

        let jitter_ms = if self.jitter {
            // Deterministic jitter: 0..25% of capped_ms using linear congruential
            let pseudo = ((attempt as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1))
                % 1000;
            let ratio = pseudo as f64 / 4000.0; // 0..0.25
            capped_ms * ratio
        } else {
            0.0
        };

        Duration::from_millis((capped_ms + jitter_ms) as u64)
    }

    /// Executes a fallible synchronous closure with retries.
    ///
    /// Uses `std::thread::sleep` for backoff delays. For async contexts see
    /// [`StreamRetryPolicy::retry_async`].
    pub fn retry<F, T, E>(&self, operation_name: &str, mut f: F) -> FaultResult<T>
    where
        F: FnMut() -> Result<T, E>,
        E: std::fmt::Debug,
    {
        for attempt in 0..=self.max_attempts {
            match f() {
                Ok(result) => {
                    if attempt > 0 {
                        info!(
                            "Operation {} succeeded after {} retries",
                            operation_name, attempt
                        );
                    }
                    return Ok(result);
                }
                Err(err) => {
                    if attempt >= self.max_attempts {
                        warn!(
                            "Operation {} failed after {} attempts: {:?}",
                            operation_name,
                            attempt + 1,
                            err
                        );
                        return Err(FaultToleranceError::MaxRetriesExceeded {
                            attempts: attempt + 1,
                            operation: operation_name.to_string(),
                        });
                    }
                    let delay = self.delay_for_attempt(attempt);
                    debug!(
                        "Operation {} attempt {} failed, retrying in {:?}",
                        operation_name,
                        attempt + 1,
                        delay
                    );
                    std::thread::sleep(delay);
                }
            }
        }
        // Unreachable but satisfies the type checker
        Err(FaultToleranceError::MaxRetriesExceeded {
            attempts: self.max_attempts + 1,
            operation: operation_name.to_string(),
        })
    }

    /// Executes a fallible async closure with retries and tokio async sleep.
    pub async fn retry_async<F, Fut, T, E>(&self, operation_name: &str, mut f: F) -> FaultResult<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        for attempt in 0..=self.max_attempts {
            match f().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!(
                            "Async operation {} succeeded after {} retries",
                            operation_name, attempt
                        );
                    }
                    return Ok(result);
                }
                Err(err) => {
                    if attempt >= self.max_attempts {
                        warn!(
                            "Async operation {} failed after {} attempts: {:?}",
                            operation_name,
                            attempt + 1,
                            err
                        );
                        return Err(FaultToleranceError::MaxRetriesExceeded {
                            attempts: attempt + 1,
                            operation: operation_name.to_string(),
                        });
                    }
                    let delay = self.delay_for_attempt(attempt);
                    debug!(
                        "Async operation {} attempt {} failed, retrying in {:?}",
                        operation_name,
                        attempt + 1,
                        delay
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
        Err(FaultToleranceError::MaxRetriesExceeded {
            attempts: self.max_attempts + 1,
            operation: operation_name.to_string(),
        })
    }
}

// ─── Stream Supervisor ────────────────────────────────────────────────────────

/// The status of a supervised worker
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerStatus {
    /// Worker is running normally
    Running,
    /// Worker has failed and is waiting for restart
    Failed,
    /// Worker is being restarted
    Restarting,
    /// Worker has been stopped intentionally
    Stopped,
    /// Worker has exceeded max restart attempts and is permanently failed
    Exhausted,
}

/// A record of a worker restart
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartRecord {
    /// Worker ID
    pub worker_id: String,
    /// Restart attempt number
    pub attempt: u32,
    /// Reason for restart
    pub reason: String,
    /// When the restart was attempted
    pub restarted_at: SystemTime,
    /// Whether the restart succeeded
    pub success: bool,
}

/// Internal worker state tracked by the supervisor
#[derive(Debug, Clone)]
struct WorkerState {
    worker_id: String,
    status: WorkerStatus,
    restart_count: u32,
    max_restarts: u32,
    last_failure: Option<SystemTime>,
    last_restart: Option<SystemTime>,
}

/// Configuration for the stream supervisor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorConfig {
    /// Maximum number of restart attempts per worker before giving up
    pub max_restarts: u32,
    /// Restart backoff policy
    pub restart_policy: StreamRetryPolicy,
    /// Whether to propagate failure to sibling workers (one-for-all strategy)
    pub one_for_all: bool,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            restart_policy: StreamRetryPolicy {
                max_attempts: 5,
                initial_delay: Duration::from_millis(500),
                backoff_multiplier: 2.0,
                max_delay: Duration::from_secs(60),
                jitter: true,
            },
            one_for_all: false,
        }
    }
}

/// Statistics for the stream supervisor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupervisorStats {
    /// Total workers registered
    pub total_workers: usize,
    /// Workers currently running
    pub running_workers: usize,
    /// Workers permanently failed
    pub exhausted_workers: usize,
    /// Total restart events
    pub total_restarts: u64,
    /// Total restart records
    pub restart_history_len: usize,
}

/// Supervises stream workers and restarts them on failure.
///
/// Tracks worker health, enforces restart limits, and optionally propagates
/// failures to sibling workers (one-for-all supervision strategy).
pub struct StreamSupervisor {
    config: SupervisorConfig,
    workers: Arc<RwLock<HashMap<String, WorkerState>>>,
    restart_history: Arc<RwLock<Vec<RestartRecord>>>,
    total_restarts: Arc<RwLock<u64>>,
}

impl StreamSupervisor {
    /// Creates a new supervisor with the given configuration.
    pub fn new(config: SupervisorConfig) -> Self {
        Self {
            config,
            workers: Arc::new(RwLock::new(HashMap::new())),
            restart_history: Arc::new(RwLock::new(Vec::new())),
            total_restarts: Arc::new(RwLock::new(0)),
        }
    }

    /// Registers a worker with the supervisor.
    pub fn register_worker(&self, worker_id: impl Into<String>) {
        let id = worker_id.into();
        self.workers.write().insert(
            id.clone(),
            WorkerState {
                worker_id: id,
                status: WorkerStatus::Running,
                restart_count: 0,
                max_restarts: self.config.max_restarts,
                last_failure: None,
                last_restart: None,
            },
        );
    }

    /// Notifies the supervisor that a worker has failed.
    ///
    /// The supervisor will attempt to restart the worker unless the restart
    /// limit has been reached.
    ///
    /// Returns the new worker status.
    pub fn report_failure(&self, worker_id: &str, reason: &str) -> FaultResult<WorkerStatus> {
        let new_status = {
            let mut workers = self.workers.write();
            let worker = workers.get_mut(worker_id).ok_or_else(|| {
                FaultToleranceError::SupervisorRestartFailed {
                    worker_id: worker_id.to_string(),
                    attempts: 0,
                }
            })?;

            worker.last_failure = Some(SystemTime::now());

            if worker.restart_count >= worker.max_restarts {
                worker.status = WorkerStatus::Exhausted;
                warn!(
                    "Worker {} permanently failed after {} restarts",
                    worker_id, worker.restart_count
                );
                WorkerStatus::Exhausted
            } else {
                worker.status = WorkerStatus::Restarting;
                worker.restart_count += 1;
                worker.last_restart = Some(SystemTime::now());
                WorkerStatus::Restarting
            }
        };

        // Record restart attempt
        let attempt = self
            .workers
            .read()
            .get(worker_id)
            .map(|w| w.restart_count)
            .unwrap_or(0);
        let record = RestartRecord {
            worker_id: worker_id.to_string(),
            attempt,
            reason: reason.to_string(),
            restarted_at: SystemTime::now(),
            success: new_status == WorkerStatus::Restarting,
        };
        let mut history = self.restart_history.write();
        if history.len() >= 10_000 {
            history.remove(0);
        }
        history.push(record);

        if new_status == WorkerStatus::Restarting {
            *self.total_restarts.write() += 1;
            info!("Restarting worker {} (attempt {})", worker_id, attempt);

            // If one-for-all strategy, mark siblings for restart too
            if self.config.one_for_all {
                let siblings: Vec<String> = self
                    .workers
                    .read()
                    .keys()
                    .filter(|k| k.as_str() != worker_id)
                    .cloned()
                    .collect();
                for sibling_id in siblings {
                    let mut workers = self.workers.write();
                    if let Some(sibling) = workers.get_mut(&sibling_id) {
                        if sibling.status == WorkerStatus::Running {
                            sibling.status = WorkerStatus::Restarting;
                            sibling.restart_count += 1;
                            sibling.last_restart = Some(SystemTime::now());
                        }
                    }
                }
            }
        }

        Ok(new_status)
    }

    /// Acknowledges that a worker has successfully restarted.
    pub fn acknowledge_restart(&self, worker_id: &str) -> FaultResult<()> {
        let mut workers = self.workers.write();
        let worker = workers.get_mut(worker_id).ok_or_else(|| {
            FaultToleranceError::SupervisorRestartFailed {
                worker_id: worker_id.to_string(),
                attempts: 0,
            }
        })?;
        worker.status = WorkerStatus::Running;
        info!("Worker {} successfully restarted", worker_id);
        Ok(())
    }

    /// Stops a worker intentionally.
    pub fn stop_worker(&self, worker_id: &str) -> FaultResult<()> {
        let mut workers = self.workers.write();
        let worker = workers.get_mut(worker_id).ok_or_else(|| {
            FaultToleranceError::SupervisorRestartFailed {
                worker_id: worker_id.to_string(),
                attempts: 0,
            }
        })?;
        worker.status = WorkerStatus::Stopped;
        info!("Worker {} stopped", worker_id);
        Ok(())
    }

    /// Returns the current status of a worker.
    pub fn worker_status(&self, worker_id: &str) -> Option<WorkerStatus> {
        self.workers.read().get(worker_id).map(|w| w.status.clone())
    }

    /// Returns all workers whose status matches the given status.
    pub fn workers_with_status(&self, status: &WorkerStatus) -> Vec<String> {
        self.workers
            .read()
            .values()
            .filter(|w| &w.status == status)
            .map(|w| w.worker_id.clone())
            .collect()
    }

    /// Returns supervisor statistics.
    pub fn stats(&self) -> SupervisorStats {
        let workers = self.workers.read();
        let running_workers = workers
            .values()
            .filter(|w| w.status == WorkerStatus::Running)
            .count();
        let exhausted_workers = workers
            .values()
            .filter(|w| w.status == WorkerStatus::Exhausted)
            .count();
        SupervisorStats {
            total_workers: workers.len(),
            running_workers,
            exhausted_workers,
            total_restarts: *self.total_restarts.read(),
            restart_history_len: self.restart_history.read().len(),
        }
    }

    /// Returns the full restart history.
    pub fn restart_history(&self) -> Vec<RestartRecord> {
        self.restart_history.read().clone()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── StreamHealthMonitor tests ────────────────────────────────────────────

    #[test]
    fn test_health_monitor_healthy_state() {
        let config = HealthMonitorConfig::default();
        let monitor = StreamHealthMonitor::new(config);
        monitor.record_metric("error_rate", 0.001);
        monitor.record_metric("latency_p99_ms", 50.0);
        monitor.record_metric("backpressure_ratio", 0.1);

        let snap = monitor.snapshot();
        assert_eq!(snap.status, StreamHealthStatus::Healthy);
        assert!(snap.active_alerts.is_empty());
    }

    #[test]
    fn test_health_monitor_warning_alert() {
        let config = HealthMonitorConfig::default();
        let monitor = StreamHealthMonitor::new(config);

        let alerts = monitor.record_metric("error_rate", 0.02);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, HealthAlertSeverity::Warning);

        let snap = monitor.snapshot();
        assert_eq!(snap.status, StreamHealthStatus::Degraded);
    }

    #[test]
    fn test_health_monitor_critical_alert() {
        let config = HealthMonitorConfig::default();
        let monitor = StreamHealthMonitor::new(config);

        let alerts = monitor.record_metric("error_rate", 0.10);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].severity, HealthAlertSeverity::Critical);

        let snap = monitor.snapshot();
        assert_eq!(snap.status, StreamHealthStatus::Critical);
    }

    #[test]
    fn test_health_monitor_recovery() {
        let config = HealthMonitorConfig::default();
        let monitor = StreamHealthMonitor::new(config);

        monitor.record_metric("error_rate", 0.10); // critical
        let snap = monitor.snapshot();
        assert_eq!(snap.status, StreamHealthStatus::Critical);

        monitor.record_metric("error_rate", 0.001); // recovered
        let snap = monitor.snapshot();
        assert!(snap.active_alerts.is_empty());
    }

    #[test]
    fn test_health_monitor_total_alerts_count() {
        let config = HealthMonitorConfig::default();
        let monitor = StreamHealthMonitor::new(config);
        monitor.record_metric("error_rate", 0.02);
        monitor.record_metric("latency_p99_ms", 200.0);
        assert_eq!(monitor.total_alerts_raised(), 2);
    }

    // ── BulkheadIsolator tests ───────────────────────────────────────────────

    #[test]
    fn test_bulkhead_acquire_and_release() {
        let mut config = BulkheadConfig::default();
        config.compartment_capacities.insert("test".to_string(), 2);
        let isolator = BulkheadIsolator::new(config);

        let p1 = isolator
            .acquire("test")
            .expect("first permit should succeed");
        let p2 = isolator
            .acquire("test")
            .expect("second permit should succeed");

        let result = isolator.acquire("test");
        assert!(
            matches!(result, Err(FaultToleranceError::BulkheadFull { .. })),
            "third permit should be rejected"
        );

        let stats = isolator
            .compartment_stats("test")
            .expect("stats should exist");
        assert_eq!(stats.active, 2);
        assert_eq!(stats.rejected, 1);

        drop(p1);
        drop(p2);

        let stats = isolator
            .compartment_stats("test")
            .expect("stats should exist");
        assert_eq!(stats.active, 0);
    }

    #[test]
    fn test_bulkhead_auto_creates_compartment() {
        let config = BulkheadConfig {
            compartment_capacities: HashMap::new(),
            default_capacity: 5,
        };
        let isolator = BulkheadIsolator::new(config);
        let permit = isolator
            .acquire("new-compartment")
            .expect("should succeed with default capacity");
        drop(permit);
    }

    #[test]
    fn test_bulkhead_different_compartments_isolated() {
        let mut config = BulkheadConfig::default();
        config.compartment_capacities.insert("a".to_string(), 1);
        config.compartment_capacities.insert("b".to_string(), 1);
        let isolator = BulkheadIsolator::new(config);

        let _pa = isolator.acquire("a").expect("a should succeed");
        // a is full
        let result_a = isolator.acquire("a");
        assert!(matches!(
            result_a,
            Err(FaultToleranceError::BulkheadFull { .. })
        ));

        // b should still be available
        let _pb = isolator.acquire("b").expect("b should be independent");
    }

    // ── StreamRetryPolicy tests ──────────────────────────────────────────────

    #[test]
    fn test_retry_policy_delay_increases() {
        let policy = StreamRetryPolicy {
            max_attempts: 5,
            initial_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(60),
            jitter: false,
        };
        let d0 = policy.delay_for_attempt(0);
        let d1 = policy.delay_for_attempt(1);
        let d2 = policy.delay_for_attempt(2);
        assert!(d0 < d1, "delay should increase");
        assert!(d1 < d2, "delay should increase");
    }

    #[test]
    fn test_retry_policy_max_delay_cap() {
        let policy = StreamRetryPolicy {
            max_attempts: 10,
            initial_delay: Duration::from_millis(100),
            backoff_multiplier: 10.0,
            max_delay: Duration::from_millis(500),
            jitter: false,
        };
        let d = policy.delay_for_attempt(5);
        assert!(
            d <= Duration::from_millis(500) + Duration::from_millis(10),
            "delay should not exceed max"
        );
    }

    #[test]
    fn test_retry_succeeds_on_first_attempt() {
        let policy = StreamRetryPolicy {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(1),
            jitter: false,
        };
        let result: FaultResult<i32> = policy.retry("test-op", || Ok::<i32, &str>(42));
        assert!(matches!(result, Ok(42)));
    }

    #[test]
    fn test_retry_exhausts_attempts() {
        let policy = StreamRetryPolicy {
            max_attempts: 2,
            initial_delay: Duration::from_millis(1),
            backoff_multiplier: 1.0,
            max_delay: Duration::from_millis(5),
            jitter: false,
        };
        let mut calls = 0u32;
        let result: FaultResult<i32> = policy.retry("always-fail", || {
            calls += 1;
            Err::<i32, &str>("always fails")
        });
        assert!(matches!(
            result,
            Err(FaultToleranceError::MaxRetriesExceeded { .. })
        ));
        // Called max_attempts + 1 = 3 times
        assert_eq!(calls, 3);
    }

    #[test]
    fn test_retry_succeeds_after_failures() {
        let policy = StreamRetryPolicy {
            max_attempts: 5,
            initial_delay: Duration::from_millis(1),
            backoff_multiplier: 1.0,
            max_delay: Duration::from_millis(10),
            jitter: false,
        };
        let mut calls = 0u32;
        let result: FaultResult<i32> = policy.retry("eventually-succeeds", || {
            calls += 1;
            if calls < 3 {
                Err::<i32, &str>("not yet")
            } else {
                Ok(99)
            }
        });
        assert!(matches!(result, Ok(99)));
        assert_eq!(calls, 3);
    }

    #[tokio::test]
    async fn test_retry_async_succeeds() {
        let policy = StreamRetryPolicy {
            max_attempts: 3,
            initial_delay: Duration::from_millis(1),
            backoff_multiplier: 1.0,
            max_delay: Duration::from_millis(5),
            jitter: false,
        };
        let calls = Arc::new(RwLock::new(0u32));
        let calls_clone = Arc::clone(&calls);
        let result: FaultResult<i32> = policy
            .retry_async("async-op", move || {
                let c = Arc::clone(&calls_clone);
                async move {
                    let mut lock = c.write();
                    *lock += 1;
                    let v = *lock;
                    drop(lock);
                    if v < 2 {
                        Err::<i32, &str>("not ready")
                    } else {
                        Ok(7)
                    }
                }
            })
            .await;
        assert!(matches!(result, Ok(7)));
    }

    // ── StreamSupervisor tests ───────────────────────────────────────────────

    #[test]
    fn test_supervisor_register_and_failure_restart() {
        let config = SupervisorConfig::default();
        let supervisor = StreamSupervisor::new(config);
        supervisor.register_worker("worker-1");

        let status = supervisor
            .report_failure("worker-1", "connection lost")
            .expect("should handle failure");
        assert_eq!(status, WorkerStatus::Restarting);

        supervisor
            .acknowledge_restart("worker-1")
            .expect("ack should succeed");
        assert_eq!(
            supervisor.worker_status("worker-1"),
            Some(WorkerStatus::Running)
        );
    }

    #[test]
    fn test_supervisor_exhausted_after_max_restarts() {
        let config = SupervisorConfig {
            max_restarts: 2,
            ..Default::default()
        };
        let supervisor = StreamSupervisor::new(config);
        supervisor.register_worker("worker-x");

        for _ in 0..2 {
            let status = supervisor
                .report_failure("worker-x", "crash")
                .expect("failure should be handled");
            if status == WorkerStatus::Restarting {
                supervisor.acknowledge_restart("worker-x").ok();
            }
        }

        let final_status = supervisor
            .report_failure("worker-x", "final crash")
            .expect("final failure should be handled");
        assert_eq!(final_status, WorkerStatus::Exhausted);

        let stats = supervisor.stats();
        assert_eq!(stats.exhausted_workers, 1);
    }

    #[test]
    fn test_supervisor_stop_worker() {
        let config = SupervisorConfig::default();
        let supervisor = StreamSupervisor::new(config);
        supervisor.register_worker("w1");
        supervisor.stop_worker("w1").expect("stop should succeed");
        assert_eq!(supervisor.worker_status("w1"), Some(WorkerStatus::Stopped));
    }

    #[test]
    fn test_supervisor_one_for_all() {
        let config = SupervisorConfig {
            max_restarts: 5,
            one_for_all: true,
            ..Default::default()
        };
        let supervisor = StreamSupervisor::new(config);
        supervisor.register_worker("w1");
        supervisor.register_worker("w2");
        supervisor.register_worker("w3");

        supervisor
            .report_failure("w1", "cascade test")
            .expect("failure should be handled");

        // w2 and w3 should also be restarting due to one-for-all
        let restarting = supervisor.workers_with_status(&WorkerStatus::Restarting);
        // w1 + at least w2 and w3
        assert!(
            restarting.len() >= 2,
            "siblings should also restart: {:?}",
            restarting
        );
    }

    #[test]
    fn test_supervisor_restart_history() {
        let config = SupervisorConfig::default();
        let supervisor = StreamSupervisor::new(config);
        supervisor.register_worker("wh");

        supervisor.report_failure("wh", "reason-1").ok();
        supervisor.acknowledge_restart("wh").ok();
        supervisor.report_failure("wh", "reason-2").ok();

        let history = supervisor.restart_history();
        assert!(history.len() >= 2);
        assert_eq!(history[0].reason, "reason-1");
    }

    #[test]
    fn test_supervisor_stats() {
        let config = SupervisorConfig::default();
        let supervisor = StreamSupervisor::new(config);
        supervisor.register_worker("s1");
        supervisor.register_worker("s2");

        supervisor.report_failure("s1", "err").ok();
        supervisor.acknowledge_restart("s1").ok();

        let stats = supervisor.stats();
        assert_eq!(stats.total_workers, 2);
        assert_eq!(stats.running_workers, 2); // s1 restarted, s2 never failed
        assert_eq!(stats.total_restarts, 1);
    }
}
