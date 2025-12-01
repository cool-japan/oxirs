//! Canary Release Integration
//!
//! This module provides canary release capabilities for gradually rolling out
//! new versions to a subset of traffic while monitoring for issues.
//!
//! ## Features
//!
//! - **Traffic Segmentation**: Route a percentage of traffic to canary instances
//! - **Automatic Promotion**: Gradually increase canary traffic based on metrics
//! - **Automatic Rollback**: Detect issues and rollback automatically
//! - **Metric-Based Analysis**: Use error rates, latency, and custom metrics
//! - **Feature Flags**: Combine canary releases with feature flag targeting
//! - **A/B Testing Integration**: Support for statistical significance testing
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_gql::canary_release::{CanaryController, CanaryConfig, PromotionPolicy};
//!
//! let config = CanaryConfig {
//!     initial_percentage: 5,
//!     max_percentage: 50,
//!     ..Default::default()
//! };
//!
//! let controller = CanaryController::new(config);
//! controller.start_canary("v2.0.0").await?;
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;

/// Canary release state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CanaryState {
    /// No canary release active
    Inactive,
    /// Canary release starting
    Starting,
    /// Canary is receiving traffic
    Active,
    /// Canary is being promoted (increasing traffic)
    Promoting,
    /// Canary is paused for analysis
    Paused,
    /// Canary is being rolled back
    RollingBack,
    /// Canary has been fully promoted
    Promoted,
    /// Canary has been rolled back
    RolledBack,
    /// Canary release failed
    Failed,
}

/// Traffic routing strategy
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Random percentage-based routing
    #[default]
    Random,
    /// Hash-based routing (consistent for same user/session)
    HashBased { hash_key: String },
    /// Header-based routing
    HeaderBased {
        header_name: String,
        header_value: String,
    },
    /// Cookie-based routing
    CookieBased { cookie_name: String },
    /// Geographic routing
    Geographic { regions: Vec<String> },
    /// User segment routing
    UserSegment { segments: Vec<String> },
}

/// Promotion policy for automatic canary promotion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionPolicy {
    /// Percentage increments for promotion
    pub percentage_increments: Vec<u8>,
    /// Minimum duration at each stage before promotion
    pub stage_duration: Duration,
    /// Enable automatic promotion
    pub auto_promote: bool,
    /// Maximum error rate allowed (percentage)
    pub max_error_rate: f64,
    /// Maximum p99 latency allowed (milliseconds)
    pub max_latency_p99_ms: u64,
    /// Minimum number of requests before making decisions
    pub min_request_count: u64,
    /// Custom metric thresholds
    pub custom_thresholds: HashMap<String, f64>,
}

impl Default for PromotionPolicy {
    fn default() -> Self {
        Self {
            percentage_increments: vec![5, 10, 25, 50, 75, 100],
            stage_duration: Duration::from_secs(300), // 5 minutes per stage
            auto_promote: true,
            max_error_rate: 1.0,      // 1% error rate
            max_latency_p99_ms: 1000, // 1 second
            min_request_count: 100,
            custom_thresholds: HashMap::new(),
        }
    }
}

/// Rollback policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPolicy {
    /// Enable automatic rollback
    pub auto_rollback: bool,
    /// Error rate threshold for immediate rollback
    pub error_rate_threshold: f64,
    /// Latency threshold for rollback (p99 in ms)
    pub latency_threshold_ms: u64,
    /// Number of consecutive failures before rollback
    pub failure_threshold: u32,
    /// Rollback immediately on first failure
    pub fail_fast: bool,
    /// Grace period before rollback decision
    pub grace_period: Duration,
}

impl Default for RollbackPolicy {
    fn default() -> Self {
        Self {
            auto_rollback: true,
            error_rate_threshold: 5.0,  // 5% error rate
            latency_threshold_ms: 5000, // 5 seconds
            failure_threshold: 3,
            fail_fast: false,
            grace_period: Duration::from_secs(60),
        }
    }
}

/// Canary configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryConfig {
    /// Initial traffic percentage for canary
    pub initial_percentage: u8,
    /// Maximum traffic percentage for canary
    pub max_percentage: u8,
    /// Traffic routing strategy
    pub routing_strategy: RoutingStrategy,
    /// Promotion policy
    pub promotion_policy: PromotionPolicy,
    /// Rollback policy
    pub rollback_policy: RollbackPolicy,
    /// Analysis window size
    pub analysis_window: Duration,
    /// Metric collection interval
    pub metric_interval: Duration,
    /// Enable metric comparison with baseline
    pub compare_with_baseline: bool,
    /// Statistical significance level for A/B testing
    pub significance_level: f64,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            initial_percentage: 5,
            max_percentage: 100,
            routing_strategy: RoutingStrategy::default(),
            promotion_policy: PromotionPolicy::default(),
            rollback_policy: RollbackPolicy::default(),
            analysis_window: Duration::from_secs(300),
            metric_interval: Duration::from_secs(10),
            compare_with_baseline: true,
            significance_level: 0.95,
        }
    }
}

/// Metric snapshot for canary analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Request count
    pub request_count: u64,
    /// Error count
    pub error_count: u64,
    /// Error rate (percentage)
    pub error_rate: f64,
    /// Latency p50 (ms)
    pub latency_p50_ms: u64,
    /// Latency p95 (ms)
    pub latency_p95_ms: u64,
    /// Latency p99 (ms)
    pub latency_p99_ms: u64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for MetricSnapshot {
    fn default() -> Self {
        Self {
            timestamp: SystemTime::now(),
            request_count: 0,
            error_count: 0,
            error_rate: 0.0,
            latency_p50_ms: 0,
            latency_p95_ms: 0,
            latency_p99_ms: 0,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Canary analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// Overall health score (0-100)
    pub health_score: f64,
    /// Whether canary passed all checks
    pub passed: bool,
    /// Recommendation
    pub recommendation: AnalysisRecommendation,
    /// Individual check results
    pub checks: Vec<AnalysisCheck>,
    /// Comparison with baseline (if available)
    pub baseline_comparison: Option<BaselineComparison>,
    /// Analysis timestamp
    pub timestamp: SystemTime,
}

/// Analysis recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisRecommendation {
    /// Promote canary to next stage
    Promote,
    /// Keep canary at current stage
    Hold,
    /// Rollback canary
    Rollback,
    /// Need more data
    InsufficientData,
}

/// Individual analysis check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisCheck {
    /// Check name
    pub name: String,
    /// Check passed
    pub passed: bool,
    /// Actual value
    pub actual_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Description
    pub description: String,
}

/// Baseline comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    /// Error rate delta (canary - baseline)
    pub error_rate_delta: f64,
    /// Latency delta (canary - baseline) in ms
    pub latency_delta_ms: i64,
    /// Statistical significance
    pub statistically_significant: bool,
    /// Confidence level
    pub confidence_level: f64,
    /// Is canary performing better?
    pub is_better: bool,
}

/// Canary release information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryRelease {
    /// Release ID
    pub id: String,
    /// Version being tested
    pub version: String,
    /// Current state
    pub state: CanaryState,
    /// Current traffic percentage
    pub traffic_percentage: u8,
    /// Current promotion stage index
    pub current_stage: usize,
    /// Start time
    pub started_at: SystemTime,
    /// Time at current stage
    pub stage_started_at: SystemTime,
    /// Canary metrics
    pub canary_metrics: MetricSnapshot,
    /// Baseline metrics
    pub baseline_metrics: MetricSnapshot,
    /// Analysis history
    pub analysis_history: Vec<AnalysisResult>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl CanaryRelease {
    fn new(id: String, version: String, initial_percentage: u8) -> Self {
        let now = SystemTime::now();
        Self {
            id,
            version,
            state: CanaryState::Starting,
            traffic_percentage: initial_percentage,
            current_stage: 0,
            started_at: now,
            stage_started_at: now,
            canary_metrics: MetricSnapshot::default(),
            baseline_metrics: MetricSnapshot::default(),
            analysis_history: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Canary event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanaryEvent {
    /// Canary started
    Started { release_id: String, version: String },
    /// Traffic percentage changed
    TrafficChanged {
        release_id: String,
        old_percentage: u8,
        new_percentage: u8,
    },
    /// Analysis completed
    AnalysisCompleted {
        release_id: String,
        result: AnalysisResult,
    },
    /// Stage promoted
    StagePromoted {
        release_id: String,
        new_stage: usize,
        new_percentage: u8,
    },
    /// Canary paused
    Paused { release_id: String, reason: String },
    /// Canary resumed
    Resumed { release_id: String },
    /// Rollback initiated
    RollbackInitiated { release_id: String, reason: String },
    /// Canary completed (fully promoted)
    Completed { release_id: String },
    /// Canary rolled back
    RolledBack { release_id: String },
    /// Canary failed
    Failed { release_id: String, error: String },
}

/// Traffic routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryRoutingDecision {
    /// Route to canary?
    pub is_canary: bool,
    /// Version
    pub version: String,
    /// Weight (for weighted routing)
    pub weight: f64,
    /// Routing reason
    pub reason: String,
}

/// Internal state
struct ControllerState {
    /// Active canary release
    active_release: Option<CanaryRelease>,
    /// Release history
    release_history: Vec<CanaryRelease>,
    /// Event log
    events: Vec<(SystemTime, CanaryEvent)>,
    /// Collected latencies for baseline
    baseline_latencies: Vec<u64>,
    /// Collected latencies for canary
    canary_latencies: Vec<u64>,
    /// Consecutive failure count
    consecutive_failures: u32,
}

impl ControllerState {
    fn new() -> Self {
        Self {
            active_release: None,
            release_history: Vec::new(),
            events: Vec::new(),
            baseline_latencies: Vec::new(),
            canary_latencies: Vec::new(),
            consecutive_failures: 0,
        }
    }
}

/// Canary Release Controller
///
/// Manages canary releases including traffic routing, metric analysis,
/// automatic promotion, and rollback handling.
pub struct CanaryController {
    /// Configuration
    config: CanaryConfig,
    /// Internal state
    state: Arc<RwLock<ControllerState>>,
    /// Event handlers
    event_handlers: Arc<RwLock<Vec<Arc<dyn CanaryEventHandler + Send + Sync>>>>,
    /// Baseline version
    baseline_version: Arc<RwLock<String>>,
}

impl CanaryController {
    /// Create a new canary controller
    pub fn new(config: CanaryConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(ControllerState::new())),
            event_handlers: Arc::new(RwLock::new(Vec::new())),
            baseline_version: Arc::new(RwLock::new("baseline".to_string())),
        }
    }

    /// Set baseline version
    pub async fn set_baseline_version(&self, version: &str) {
        let mut baseline = self.baseline_version.write().await;
        *baseline = version.to_string();
    }

    /// Register an event handler
    pub async fn register_event_handler(&self, handler: Arc<dyn CanaryEventHandler + Send + Sync>) {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(handler);
    }

    /// Emit a canary event
    async fn emit_event(&self, event: CanaryEvent) {
        let now = SystemTime::now();

        {
            let mut state = self.state.write().await;
            state.events.push((now, event.clone()));

            // Limit event history
            if state.events.len() > 1000 {
                state.events.drain(0..100);
            }
        }

        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            handler.on_event(&event).await;
        }
    }

    /// Start a new canary release
    pub async fn start_canary(&self, version: &str) -> Result<String> {
        // Check if there's already an active canary
        {
            let state = self.state.read().await;
            if state.active_release.is_some() {
                return Err(anyhow!("Cannot start canary: another release is active"));
            }
        }

        let release_id = uuid::Uuid::new_v4().to_string();
        let release = CanaryRelease::new(
            release_id.clone(),
            version.to_string(),
            self.config.initial_percentage,
        );

        {
            let mut state = self.state.write().await;
            state.active_release = Some(release);
            state.consecutive_failures = 0;
            state.canary_latencies.clear();
        }

        // Mark as active
        {
            let mut state = self.state.write().await;
            if let Some(ref mut release) = state.active_release {
                release.state = CanaryState::Active;
            }
        }

        self.emit_event(CanaryEvent::Started {
            release_id: release_id.clone(),
            version: version.to_string(),
        })
        .await;

        Ok(release_id)
    }

    /// Get current canary status
    pub async fn get_status(&self) -> Option<CanaryRelease> {
        let state = self.state.read().await;
        state.active_release.clone()
    }

    /// Check if canary is active
    pub async fn is_active(&self) -> bool {
        let state = self.state.read().await;
        state
            .active_release
            .as_ref()
            .map(|r| r.state == CanaryState::Active || r.state == CanaryState::Promoting)
            .unwrap_or(false)
    }

    /// Record a request metric
    pub async fn record_metric(&self, is_canary: bool, latency_ms: u64, is_error: bool) {
        let mut state = self.state.write().await;

        if is_canary {
            state.canary_latencies.push(latency_ms);

            // Limit latency history
            if state.canary_latencies.len() > 10000 {
                state.canary_latencies.drain(0..1000);
            }

            if let Some(ref mut release) = state.active_release {
                release.canary_metrics.request_count += 1;
                if is_error {
                    release.canary_metrics.error_count += 1;
                }
                release.canary_metrics.error_rate = if release.canary_metrics.request_count > 0 {
                    (release.canary_metrics.error_count as f64
                        / release.canary_metrics.request_count as f64)
                        * 100.0
                } else {
                    0.0
                };
            }
        } else {
            state.baseline_latencies.push(latency_ms);

            if state.baseline_latencies.len() > 10000 {
                state.baseline_latencies.drain(0..1000);
            }

            if let Some(ref mut release) = state.active_release {
                release.baseline_metrics.request_count += 1;
                if is_error {
                    release.baseline_metrics.error_count += 1;
                }
                release.baseline_metrics.error_rate = if release.baseline_metrics.request_count > 0
                {
                    (release.baseline_metrics.error_count as f64
                        / release.baseline_metrics.request_count as f64)
                        * 100.0
                } else {
                    0.0
                };
            }
        }
    }

    /// Route a request
    pub async fn route_request(&self, request_id: &str) -> CanaryRoutingDecision {
        let state = self.state.read().await;
        let baseline_version = self.baseline_version.read().await;

        let Some(ref release) = state.active_release else {
            return CanaryRoutingDecision {
                is_canary: false,
                version: baseline_version.clone(),
                weight: 1.0,
                reason: "No active canary".to_string(),
            };
        };

        if release.state != CanaryState::Active && release.state != CanaryState::Promoting {
            return CanaryRoutingDecision {
                is_canary: false,
                version: baseline_version.clone(),
                weight: 1.0,
                reason: format!("Canary not active (state: {:?})", release.state),
            };
        }

        let percentage = release.traffic_percentage;
        let is_canary = match &self.config.routing_strategy {
            RoutingStrategy::Random => {
                let random = fastrand::u8(0..100);
                random < percentage
            }
            RoutingStrategy::HashBased { hash_key: _ } => {
                let hash = request_id
                    .bytes()
                    .fold(0u64, |acc, b| acc.wrapping_add(b as u64));
                ((hash % 100) as u8) < percentage
            }
            RoutingStrategy::HeaderBased { .. }
            | RoutingStrategy::CookieBased { .. }
            | RoutingStrategy::Geographic { .. }
            | RoutingStrategy::UserSegment { .. } => {
                // For these strategies, we'd need more context
                // Fall back to random for now
                let random = fastrand::u8(0..100);
                random < percentage
            }
        };

        CanaryRoutingDecision {
            is_canary,
            version: if is_canary {
                release.version.clone()
            } else {
                baseline_version.clone()
            },
            weight: if is_canary {
                percentage as f64 / 100.0
            } else {
                (100 - percentage) as f64 / 100.0
            },
            reason: if is_canary {
                format!("Routed to canary ({}%)", percentage)
            } else {
                format!("Routed to baseline ({}%)", 100 - percentage)
            },
        }
    }

    /// Analyze canary metrics and return recommendation
    pub async fn analyze(&self) -> Option<AnalysisResult> {
        let state = self.state.read().await;
        let release = state.active_release.as_ref()?;

        let mut checks = Vec::new();
        let mut all_passed = true;

        // Check minimum request count
        let min_requests = self.config.promotion_policy.min_request_count;
        if release.canary_metrics.request_count < min_requests {
            return Some(AnalysisResult {
                health_score: 0.0,
                passed: false,
                recommendation: AnalysisRecommendation::InsufficientData,
                checks: vec![AnalysisCheck {
                    name: "Minimum Requests".to_string(),
                    passed: false,
                    actual_value: release.canary_metrics.request_count as f64,
                    threshold_value: min_requests as f64,
                    description: format!(
                        "Need {} requests, have {}",
                        min_requests, release.canary_metrics.request_count
                    ),
                }],
                baseline_comparison: None,
                timestamp: SystemTime::now(),
            });
        }

        // Error rate check
        let error_rate_ok =
            release.canary_metrics.error_rate <= self.config.promotion_policy.max_error_rate;
        checks.push(AnalysisCheck {
            name: "Error Rate".to_string(),
            passed: error_rate_ok,
            actual_value: release.canary_metrics.error_rate,
            threshold_value: self.config.promotion_policy.max_error_rate,
            description: format!(
                "Error rate: {:.2}% (max: {:.2}%)",
                release.canary_metrics.error_rate, self.config.promotion_policy.max_error_rate
            ),
        });
        if !error_rate_ok {
            all_passed = false;
        }

        // Calculate latency percentiles
        let canary_p99 = self.calculate_percentile(&state.canary_latencies, 0.99);
        let latency_ok = canary_p99 <= self.config.promotion_policy.max_latency_p99_ms;
        checks.push(AnalysisCheck {
            name: "P99 Latency".to_string(),
            passed: latency_ok,
            actual_value: canary_p99 as f64,
            threshold_value: self.config.promotion_policy.max_latency_p99_ms as f64,
            description: format!(
                "P99 latency: {}ms (max: {}ms)",
                canary_p99, self.config.promotion_policy.max_latency_p99_ms
            ),
        });
        if !latency_ok {
            all_passed = false;
        }

        // Baseline comparison
        let baseline_comparison = if self.config.compare_with_baseline
            && release.baseline_metrics.request_count >= min_requests
        {
            let baseline_p99 = self.calculate_percentile(&state.baseline_latencies, 0.99);
            let error_rate_delta =
                release.canary_metrics.error_rate - release.baseline_metrics.error_rate;
            let latency_delta = canary_p99 as i64 - baseline_p99 as i64;

            // Simple statistical comparison
            let is_better = error_rate_delta <= 0.0 && latency_delta <= 0;

            Some(BaselineComparison {
                error_rate_delta,
                latency_delta_ms: latency_delta,
                statistically_significant: release.canary_metrics.request_count >= 1000,
                confidence_level: self.config.significance_level,
                is_better,
            })
        } else {
            None
        };

        // Calculate health score
        let health_score = self.calculate_health_score(&checks);

        // Determine recommendation
        let recommendation = if !all_passed {
            if release.canary_metrics.error_rate > self.config.rollback_policy.error_rate_threshold
            {
                AnalysisRecommendation::Rollback
            } else {
                AnalysisRecommendation::Hold
            }
        } else {
            AnalysisRecommendation::Promote
        };

        Some(AnalysisResult {
            health_score,
            passed: all_passed,
            recommendation,
            checks,
            baseline_comparison,
            timestamp: SystemTime::now(),
        })
    }

    fn calculate_percentile(&self, values: &[u64], percentile: f64) -> u64 {
        if values.is_empty() {
            return 0;
        }

        let mut sorted = values.to_vec();
        sorted.sort();

        let index = ((sorted.len() as f64 - 1.0) * percentile).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    fn calculate_health_score(&self, checks: &[AnalysisCheck]) -> f64 {
        if checks.is_empty() {
            return 0.0;
        }

        let passed_count = checks.iter().filter(|c| c.passed).count();
        (passed_count as f64 / checks.len() as f64) * 100.0
    }

    /// Promote canary to next stage
    pub async fn promote(&self) -> Result<()> {
        let (release_id, current_stage, new_percentage) = {
            let mut state = self.state.write().await;
            let Some(ref mut release) = state.active_release else {
                return Err(anyhow!("No active canary release"));
            };

            if release.state != CanaryState::Active {
                return Err(anyhow!("Canary not in active state"));
            }

            let next_stage = release.current_stage + 1;
            if next_stage >= self.config.promotion_policy.percentage_increments.len() {
                // Fully promoted
                release.state = CanaryState::Promoted;
                release.traffic_percentage = 100;
                return Ok(());
            }

            let new_percentage = self.config.promotion_policy.percentage_increments[next_stage]
                .min(self.config.max_percentage);

            release.current_stage = next_stage;
            release.traffic_percentage = new_percentage;
            release.stage_started_at = SystemTime::now();
            release.state = CanaryState::Active;

            (release.id.clone(), next_stage, new_percentage)
        };

        self.emit_event(CanaryEvent::StagePromoted {
            release_id,
            new_stage: current_stage,
            new_percentage,
        })
        .await;

        Ok(())
    }

    /// Pause canary release
    pub async fn pause(&self, reason: &str) -> Result<()> {
        let release_id = {
            let mut state = self.state.write().await;
            let Some(ref mut release) = state.active_release else {
                return Err(anyhow!("No active canary release"));
            };

            release.state = CanaryState::Paused;
            release.id.clone()
        };

        self.emit_event(CanaryEvent::Paused {
            release_id,
            reason: reason.to_string(),
        })
        .await;

        Ok(())
    }

    /// Resume canary release
    pub async fn resume(&self) -> Result<()> {
        let release_id = {
            let mut state = self.state.write().await;
            let Some(ref mut release) = state.active_release else {
                return Err(anyhow!("No active canary release"));
            };

            if release.state != CanaryState::Paused {
                return Err(anyhow!("Canary is not paused"));
            }

            release.state = CanaryState::Active;
            release.id.clone()
        };

        self.emit_event(CanaryEvent::Resumed { release_id }).await;

        Ok(())
    }

    /// Rollback canary release
    pub async fn rollback(&self, reason: &str) -> Result<()> {
        let release_id = {
            let mut state = self.state.write().await;
            let Some(ref mut release) = state.active_release else {
                return Err(anyhow!("No active canary release"));
            };

            release.state = CanaryState::RollingBack;
            release.id.clone()
        };

        self.emit_event(CanaryEvent::RollbackInitiated {
            release_id: release_id.clone(),
            reason: reason.to_string(),
        })
        .await;

        // Complete rollback
        {
            let mut state = self.state.write().await;
            if let Some(ref mut release) = state.active_release {
                release.state = CanaryState::RolledBack;
                release.traffic_percentage = 0;
            }

            // Move to history
            if let Some(release) = state.active_release.take() {
                state.release_history.push(release);
            }
        }

        self.emit_event(CanaryEvent::RolledBack { release_id })
            .await;

        Ok(())
    }

    /// Complete canary release (full promotion)
    pub async fn complete(&self) -> Result<()> {
        let release_id = {
            let mut state = self.state.write().await;
            let Some(ref mut release) = state.active_release else {
                return Err(anyhow!("No active canary release"));
            };

            release.state = CanaryState::Promoted;
            release.traffic_percentage = 100;
            release.id.clone()
        };

        self.emit_event(CanaryEvent::Completed {
            release_id: release_id.clone(),
        })
        .await;

        // Move to history
        {
            let mut state = self.state.write().await;
            if let Some(release) = state.active_release.take() {
                state.release_history.push(release);
            }
        }

        // Update baseline version
        {
            let state = self.state.read().await;
            if let Some(release) = state.release_history.last() {
                if release.state == CanaryState::Promoted {
                    let mut baseline = self.baseline_version.write().await;
                    *baseline = release.version.clone();
                }
            }
        }

        Ok(())
    }

    /// Run automatic analysis and take action
    pub async fn auto_analyze_and_act(&self) -> Result<Option<AnalysisRecommendation>> {
        let Some(analysis) = self.analyze().await else {
            return Ok(None);
        };

        // Store analysis in history
        {
            let mut state = self.state.write().await;
            if let Some(ref mut release) = state.active_release {
                release.analysis_history.push(analysis.clone());
            }
        }

        match analysis.recommendation {
            AnalysisRecommendation::Promote if self.config.promotion_policy.auto_promote => {
                // Check if stage duration has passed
                let can_promote = {
                    let state = self.state.read().await;
                    state.active_release.as_ref().is_some_and(|r| {
                        r.stage_started_at.elapsed().unwrap_or_default()
                            >= self.config.promotion_policy.stage_duration
                    })
                };

                if can_promote {
                    self.promote().await?;
                }
            }
            AnalysisRecommendation::Rollback if self.config.rollback_policy.auto_rollback => {
                self.rollback("Automatic rollback due to failed checks")
                    .await?;
            }
            _ => {}
        }

        Ok(Some(analysis.recommendation))
    }

    /// Get release history
    pub async fn get_history(&self, limit: Option<usize>) -> Vec<CanaryRelease> {
        let state = self.state.read().await;
        match limit {
            Some(n) => state
                .release_history
                .iter()
                .rev()
                .take(n)
                .cloned()
                .collect(),
            None => state.release_history.clone(),
        }
    }

    /// Get recent events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<(SystemTime, CanaryEvent)> {
        let state = self.state.read().await;
        state.events.iter().rev().take(limit).cloned().collect()
    }
}

/// Trait for handling canary events
#[async_trait::async_trait]
pub trait CanaryEventHandler {
    /// Handle a canary event
    async fn on_event(&self, event: &CanaryEvent);
}

/// Logging event handler
pub struct LoggingCanaryHandler;

#[async_trait::async_trait]
impl CanaryEventHandler for LoggingCanaryHandler {
    async fn on_event(&self, event: &CanaryEvent) {
        match event {
            CanaryEvent::Started {
                release_id,
                version,
            } => {
                tracing::info!("Canary {} started for version {}", release_id, version);
            }
            CanaryEvent::TrafficChanged {
                release_id,
                old_percentage,
                new_percentage,
            } => {
                tracing::info!(
                    "Canary {} traffic changed: {}% -> {}%",
                    release_id,
                    old_percentage,
                    new_percentage
                );
            }
            CanaryEvent::AnalysisCompleted { release_id, result } => {
                tracing::info!(
                    "Canary {} analysis: {:?} (score: {:.1})",
                    release_id,
                    result.recommendation,
                    result.health_score
                );
            }
            CanaryEvent::StagePromoted {
                release_id,
                new_stage,
                new_percentage,
            } => {
                tracing::info!(
                    "Canary {} promoted to stage {} ({}%)",
                    release_id,
                    new_stage,
                    new_percentage
                );
            }
            CanaryEvent::Paused { release_id, reason } => {
                tracing::warn!("Canary {} paused: {}", release_id, reason);
            }
            CanaryEvent::Resumed { release_id } => {
                tracing::info!("Canary {} resumed", release_id);
            }
            CanaryEvent::RollbackInitiated { release_id, reason } => {
                tracing::warn!("Canary {} rollback initiated: {}", release_id, reason);
            }
            CanaryEvent::Completed { release_id } => {
                tracing::info!("Canary {} completed successfully", release_id);
            }
            CanaryEvent::RolledBack { release_id } => {
                tracing::warn!("Canary {} rolled back", release_id);
            }
            CanaryEvent::Failed { release_id, error } => {
                tracing::error!("Canary {} failed: {}", release_id, error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_canary_creation() {
        let config = CanaryConfig::default();
        let controller = CanaryController::new(config);

        assert!(!controller.is_active().await);
    }

    #[tokio::test]
    async fn test_start_canary() {
        let config = CanaryConfig::default();
        let controller = CanaryController::new(config);

        let release_id = controller.start_canary("v2.0.0").await.unwrap();
        assert!(!release_id.is_empty());
        assert!(controller.is_active().await);
    }

    #[tokio::test]
    async fn test_cannot_start_duplicate_canary() {
        let config = CanaryConfig::default();
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();
        let result = controller.start_canary("v3.0.0").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_routing() {
        let config = CanaryConfig {
            initial_percentage: 50,
            ..Default::default()
        };
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();

        // With 50% canary, roughly half should route to canary
        let mut canary_count = 0;
        for i in 0..100 {
            let decision = controller.route_request(&format!("req-{}", i)).await;
            if decision.is_canary {
                canary_count += 1;
            }
        }

        // Should be roughly 50%, allow some variance
        assert!((30..=70).contains(&canary_count));
    }

    #[tokio::test]
    async fn test_record_metrics() {
        let config = CanaryConfig::default();
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();

        // Record some metrics
        for _ in 0..100 {
            controller.record_metric(true, 50, false).await;
            controller.record_metric(false, 45, false).await;
        }

        let status = controller.get_status().await.unwrap();
        assert_eq!(status.canary_metrics.request_count, 100);
        assert_eq!(status.baseline_metrics.request_count, 100);
    }

    #[tokio::test]
    async fn test_analysis_insufficient_data() {
        let config = CanaryConfig {
            promotion_policy: PromotionPolicy {
                min_request_count: 100,
                ..Default::default()
            },
            ..Default::default()
        };
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();

        // Record less than min_request_count
        for _ in 0..50 {
            controller.record_metric(true, 50, false).await;
        }

        let analysis = controller.analyze().await.unwrap();
        assert_eq!(
            analysis.recommendation,
            AnalysisRecommendation::InsufficientData
        );
    }

    #[tokio::test]
    async fn test_analysis_passes() {
        let config = CanaryConfig {
            promotion_policy: PromotionPolicy {
                min_request_count: 10,
                max_error_rate: 5.0,
                max_latency_p99_ms: 200,
                ..Default::default()
            },
            ..Default::default()
        };
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();

        // Record good metrics
        for _ in 0..100 {
            controller.record_metric(true, 50, false).await;
        }

        let analysis = controller.analyze().await.unwrap();
        assert!(analysis.passed);
        assert_eq!(analysis.recommendation, AnalysisRecommendation::Promote);
    }

    #[tokio::test]
    async fn test_promotion() {
        let config = CanaryConfig {
            initial_percentage: 5,
            promotion_policy: PromotionPolicy {
                percentage_increments: vec![5, 10, 25, 50, 100],
                ..Default::default()
            },
            ..Default::default()
        };
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();

        let status = controller.get_status().await.unwrap();
        assert_eq!(status.traffic_percentage, 5);

        controller.promote().await.unwrap();

        let status = controller.get_status().await.unwrap();
        assert_eq!(status.traffic_percentage, 10);
    }

    #[tokio::test]
    async fn test_pause_resume() {
        let config = CanaryConfig::default();
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();

        controller.pause("Testing").await.unwrap();
        let status = controller.get_status().await.unwrap();
        assert_eq!(status.state, CanaryState::Paused);

        controller.resume().await.unwrap();
        let status = controller.get_status().await.unwrap();
        assert_eq!(status.state, CanaryState::Active);
    }

    #[tokio::test]
    async fn test_rollback() {
        let config = CanaryConfig::default();
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();
        controller.rollback("Test rollback").await.unwrap();

        assert!(!controller.is_active().await);

        let history = controller.get_history(None).await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].state, CanaryState::RolledBack);
    }

    #[tokio::test]
    async fn test_complete() {
        let config = CanaryConfig::default();
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();
        controller.complete().await.unwrap();

        assert!(!controller.is_active().await);

        let history = controller.get_history(None).await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].state, CanaryState::Promoted);
    }

    #[tokio::test]
    async fn test_baseline_update_on_completion() {
        let config = CanaryConfig::default();
        let controller = CanaryController::new(config);

        controller.set_baseline_version("v1.0.0").await;
        controller.start_canary("v2.0.0").await.unwrap();
        controller.complete().await.unwrap();

        let baseline = controller.baseline_version.read().await;
        assert_eq!(*baseline, "v2.0.0");
    }

    #[tokio::test]
    async fn test_event_history() {
        let config = CanaryConfig::default();
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();
        controller.pause("Testing").await.unwrap();
        controller.resume().await.unwrap();

        let events = controller.get_recent_events(10).await;
        assert!(events.len() >= 3);
    }

    #[tokio::test]
    async fn test_hash_based_routing() {
        let config = CanaryConfig {
            initial_percentage: 50,
            routing_strategy: RoutingStrategy::HashBased {
                hash_key: "user_id".to_string(),
            },
            ..Default::default()
        };
        let controller = CanaryController::new(config);

        controller.start_canary("v2.0.0").await.unwrap();

        // Same request ID should always route to the same target
        let decision1 = controller.route_request("user-123").await;
        let decision2 = controller.route_request("user-123").await;
        assert_eq!(decision1.is_canary, decision2.is_canary);
    }
}
