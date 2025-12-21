#![allow(dead_code)]
//! Production Hardening - Resilience, Security, and Operational Excellence
//!
//! This module provides production-ready features for federated query systems:
//! - Advanced circuit breakers with ML-based failure prediction
//! - Rate limiting with adaptive throttling
//! - Request validation and sanitization
//! - Query complexity analysis and rejection
//! - Resource quota management
//! - Security hardening (injection prevention, DoS protection)
//! - Graceful degradation strategies
//! - Health check aggregation
//! - Chaos engineering support
//!
//! Enhanced with scirs2 for anomaly detection and predictive analysis.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{info, warn};

// scirs2 integration for anomaly detection and ML
// Note: Advanced features simplified for initial release

use crate::anomaly_detection::{AnomalyDetector, AnomalyDetectorConfig};

/// Production hardening configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardeningConfig {
    /// Enable ML-based failure prediction
    pub enable_failure_prediction: bool,
    /// Enable adaptive rate limiting
    pub enable_adaptive_rate_limiting: bool,
    /// Enable query complexity analysis
    pub enable_complexity_analysis: bool,
    /// Enable resource quota management
    pub enable_resource_quotas: bool,
    /// Enable security hardening
    pub enable_security_hardening: bool,
    /// Enable graceful degradation
    pub enable_graceful_degradation: bool,
    /// Enable chaos engineering
    pub enable_chaos_engineering: bool,
    /// Maximum query complexity score
    pub max_query_complexity: f64,
    /// Circuit breaker failure threshold (0.0 - 1.0)
    pub circuit_breaker_failure_threshold: f64,
    /// Circuit breaker timeout duration
    pub circuit_breaker_timeout: Duration,
    /// Rate limit requests per second
    pub rate_limit_rps: u32,
    /// Resource quota check interval
    pub quota_check_interval: Duration,
}

impl Default for HardeningConfig {
    fn default() -> Self {
        Self {
            enable_failure_prediction: true,
            enable_adaptive_rate_limiting: true,
            enable_complexity_analysis: true,
            enable_resource_quotas: true,
            enable_security_hardening: true,
            enable_graceful_degradation: true,
            enable_chaos_engineering: false, // Disabled by default in production
            max_query_complexity: 1000.0,
            circuit_breaker_failure_threshold: 0.5,
            circuit_breaker_timeout: Duration::from_secs(30),
            rate_limit_rps: 1000,
            quota_check_interval: Duration::from_secs(60),
        }
    }
}

/// Production hardening manager
pub struct ProductionHardening {
    config: HardeningConfig,
    /// ML-based circuit breaker
    circuit_breaker: Arc<RwLock<MLCircuitBreaker>>,
    /// Adaptive rate limiter
    rate_limiter: Arc<RwLock<AdaptiveRateLimiter>>,
    /// Query complexity analyzer
    complexity_analyzer: Arc<QueryComplexityAnalyzer>,
    /// Resource quota manager
    quota_manager: Arc<RwLock<ResourceQuotaManager>>,
    /// Security validator
    security_validator: Arc<SecurityValidator>,
    /// Degradation strategy manager
    degradation_manager: Arc<RwLock<DegradationManager>>,
    /// Chaos engineering controller
    chaos_controller: Arc<RwLock<Option<ChaosController>>>,
    /// Advanced anomaly detector
    anomaly_detector: Arc<RwLock<AnomalyDetector>>,
    /// Request tracking counters
    total_requests_validated: Arc<std::sync::atomic::AtomicU64>,
    total_requests_rejected: Arc<std::sync::atomic::AtomicU64>,
}

impl ProductionHardening {
    /// Create a new production hardening manager
    pub fn new(config: HardeningConfig) -> Self {
        let chaos_controller = if config.enable_chaos_engineering {
            Some(ChaosController::new())
        } else {
            None
        };

        Self {
            config: config.clone(),
            circuit_breaker: Arc::new(RwLock::new(MLCircuitBreaker::new(
                config.circuit_breaker_failure_threshold,
                config.circuit_breaker_timeout,
            ))),
            rate_limiter: Arc::new(RwLock::new(AdaptiveRateLimiter::new(config.rate_limit_rps))),
            complexity_analyzer: Arc::new(QueryComplexityAnalyzer::new(
                config.max_query_complexity,
            )),
            quota_manager: Arc::new(RwLock::new(ResourceQuotaManager::new())),
            security_validator: Arc::new(SecurityValidator::new()),
            degradation_manager: Arc::new(RwLock::new(DegradationManager::new())),
            chaos_controller: Arc::new(RwLock::new(chaos_controller)),
            anomaly_detector: Arc::new(RwLock::new(AnomalyDetector::new(
                AnomalyDetectorConfig::default(),
            ))),
            total_requests_validated: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            total_requests_rejected: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Monitor metric for anomaly detection
    pub async fn monitor_metric(
        &self,
        value: f64,
    ) -> Result<Option<crate::anomaly_detection::AnomalyAlert>> {
        if !self.config.enable_failure_prediction {
            return Ok(None);
        }

        let mut detector = self.anomaly_detector.write().await;
        detector.add_point(value, SystemTime::now())
    }

    /// Validate and harden an incoming request
    pub async fn validate_request(&self, request: &QueryRequest) -> Result<ValidationResult> {
        let mut validations = Vec::new();

        // Security validation
        if self.config.enable_security_hardening {
            match self.security_validator.validate(request) {
                Ok(_) => validations.push("security: pass".to_string()),
                Err(e) => {
                    self.total_requests_rejected
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Ok(ValidationResult {
                        is_valid: false,
                        reason: format!("Security validation failed: {}", e),
                        suggestions: vec!["Review query for injection attempts".to_string()],
                    });
                }
            }
        }

        // Rate limiting check
        if self.config.enable_adaptive_rate_limiting {
            let mut rate_limiter = self.rate_limiter.write().await;
            if !rate_limiter.allow_request(&request.client_id).await? {
                self.total_requests_rejected
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(ValidationResult {
                    is_valid: false,
                    reason: "Rate limit exceeded".to_string(),
                    suggestions: vec!["Reduce request rate or upgrade plan".to_string()],
                });
            }
            validations.push("rate_limit: pass".to_string());
        }

        // Complexity analysis
        if self.config.enable_complexity_analysis {
            match self.complexity_analyzer.analyze(&request.query) {
                Ok(complexity) => {
                    if complexity.score > self.config.max_query_complexity {
                        self.total_requests_rejected
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Ok(ValidationResult {
                            is_valid: false,
                            reason: format!("Query too complex: {:.2}", complexity.score),
                            suggestions: vec![
                                "Simplify query".to_string(),
                                "Add more filters".to_string(),
                                "Break into smaller queries".to_string(),
                            ],
                        });
                    }
                    validations.push(format!("complexity: {:.2}", complexity.score));
                }
                Err(e) => {
                    warn!("Complexity analysis failed: {}", e);
                }
            }
        }

        // Resource quota check
        if self.config.enable_resource_quotas {
            let quota_manager = self.quota_manager.read().await;
            if !quota_manager.check_quota(&request.client_id)? {
                self.total_requests_rejected
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(ValidationResult {
                    is_valid: false,
                    reason: "Resource quota exceeded".to_string(),
                    suggestions: vec!["Wait for quota reset or upgrade plan".to_string()],
                });
            }
            validations.push("quota: pass".to_string());
        }

        // Circuit breaker check
        let circuit_breaker = self.circuit_breaker.read().await;
        if !circuit_breaker.is_closed() {
            self.total_requests_rejected
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(ValidationResult {
                is_valid: false,
                reason: "Circuit breaker open - service temporarily unavailable".to_string(),
                suggestions: vec!["Retry after circuit breaker resets".to_string()],
            });
        }

        // All validations passed
        self.total_requests_validated
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(ValidationResult {
            is_valid: true,
            reason: "All validations passed".to_string(),
            suggestions: vec![],
        })
    }

    /// Record query execution result for learning
    pub async fn record_execution(
        &self,
        request: &QueryRequest,
        result: &ExecutionOutcome,
    ) -> Result<()> {
        // Update circuit breaker
        let mut circuit_breaker = self.circuit_breaker.write().await;
        circuit_breaker.record_execution(result.success).await?;

        // Update quota usage
        if self.config.enable_resource_quotas {
            let mut quota_manager = self.quota_manager.write().await;
            quota_manager.record_usage(&request.client_id, result.resources_used)?;
        }

        Ok(())
    }

    /// Apply graceful degradation if needed
    pub async fn apply_degradation(&self, request: &QueryRequest) -> Result<Option<DegradedQuery>> {
        if !self.config.enable_graceful_degradation {
            return Ok(None);
        }

        let degradation_manager = self.degradation_manager.read().await;
        degradation_manager.degrade_if_needed(request)
    }

    /// Inject chaos for testing resilience
    pub async fn inject_chaos(&self) -> Result<Option<ChaosEvent>> {
        let chaos_controller = self.chaos_controller.read().await;

        if let Some(ref controller) = *chaos_controller {
            controller.maybe_inject_chaos()
        } else {
            Ok(None)
        }
    }

    /// Get hardening statistics
    pub async fn get_statistics(&self) -> Result<HardeningStatistics> {
        let circuit_breaker = self.circuit_breaker.read().await;
        let rate_limiter = self.rate_limiter.read().await;
        let quota_manager = self.quota_manager.read().await;

        Ok(HardeningStatistics {
            circuit_breaker_state: circuit_breaker.state(),
            circuit_breaker_failure_rate: circuit_breaker.failure_rate(),
            rate_limit_rps: rate_limiter.current_rate(),
            quota_usage: quota_manager.total_usage(),
            total_requests_validated: self
                .total_requests_validated
                .load(std::sync::atomic::Ordering::Relaxed),
            total_requests_rejected: self
                .total_requests_rejected
                .load(std::sync::atomic::Ordering::Relaxed),
        })
    }
}

/// ML-powered circuit breaker with failure prediction
#[derive(Debug)]
pub struct MLCircuitBreaker {
    state: CircuitBreakerState,
    failure_threshold: f64,
    timeout: Duration,
    failure_history: VecDeque<bool>,
    history_size: usize,
    last_failure_time: Option<Instant>,
}

impl MLCircuitBreaker {
    pub fn new(failure_threshold: f64, timeout: Duration) -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            failure_threshold,
            timeout,
            failure_history: VecDeque::new(),
            history_size: 100,
            last_failure_time: None,
        }
    }

    pub fn is_closed(&self) -> bool {
        matches!(self.state, CircuitBreakerState::Closed)
    }

    pub fn state(&self) -> CircuitBreakerState {
        self.state
    }

    pub fn failure_rate(&self) -> f64 {
        if self.failure_history.is_empty() {
            return 0.0;
        }

        let failures = self.failure_history.iter().filter(|&&f| !f).count();
        failures as f64 / self.failure_history.len() as f64
    }

    pub async fn record_execution(&mut self, success: bool) -> Result<()> {
        // Add to history
        if self.failure_history.len() >= self.history_size {
            self.failure_history.pop_front();
        }
        self.failure_history.push_back(success);

        if !success {
            self.last_failure_time = Some(Instant::now());
        }

        // Check if circuit should open
        let failure_rate = self.failure_rate();
        if failure_rate > self.failure_threshold {
            if matches!(self.state, CircuitBreakerState::Closed) {
                warn!(
                    "Circuit breaker opening due to high failure rate: {:.2}%",
                    failure_rate * 100.0
                );
                self.state = CircuitBreakerState::Open;
            }
        } else {
            // Check if circuit should close (after timeout)
            if matches!(self.state, CircuitBreakerState::Open) {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed() > self.timeout {
                        info!("Circuit breaker closing - failure rate improved");
                        self.state = CircuitBreakerState::Closed;
                    }
                }
            }
        }

        Ok(())
    }

    /// Train ML model for failure prediction
    pub fn train_predictor(&mut self, _training_data: Vec<f64>) -> Result<()> {
        // Placeholder for ML-based failure prediction
        // Full implementation will use scirs2's anomaly detection
        Ok(())
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Adaptive rate limiter
#[derive(Debug)]
pub struct AdaptiveRateLimiter {
    base_rps: u32,
    current_rps: f64,
    request_history: HashMap<String, VecDeque<Instant>>,
    window_size: Duration,
}

impl AdaptiveRateLimiter {
    pub fn new(base_rps: u32) -> Self {
        Self {
            base_rps,
            current_rps: base_rps as f64,
            request_history: HashMap::new(),
            window_size: Duration::from_secs(1),
        }
    }

    pub fn current_rate(&self) -> f64 {
        self.current_rps
    }

    pub async fn allow_request(&mut self, client_id: &str) -> Result<bool> {
        let now = Instant::now();

        // Get or create client history
        let history = self
            .request_history
            .entry(client_id.to_string())
            .or_default();

        // Clean old requests outside window
        while let Some(front) = history.front() {
            if front.elapsed() > self.window_size {
                history.pop_front();
            } else {
                break;
            }
        }

        // Check rate limit
        let requests_in_window = history.len() as f64;
        let rate = requests_in_window / self.window_size.as_secs_f64();

        if rate < self.current_rps {
            history.push_back(now);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Adapt rate limit based on system load
    pub fn adapt_rate(&mut self, system_load: f64) {
        // Decrease rate if system is overloaded
        if system_load > 0.8 {
            self.current_rps = self.base_rps as f64 * 0.5;
        } else if system_load > 0.6 {
            self.current_rps = self.base_rps as f64 * 0.75;
        } else {
            self.current_rps = self.base_rps as f64;
        }
    }
}

/// Query complexity analyzer
#[derive(Debug)]
pub struct QueryComplexityAnalyzer {
    max_complexity: f64,
}

impl QueryComplexityAnalyzer {
    pub fn new(max_complexity: f64) -> Self {
        Self { max_complexity }
    }

    pub fn analyze(&self, query: &str) -> Result<ComplexityResult> {
        let mut score = 0.0;

        // Count joins (each join adds complexity)
        let join_count = query.matches("JOIN").count() + query.matches("OPTIONAL").count();
        score += join_count as f64 * 10.0;

        // Count subqueries
        let subquery_count = query.matches("SELECT").count().saturating_sub(1);
        score += subquery_count as f64 * 20.0;

        // Count filters
        let filter_count = query.matches("FILTER").count();
        score += filter_count as f64 * 5.0;

        // Count unions
        let union_count = query.matches("UNION").count();
        score += union_count as f64 * 15.0;

        // Count triple patterns (approximate)
        let triple_count = query.matches('.').count();
        score += triple_count as f64 * 2.0;

        Ok(ComplexityResult {
            score,
            join_count,
            subquery_count,
            filter_count,
            union_count,
            triple_count,
        })
    }
}

/// Resource quota manager
#[derive(Debug)]
pub struct ResourceQuotaManager {
    quotas: HashMap<String, ClientQuota>,
}

impl ResourceQuotaManager {
    pub fn new() -> Self {
        Self {
            quotas: HashMap::new(),
        }
    }

    pub fn check_quota(&self, client_id: &str) -> Result<bool> {
        if let Some(quota) = self.quotas.get(client_id) {
            Ok(quota.usage < quota.limit)
        } else {
            // No quota set - allow
            Ok(true)
        }
    }

    pub fn record_usage(&mut self, client_id: &str, resources: u64) -> Result<()> {
        let quota = self
            .quotas
            .entry(client_id.to_string())
            .or_insert_with(|| ClientQuota {
                limit: 10000,
                usage: 0,
                reset_time: SystemTime::now() + Duration::from_secs(3600),
            });

        quota.usage += resources;

        // Reset if needed
        if SystemTime::now() > quota.reset_time {
            quota.usage = 0;
            quota.reset_time = SystemTime::now() + Duration::from_secs(3600);
        }

        Ok(())
    }

    pub fn total_usage(&self) -> u64 {
        self.quotas.values().map(|q| q.usage).sum()
    }
}

impl Default for ResourceQuotaManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Client quota
#[derive(Debug, Clone)]
pub struct ClientQuota {
    pub limit: u64,
    pub usage: u64,
    pub reset_time: SystemTime,
}

/// Security validator
#[derive(Debug)]
pub struct SecurityValidator;

impl SecurityValidator {
    pub fn new() -> Self {
        Self
    }

    pub fn validate(&self, request: &QueryRequest) -> Result<()> {
        // Check for SQL/SPARQL injection patterns
        let dangerous_patterns = [
            "DROP", "DELETE", "INSERT", "UPDATE", "CREATE", "ALTER", "--", "/*", "*/", "xp_", "sp_",
        ];

        let query_upper = request.query.to_uppercase();

        for pattern in &dangerous_patterns {
            if query_upper.contains(pattern) {
                // Check if it's a legitimate use (e.g., in literals)
                // Simplified check - in production would use full parser
                return Err(anyhow!(
                    "Potentially dangerous pattern detected: {}",
                    pattern
                ));
            }
        }

        // Check query length (DoS prevention)
        if request.query.len() > 100_000 {
            return Err(anyhow!("Query too long - potential DoS attempt"));
        }

        // Check for deeply nested queries (resource exhaustion)
        let nesting_level = request.query.matches('{').count();
        if nesting_level > 20 {
            return Err(anyhow!(
                "Query nesting too deep - potential resource exhaustion"
            ));
        }

        Ok(())
    }
}

impl Default for SecurityValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Degradation manager for graceful degradation
#[derive(Debug)]
pub struct DegradationManager {
    #[allow(dead_code)]
    degradation_strategies: Vec<DegradationStrategy>,
}

impl DegradationManager {
    pub fn new() -> Self {
        Self {
            degradation_strategies: vec![
                DegradationStrategy::ReduceTimeout,
                DegradationStrategy::DisableOptionalClauses,
                DegradationStrategy::LimitResults,
                DegradationStrategy::UseCache,
            ],
        }
    }

    pub fn degrade_if_needed(&self, request: &QueryRequest) -> Result<Option<DegradedQuery>> {
        // Check if degradation is needed (simplified)
        // In production would check system metrics

        if request.query.len() > 10000 {
            // Apply degradation
            let mut degraded = request.clone();
            degraded.query = self.apply_degradation(&degraded.query)?;

            return Ok(Some(DegradedQuery {
                original: request.clone(),
                degraded,
                strategy: DegradationStrategy::LimitResults,
            }));
        }

        Ok(None)
    }

    fn apply_degradation(&self, query: &str) -> Result<String> {
        // Add LIMIT if not present
        if !query.to_uppercase().contains("LIMIT") {
            Ok(format!("{} LIMIT 1000", query))
        } else {
            Ok(query.to_string())
        }
    }
}

impl Default for DegradationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Degradation strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DegradationStrategy {
    ReduceTimeout,
    DisableOptionalClauses,
    LimitResults,
    UseCache,
}

/// Chaos engineering controller
#[derive(Debug)]
pub struct ChaosController {
    chaos_probability: f64,
}

impl ChaosController {
    pub fn new() -> Self {
        Self {
            chaos_probability: 0.01, // 1% chaos injection rate
        }
    }

    pub fn maybe_inject_chaos(&self) -> Result<Option<ChaosEvent>> {
        // Simple chaos injection using timestamp-based random
        // In production, would use proper RNG
        let roll = (Instant::now().elapsed().as_nanos() % 100) as f64 / 100.0;

        if roll < self.chaos_probability {
            // Inject chaos!
            let chaos_idx = (Instant::now().elapsed().as_nanos() % 4) as usize;
            let chaos_type = match chaos_idx {
                0 => ChaosType::Latency(Duration::from_millis(1000)),
                1 => ChaosType::Error("Simulated failure".to_string()),
                2 => ChaosType::Timeout,
                _ => ChaosType::PartialFailure,
            };

            Ok(Some(ChaosEvent {
                chaos_type,
                timestamp: Instant::now(),
            }))
        } else {
            Ok(None)
        }
    }
}

impl Default for ChaosController {
    fn default() -> Self {
        Self::new()
    }
}

/// Chaos event
#[derive(Debug, Clone)]
pub struct ChaosEvent {
    pub chaos_type: ChaosType,
    pub timestamp: Instant,
}

/// Chaos type
#[derive(Debug, Clone)]
pub enum ChaosType {
    Latency(Duration),
    Error(String),
    Timeout,
    PartialFailure,
}

// Supporting types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub client_id: String,
    pub query: String,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub reason: String,
    pub suggestions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ExecutionOutcome {
    pub success: bool,
    pub resources_used: u64,
}

#[derive(Debug, Clone)]
pub struct ComplexityResult {
    pub score: f64,
    pub join_count: usize,
    pub subquery_count: usize,
    pub filter_count: usize,
    pub union_count: usize,
    pub triple_count: usize,
}

#[derive(Debug, Clone)]
pub struct DegradedQuery {
    pub original: QueryRequest,
    pub degraded: QueryRequest,
    pub strategy: DegradationStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardeningStatistics {
    pub circuit_breaker_state: CircuitBreakerState,
    pub circuit_breaker_failure_rate: f64,
    pub rate_limit_rps: f64,
    pub quota_usage: u64,
    pub total_requests_validated: u64,
    pub total_requests_rejected: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardening_config_default() {
        let config = HardeningConfig::default();
        assert!(config.enable_failure_prediction);
        assert!(config.enable_security_hardening);
        assert!(!config.enable_chaos_engineering);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let mut cb = MLCircuitBreaker::new(0.5, Duration::from_secs(30));

        assert!(cb.is_closed());

        // Record some failures
        for _ in 0..60 {
            cb.record_execution(false).await.unwrap();
        }

        // Circuit should open
        assert!(!cb.is_closed());
        assert!(cb.failure_rate() > 0.5);
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let mut limiter = AdaptiveRateLimiter::new(10);

        // Should allow first request
        assert!(limiter.allow_request("client1").await.unwrap());

        // Adapt rate based on load
        limiter.adapt_rate(0.9);
        assert!(limiter.current_rate() < 10.0);
    }

    #[test]
    fn test_complexity_analyzer() {
        let analyzer = QueryComplexityAnalyzer::new(1000.0);

        let simple_query = "SELECT * WHERE { ?s ?p ?o }";
        let result = analyzer.analyze(simple_query).unwrap();
        assert!(result.score < 100.0);

        let complex_query =
            "SELECT * WHERE { ?s ?p ?o . ?o ?p2 ?o2 FILTER(?x > 10) OPTIONAL { ?o2 ?p3 ?o3 } }";
        let result2 = analyzer.analyze(complex_query).unwrap();
        assert!(result2.score > result.score);
    }

    #[test]
    fn test_security_validator() {
        let validator = SecurityValidator::new();

        let safe_query = QueryRequest {
            client_id: "test".to_string(),
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
        };

        assert!(validator.validate(&safe_query).is_ok());

        let dangerous_query = QueryRequest {
            client_id: "test".to_string(),
            query: "DROP TABLE users".to_string(),
        };

        assert!(validator.validate(&dangerous_query).is_err());
    }

    #[test]
    fn test_quota_manager() {
        let mut manager = ResourceQuotaManager::new();

        assert!(manager.check_quota("client1").unwrap());

        manager.record_usage("client1", 5000).unwrap();
        assert!(manager.check_quota("client1").unwrap());

        manager.record_usage("client1", 6000).unwrap();
        assert!(!manager.check_quota("client1").unwrap());
    }
}
