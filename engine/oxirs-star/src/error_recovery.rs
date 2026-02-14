//! Advanced error handling and recovery mechanisms for RDF-star operations.
//!
//! This module provides comprehensive error handling, recovery strategies,
//! and resilience features for robust RDF-star processing.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn};

use crate::model::{StarGraph, StarQuad, StarTerm, StarTriple};
use crate::parser::{ParseError, StarFormat};
use crate::{StarConfig, StarError, StarResult};

/// Error recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecoveryConfig {
    /// Enable automatic error recovery
    pub auto_recovery: bool,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Base retry delay (milliseconds)
    pub base_retry_delay: u64,
    /// Maximum retry delay (milliseconds)
    pub max_retry_delay: u64,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Timeout configuration
    pub timeouts: TimeoutConfig,
    /// Fallback strategies
    pub fallbacks: FallbackConfig,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,
    /// Failure threshold to open circuit
    pub failure_threshold: usize,
    /// Success threshold to close circuit
    pub success_threshold: usize,
    /// Timeout for half-open state (seconds)
    pub timeout: u64,
    /// Minimum throughput for circuit breaker evaluation
    pub minimum_throughput: usize,
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Parse operation timeout (seconds)
    pub parse_timeout: u64,
    /// Serialization timeout (seconds)
    pub serialization_timeout: u64,
    /// Query timeout (seconds)
    pub query_timeout: u64,
    /// Network operation timeout (seconds)
    pub network_timeout: u64,
}

/// Fallback configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackConfig {
    /// Enable graceful degradation
    pub graceful_degradation: bool,
    /// Fallback data sources
    pub fallback_sources: Vec<String>,
    /// Partial result acceptance
    pub accept_partial_results: bool,
    /// Default fallback values
    pub default_values: HashMap<String, String>,
}

/// Advanced error types with context and recovery information
#[derive(Error, Debug, Clone)]
pub enum AdvancedStarError {
    #[error("Parsing error with recovery context: {message}")]
    ParseErrorWithContext {
        message: String,
        line: Option<usize>,
        column: Option<usize>,
        context: ErrorContext,
        recoverable: bool,
        suggested_fix: Option<String>,
    },

    #[error("Serialization error with recovery context: {message}")]
    SerializationErrorWithContext {
        message: String,
        format: StarFormat,
        context: ErrorContext,
        recoverable: bool,
        alternative_formats: Vec<StarFormat>,
    },

    #[error("Query execution error with recovery context: {message}")]
    QueryErrorWithContext {
        message: String,
        query_fragment: Option<String>,
        context: ErrorContext,
        recoverable: bool,
        retry_strategy: Option<RetryStrategy>,
    },

    #[error("Network error with recovery context: {message}")]
    NetworkErrorWithContext {
        message: String,
        endpoint: Option<String>,
        status_code: Option<u16>,
        context: ErrorContext,
        recoverable: bool,
        backoff_delay: Option<Duration>,
    },

    #[error("Resource exhaustion error: {message}")]
    ResourceExhaustion {
        message: String,
        resource_type: ResourceType,
        current_usage: u64,
        limit: u64,
        context: ErrorContext,
        recovery_actions: Vec<RecoveryAction>,
    },

    #[error("Timeout error: {message}")]
    TimeoutError {
        message: String,
        operation: String,
        timeout_duration: Duration,
        elapsed: Duration,
        context: ErrorContext,
        retry_recommended: bool,
    },

    #[error("Circuit breaker open: {message}")]
    CircuitBreakerOpen {
        message: String,
        service: String,
        failure_count: usize,
        context: ErrorContext,
        retry_after: Duration,
    },
}

/// Error context for enhanced debugging and recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Unique error ID for tracking
    pub error_id: String,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Operation being performed
    pub operation: String,
    /// Input data size
    pub input_size: Option<usize>,
    /// Processing stage where error occurred
    pub stage: ProcessingStage,
    /// Environment information
    pub environment: EnvironmentInfo,
    /// Stack trace (if available)
    pub stack_trace: Option<String>,
    /// Related operations
    pub related_operations: Vec<String>,
}

/// Processing stages for error categorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStage {
    Input,
    Parsing,
    Validation,
    Processing,
    Serialization,
    Output,
    Network,
    Storage,
}

/// Environment information for error context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    /// Available memory (bytes)
    pub available_memory: u64,
    /// CPU load (percentage)
    pub cpu_load: f64,
    /// Active connections
    pub active_connections: usize,
    /// Thread pool status
    pub thread_pool_status: ThreadPoolStatus,
}

/// Thread pool status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolStatus {
    /// Active threads
    pub active_threads: usize,
    /// Maximum threads
    pub max_threads: usize,
    /// Queue size
    pub queue_size: usize,
}

/// Resource types for resource exhaustion errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    Memory,
    CpuTime,
    NetworkBandwidth,
    DiskSpace,
    FileHandles,
    ThreadPool,
    ConnectionPool,
}

/// Recovery actions for resource exhaustion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    FreeMemory,
    ReduceConcurrency,
    IncreaseTimeout,
    SwitchToAlternative,
    RequestMoreResources,
    EnableCompression,
    ClearCaches,
}

/// Retry strategies for different error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetryStrategy {
    Immediate,
    FixedDelay(Duration),
    ExponentialBackoff { base: Duration, max: Duration },
    LinearBackoff { increment: Duration, max: Duration },
    CustomDelay(Vec<Duration>),
}

/// Error recovery manager
pub struct ErrorRecoveryManager {
    config: ErrorRecoveryConfig,
    retry_state: Arc<Mutex<RetryState>>,
    circuit_breakers: Arc<Mutex<HashMap<String, CircuitBreaker>>>,
    error_history: Arc<Mutex<VecDeque<ErrorRecord>>>,
    recovery_strategies: HashMap<String, Box<dyn RecoveryStrategy>>,
}

/// Retry state tracking
#[derive(Debug, Clone)]
pub struct RetryState {
    /// Operation ID to retry count mapping
    pub retry_counts: HashMap<String, usize>,
    /// Last retry timestamps
    pub last_retry_times: HashMap<String, Instant>,
    /// Successful operations (for circuit breaker)
    pub success_counts: HashMap<String, usize>,
}

/// Circuit breaker for preventing cascading failures
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Current state
    pub state: CircuitBreakerState,
    /// Failure count in current window
    pub failure_count: usize,
    /// Success count in current window
    pub success_count: usize,
    /// Last failure time
    pub last_failure_time: Option<Instant>,
    /// Configuration
    pub config: CircuitBreakerConfig,
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Error record for tracking and analysis
#[derive(Debug, Clone)]
pub struct ErrorRecord {
    /// Error details
    pub error: AdvancedStarError,
    /// Recovery attempts made
    pub recovery_attempts: Vec<RecoveryAttempt>,
    /// Final outcome
    pub outcome: RecoveryOutcome,
    /// Total time spent on recovery
    pub recovery_time: Duration,
}

/// Recovery attempt record
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    /// Strategy used
    pub strategy: String,
    /// Start time
    pub start_time: Instant,
    /// Duration
    pub duration: Duration,
    /// Success indicator
    pub success: bool,
    /// Error (if failed)
    pub error: Option<String>,
}

/// Recovery outcome
#[derive(Debug, Clone)]
pub enum RecoveryOutcome {
    Success,
    PartialSuccess,
    Failed,
    Aborted,
}

/// Recovery strategy trait
pub trait RecoveryStrategy: Send + Sync {
    fn can_recover(&self, error: &AdvancedStarError) -> bool;
    fn recover(&self, error: &AdvancedStarError, context: &ErrorContext) -> RecoveryResult;
    fn get_name(&self) -> &str;
}

/// Recovery result
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    pub success: bool,
    pub retry_recommended: bool,
    pub delay_before_retry: Option<Duration>,
    pub alternative_action: Option<String>,
    pub partial_result: Option<Vec<u8>>,
}

/// Parsing error recovery strategy
pub struct ParseErrorRecovery;

/// Serialization error recovery strategy
pub struct SerializationErrorRecovery;

/// Network error recovery strategy
pub struct NetworkErrorRecovery;

/// Resource exhaustion recovery strategy
pub struct ResourceExhaustionRecovery;

/// Timeout error recovery strategy
pub struct TimeoutErrorRecovery;

impl Default for ErrorRecoveryConfig {
    fn default() -> Self {
        Self {
            auto_recovery: true,
            max_retries: 3,
            base_retry_delay: 1000,
            max_retry_delay: 30000,
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 5,
                success_threshold: 3,
                timeout: 60,
                minimum_throughput: 10,
            },
            timeouts: TimeoutConfig {
                parse_timeout: 30,
                serialization_timeout: 30,
                query_timeout: 60,
                network_timeout: 30,
            },
            fallbacks: FallbackConfig {
                graceful_degradation: true,
                fallback_sources: Vec::new(),
                accept_partial_results: true,
                default_values: HashMap::new(),
            },
        }
    }
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new(config: ErrorRecoveryConfig) -> Self {
        let mut manager = Self {
            config,
            retry_state: Arc::new(Mutex::new(RetryState {
                retry_counts: HashMap::new(),
                last_retry_times: HashMap::new(),
                success_counts: HashMap::new(),
            })),
            circuit_breakers: Arc::new(Mutex::new(HashMap::new())),
            error_history: Arc::new(Mutex::new(VecDeque::new())),
            recovery_strategies: HashMap::new(),
        };

        // Register default recovery strategies
        manager.register_strategy(Box::new(ParseErrorRecovery));
        manager.register_strategy(Box::new(SerializationErrorRecovery));
        manager.register_strategy(Box::new(NetworkErrorRecovery));
        manager.register_strategy(Box::new(ResourceExhaustionRecovery));
        manager.register_strategy(Box::new(TimeoutErrorRecovery));

        manager
    }

    /// Register a recovery strategy
    pub fn register_strategy(&mut self, strategy: Box<dyn RecoveryStrategy>) {
        self.recovery_strategies.insert(strategy.get_name().to_string(), strategy);
    }

    /// Handle an error with automatic recovery
    pub async fn handle_error(&self, error: AdvancedStarError) -> Result<RecoveryResult, AdvancedStarError> {
        if !self.config.auto_recovery {
            return Err(error);
        }

        let operation_id = self.generate_operation_id(&error);

        // Check circuit breaker
        if self.is_circuit_open(&operation_id).await {
            return Err(AdvancedStarError::CircuitBreakerOpen {
                message: "Circuit breaker is open".to_string(),
                service: operation_id.clone(),
                failure_count: self.get_failure_count(&operation_id).await,
                context: self.create_error_context(&error),
                retry_after: Duration::from_secs(self.config.circuit_breaker.timeout),
            });
        }

        // Check retry limits
        if !self.should_retry(&operation_id).await {
            warn!("Maximum retry attempts reached for operation: {}", operation_id);
            self.record_failure(&operation_id).await;
            return Err(error);
        }

        // Find appropriate recovery strategy
        let strategy = self.find_recovery_strategy(&error);

        if let Some(strategy) = strategy {
            info!("Attempting recovery for error using strategy: {}", strategy.get_name());

            let context = self.create_error_context(&error);
            let recovery_start = Instant::now();

            let result = strategy.recover(&error, &context);

            let recovery_time = recovery_start.elapsed();

            if result.success {
                info!("Error recovery successful");
                self.record_success(&operation_id).await;
                self.record_recovery_attempt(&error, strategy.get_name(), recovery_time, true, None).await;
                Ok(result)
            } else {
                warn!("Error recovery failed");
                self.record_failure(&operation_id).await;
                self.record_recovery_attempt(&error, strategy.get_name(), recovery_time, false, Some("Recovery strategy failed".to_string())).await;

                if result.retry_recommended {
                    self.schedule_retry(&operation_id, result.delay_before_retry).await;
                }

                Err(error)
            }
        } else {
            warn!("No recovery strategy found for error");
            self.record_failure(&operation_id).await;
            Err(error)
        }
    }

    /// Create enhanced error context
    pub fn create_error_context(&self, error: &AdvancedStarError) -> ErrorContext {
        ErrorContext {
            error_id: uuid::Uuid::new_v4().to_string(),
            timestamp: std::time::SystemTime::now(),
            operation: self.extract_operation_from_error(error),
            input_size: None,
            stage: self.determine_processing_stage(error),
            environment: self.collect_environment_info(),
            stack_trace: None, // Could be filled with actual stack trace
            related_operations: Vec::new(),
        }
    }

    /// Convert standard StarError to enhanced error
    pub fn enhance_error(&self, error: StarError, context: Option<ErrorContext>) -> AdvancedStarError {
        let ctx = context.unwrap_or_else(|| self.create_error_context(&AdvancedStarError::ParseErrorWithContext {
            message: "Unknown error".to_string(),
            line: None,
            column: None,
            context: self.create_default_context(),
            recoverable: false,
            suggested_fix: None,
        }));

        match error {
            StarError::ParseError { message, line, column } => {
                AdvancedStarError::ParseErrorWithContext {
                    message,
                    line,
                    column,
                    context: ctx,
                    recoverable: true,
                    suggested_fix: self.suggest_parse_fix(&message),
                }
            }
            StarError::SerializationError { message } => {
                AdvancedStarError::SerializationErrorWithContext {
                    message,
                    format: StarFormat::TurtleStar, // Default, could be determined from context
                    context: ctx,
                    recoverable: true,
                    alternative_formats: vec![StarFormat::NTriplesStar, StarFormat::JsonLdStar],
                }
            }
            StarError::QueryError { message } => {
                AdvancedStarError::QueryErrorWithContext {
                    message,
                    query_fragment: None,
                    context: ctx,
                    recoverable: true,
                    retry_strategy: Some(RetryStrategy::ExponentialBackoff {
                        base: Duration::from_millis(self.config.base_retry_delay),
                        max: Duration::from_millis(self.config.max_retry_delay),
                    }),
                }
            }
            StarError::ValidationError { message } => {
                AdvancedStarError::ParseErrorWithContext {
                    message,
                    line: None,
                    column: None,
                    context: ctx,
                    recoverable: false,
                    suggested_fix: Some("Validate input data before processing".to_string()),
                }
            }
            StarError::IoError { message } => {
                AdvancedStarError::NetworkErrorWithContext {
                    message,
                    endpoint: None,
                    status_code: None,
                    context: ctx,
                    recoverable: true,
                    backoff_delay: Some(Duration::from_millis(self.config.base_retry_delay)),
                }
            }
        }
    }

    /// Get error statistics
    pub async fn get_error_statistics(&self) -> ErrorStatistics {
        let history = self.error_history.lock().expect("lock should not be poisoned");

        let total_errors = history.len();
        let successful_recoveries = history.iter()
            .filter(|record| matches!(record.outcome, RecoveryOutcome::Success))
            .count();
        let partial_recoveries = history.iter()
            .filter(|record| matches!(record.outcome, RecoveryOutcome::PartialSuccess))
            .count();
        let failed_recoveries = history.iter()
            .filter(|record| matches!(record.outcome, RecoveryOutcome::Failed))
            .count();

        let recovery_rate = if total_errors > 0 {
            (successful_recoveries + partial_recoveries) as f64 / total_errors as f64
        } else {
            0.0
        };

        let avg_recovery_time = if !history.is_empty() {
            history.iter()
                .map(|record| record.recovery_time.as_millis() as f64)
                .sum::<f64>() / history.len() as f64
        } else {
            0.0
        };

        ErrorStatistics {
            total_errors,
            successful_recoveries,
            partial_recoveries,
            failed_recoveries,
            recovery_rate,
            avg_recovery_time,
            error_types: self.get_error_type_counts(&history),
        }
    }

    /// Clear error history
    pub async fn clear_error_history(&self) {
        let mut history = self.error_history.lock().expect("lock should not be poisoned");
        history.clear();
    }

    // Private helper methods

    fn find_recovery_strategy(&self, error: &AdvancedStarError) -> Option<&Box<dyn RecoveryStrategy>> {
        self.recovery_strategies.values()
            .find(|strategy| strategy.can_recover(error))
    }

    fn generate_operation_id(&self, error: &AdvancedStarError) -> String {
        match error {
            AdvancedStarError::ParseErrorWithContext { context, .. } => context.operation.clone(),
            AdvancedStarError::SerializationErrorWithContext { context, .. } => context.operation.clone(),
            AdvancedStarError::QueryErrorWithContext { context, .. } => context.operation.clone(),
            AdvancedStarError::NetworkErrorWithContext { context, .. } => context.operation.clone(),
            AdvancedStarError::ResourceExhaustion { context, .. } => context.operation.clone(),
            AdvancedStarError::TimeoutError { context, .. } => context.operation.clone(),
            AdvancedStarError::CircuitBreakerOpen { service, .. } => service.clone(),
        }
    }

    async fn should_retry(&self, operation_id: &str) -> bool {
        if let Ok(state) = self.retry_state.lock() {
            let retry_count = state.retry_counts.get(operation_id).unwrap_or(&0);
            *retry_count < self.config.max_retries
        } else {
            false
        }
    }

    async fn is_circuit_open(&self, operation_id: &str) -> bool {
        if let Ok(breakers) = self.circuit_breakers.lock() {
            if let Some(breaker) = breakers.get(operation_id) {
                breaker.state == CircuitBreakerState::Open
            } else {
                false
            }
        } else {
            false
        }
    }

    async fn get_failure_count(&self, operation_id: &str) -> usize {
        if let Ok(breakers) = self.circuit_breakers.lock() {
            if let Some(breaker) = breakers.get(operation_id) {
                breaker.failure_count
            } else {
                0
            }
        } else {
            0
        }
    }

    async fn record_success(&self, operation_id: &str) {
        if let Ok(mut state) = self.retry_state.lock() {
            state.retry_counts.remove(operation_id);
            let success_count = state.success_counts.entry(operation_id.to_string()).or_insert(0);
            *success_count += 1;
        }

        if let Ok(mut breakers) = self.circuit_breakers.lock() {
            let breaker = breakers.entry(operation_id.to_string())
                .or_insert_with(|| CircuitBreaker {
                    state: CircuitBreakerState::Closed,
                    failure_count: 0,
                    success_count: 0,
                    last_failure_time: None,
                    config: self.config.circuit_breaker.clone(),
                });

            breaker.success_count += 1;

            if breaker.state == CircuitBreakerState::HalfOpen &&
               breaker.success_count >= breaker.config.success_threshold {
                breaker.state = CircuitBreakerState::Closed;
                breaker.failure_count = 0;
            }
        }
    }

    async fn record_failure(&self, operation_id: &str) {
        if let Ok(mut state) = self.retry_state.lock() {
            let retry_count = state.retry_counts.entry(operation_id.to_string()).or_insert(0);
            *retry_count += 1;
        }

        if let Ok(mut breakers) = self.circuit_breakers.lock() {
            let breaker = breakers.entry(operation_id.to_string())
                .or_insert_with(|| CircuitBreaker {
                    state: CircuitBreakerState::Closed,
                    failure_count: 0,
                    success_count: 0,
                    last_failure_time: None,
                    config: self.config.circuit_breaker.clone(),
                });

            breaker.failure_count += 1;
            breaker.last_failure_time = Some(Instant::now());

            if breaker.failure_count >= breaker.config.failure_threshold {
                breaker.state = CircuitBreakerState::Open;
            }
        }
    }

    async fn schedule_retry(&self, operation_id: &str, delay: Option<Duration>) {
        if let Ok(mut state) = self.retry_state.lock() {
            let now = Instant::now();
            let retry_time = now + delay.unwrap_or_else(|| {
                Duration::from_millis(self.config.base_retry_delay)
            });

            state.last_retry_times.insert(operation_id.to_string(), retry_time);
        }
    }

    async fn record_recovery_attempt(
        &self,
        error: &AdvancedStarError,
        strategy: &str,
        duration: Duration,
        success: bool,
        error_msg: Option<String>
    ) {
        let attempt = RecoveryAttempt {
            strategy: strategy.to_string(),
            start_time: Instant::now() - duration,
            duration,
            success,
            error: error_msg,
        };

        if let Ok(mut history) = self.error_history.lock() {
            // Find the corresponding error record or create a new one
            if let Some(record) = history.back_mut() {
                record.recovery_attempts.push(attempt);
                record.recovery_time += duration;
                if success {
                    record.outcome = RecoveryOutcome::Success;
                }
            } else {
                // Create a new error record
                let record = ErrorRecord {
                    error: error.clone(),
                    recovery_attempts: vec![attempt],
                    outcome: if success { RecoveryOutcome::Success } else { RecoveryOutcome::Failed },
                    recovery_time: duration,
                };
                history.push_back(record);
            }

            // Keep only the last 1000 error records
            while history.len() > 1000 {
                history.pop_front();
            }
        }
    }

    fn extract_operation_from_error(&self, error: &AdvancedStarError) -> String {
        match error {
            AdvancedStarError::ParseErrorWithContext { .. } => "parse".to_string(),
            AdvancedStarError::SerializationErrorWithContext { .. } => "serialize".to_string(),
            AdvancedStarError::QueryErrorWithContext { .. } => "query".to_string(),
            AdvancedStarError::NetworkErrorWithContext { .. } => "network".to_string(),
            AdvancedStarError::ResourceExhaustion { .. } => "resource".to_string(),
            AdvancedStarError::TimeoutError { operation, .. } => operation.clone(),
            AdvancedStarError::CircuitBreakerOpen { service, .. } => service.clone(),
        }
    }

    fn determine_processing_stage(&self, error: &AdvancedStarError) -> ProcessingStage {
        match error {
            AdvancedStarError::ParseErrorWithContext { .. } => ProcessingStage::Parsing,
            AdvancedStarError::SerializationErrorWithContext { .. } => ProcessingStage::Serialization,
            AdvancedStarError::QueryErrorWithContext { .. } => ProcessingStage::Processing,
            AdvancedStarError::NetworkErrorWithContext { .. } => ProcessingStage::Network,
            AdvancedStarError::ResourceExhaustion { .. } => ProcessingStage::Processing,
            AdvancedStarError::TimeoutError { .. } => ProcessingStage::Processing,
            AdvancedStarError::CircuitBreakerOpen { .. } => ProcessingStage::Network,
        }
    }

    fn collect_environment_info(&self) -> EnvironmentInfo {
        EnvironmentInfo {
            available_memory: 1024 * 1024 * 1024, // 1GB placeholder
            cpu_load: 0.5, // 50% placeholder
            active_connections: 10, // placeholder
            thread_pool_status: ThreadPoolStatus {
                active_threads: 4,
                max_threads: 8,
                queue_size: 0,
            },
        }
    }

    fn create_default_context(&self) -> ErrorContext {
        ErrorContext {
            error_id: uuid::Uuid::new_v4().to_string(),
            timestamp: std::time::SystemTime::now(),
            operation: "unknown".to_string(),
            input_size: None,
            stage: ProcessingStage::Processing,
            environment: self.collect_environment_info(),
            stack_trace: None,
            related_operations: Vec::new(),
        }
    }

    fn suggest_parse_fix(&self, message: &str) -> Option<String> {
        if message.contains("quoted triple") {
            Some("Check quoted triple syntax: << subject predicate object >>".to_string())
        } else if message.contains("prefix") {
            Some("Verify namespace prefix declarations".to_string())
        } else if message.contains("literal") {
            Some("Check literal syntax and datatype declarations".to_string())
        } else {
            Some("Validate RDF-star syntax according to specification".to_string())
        }
    }

    fn get_error_type_counts(&self, history: &VecDeque<ErrorRecord>) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for record in history {
            let error_type = match &record.error {
                AdvancedStarError::ParseErrorWithContext { .. } => "parse_error",
                AdvancedStarError::SerializationErrorWithContext { .. } => "serialization_error",
                AdvancedStarError::QueryErrorWithContext { .. } => "query_error",
                AdvancedStarError::NetworkErrorWithContext { .. } => "network_error",
                AdvancedStarError::ResourceExhaustion { .. } => "resource_exhaustion",
                AdvancedStarError::TimeoutError { .. } => "timeout_error",
                AdvancedStarError::CircuitBreakerOpen { .. } => "circuit_breaker_open",
            };

            *counts.entry(error_type.to_string()).or_insert(0) += 1;
        }

        counts
    }
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Total number of errors
    pub total_errors: usize,
    /// Number of successful recoveries
    pub successful_recoveries: usize,
    /// Number of partial recoveries
    pub partial_recoveries: usize,
    /// Number of failed recoveries
    pub failed_recoveries: usize,
    /// Overall recovery rate
    pub recovery_rate: f64,
    /// Average recovery time (milliseconds)
    pub avg_recovery_time: f64,
    /// Error counts by type
    pub error_types: HashMap<String, usize>,
}

// Recovery strategy implementations

impl RecoveryStrategy for ParseErrorRecovery {
    fn can_recover(&self, error: &AdvancedStarError) -> bool {
        matches!(error, AdvancedStarError::ParseErrorWithContext { recoverable: true, .. })
    }

    fn recover(&self, error: &AdvancedStarError, _context: &ErrorContext) -> RecoveryResult {
        if let AdvancedStarError::ParseErrorWithContext { message, .. } = error {
            // Implement parse error recovery logic
            if message.contains("quoted triple") {
                RecoveryResult {
                    success: false,
                    retry_recommended: true,
                    delay_before_retry: Some(Duration::from_millis(1000)),
                    alternative_action: Some("Try parsing with lenient mode".to_string()),
                    partial_result: None,
                }
            } else {
                RecoveryResult {
                    success: false,
                    retry_recommended: false,
                    delay_before_retry: None,
                    alternative_action: Some("Manual syntax correction required".to_string()),
                    partial_result: None,
                }
            }
        } else {
            RecoveryResult {
                success: false,
                retry_recommended: false,
                delay_before_retry: None,
                alternative_action: None,
                partial_result: None,
            }
        }
    }

    fn get_name(&self) -> &str {
        "parse_error_recovery"
    }
}

impl RecoveryStrategy for SerializationErrorRecovery {
    fn can_recover(&self, error: &AdvancedStarError) -> bool {
        matches!(error, AdvancedStarError::SerializationErrorWithContext { recoverable: true, .. })
    }

    fn recover(&self, error: &AdvancedStarError, _context: &ErrorContext) -> RecoveryResult {
        if let AdvancedStarError::SerializationErrorWithContext { alternative_formats, .. } = error {
            if !alternative_formats.is_empty() {
                RecoveryResult {
                    success: false,
                    retry_recommended: true,
                    delay_before_retry: Some(Duration::from_millis(500)),
                    alternative_action: Some(format!("Try alternative format: {:?}", alternative_formats[0])),
                    partial_result: None,
                }
            } else {
                RecoveryResult {
                    success: false,
                    retry_recommended: false,
                    delay_before_retry: None,
                    alternative_action: None,
                    partial_result: None,
                }
            }
        } else {
            RecoveryResult {
                success: false,
                retry_recommended: false,
                delay_before_retry: None,
                alternative_action: None,
                partial_result: None,
            }
        }
    }

    fn get_name(&self) -> &str {
        "serialization_error_recovery"
    }
}

impl RecoveryStrategy for NetworkErrorRecovery {
    fn can_recover(&self, error: &AdvancedStarError) -> bool {
        matches!(error, AdvancedStarError::NetworkErrorWithContext { recoverable: true, .. })
    }

    fn recover(&self, error: &AdvancedStarError, _context: &ErrorContext) -> RecoveryResult {
        if let AdvancedStarError::NetworkErrorWithContext { status_code, backoff_delay, .. } = error {
            let should_retry = match status_code {
                Some(500..=599) => true, // Server errors are retryable
                Some(429) => true, // Rate limiting is retryable
                Some(408) => true, // Request timeout is retryable
                _ => false,
            };

            RecoveryResult {
                success: false,
                retry_recommended: should_retry,
                delay_before_retry: *backoff_delay,
                alternative_action: if should_retry {
                    Some("Retry with exponential backoff".to_string())
                } else {
                    Some("Check network connectivity and endpoint status".to_string())
                },
                partial_result: None,
            }
        } else {
            RecoveryResult {
                success: false,
                retry_recommended: false,
                delay_before_retry: None,
                alternative_action: None,
                partial_result: None,
            }
        }
    }

    fn get_name(&self) -> &str {
        "network_error_recovery"
    }
}

impl RecoveryStrategy for ResourceExhaustionRecovery {
    fn can_recover(&self, error: &AdvancedStarError) -> bool {
        matches!(error, AdvancedStarError::ResourceExhaustion { .. })
    }

    fn recover(&self, error: &AdvancedStarError, _context: &ErrorContext) -> RecoveryResult {
        if let AdvancedStarError::ResourceExhaustion { recovery_actions, .. } = error {
            // Implement resource-specific recovery
            let success = recovery_actions.iter().any(|action| {
                match action {
                    RecoveryAction::FreeMemory => {
                        // Trigger garbage collection or clear caches
                        true
                    }
                    RecoveryAction::ReduceConcurrency => {
                        // Reduce thread pool size or connection limits
                        true
                    }
                    _ => false,
                }
            });

            RecoveryResult {
                success,
                retry_recommended: true,
                delay_before_retry: Some(Duration::from_millis(2000)),
                alternative_action: Some("Resource optimization applied".to_string()),
                partial_result: None,
            }
        } else {
            RecoveryResult {
                success: false,
                retry_recommended: false,
                delay_before_retry: None,
                alternative_action: None,
                partial_result: None,
            }
        }
    }

    fn get_name(&self) -> &str {
        "resource_exhaustion_recovery"
    }
}

impl RecoveryStrategy for TimeoutErrorRecovery {
    fn can_recover(&self, error: &AdvancedStarError) -> bool {
        matches!(error, AdvancedStarError::TimeoutError { retry_recommended: true, .. })
    }

    fn recover(&self, error: &AdvancedStarError, _context: &ErrorContext) -> RecoveryResult {
        if let AdvancedStarError::TimeoutError { timeout_duration, .. } = error {
            // Increase timeout for retry
            let new_timeout = *timeout_duration * 2;

            RecoveryResult {
                success: false,
                retry_recommended: true,
                delay_before_retry: Some(Duration::from_millis(1000)),
                alternative_action: Some(format!("Retry with increased timeout: {:?}", new_timeout)),
                partial_result: None,
            }
        } else {
            RecoveryResult {
                success: false,
                retry_recommended: false,
                delay_before_retry: None,
                alternative_action: None,
                partial_result: None,
            }
        }
    }

    fn get_name(&self) -> &str {
        "timeout_error_recovery"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_error_recovery_manager_creation() {
        let config = ErrorRecoveryConfig::default();
        let manager = ErrorRecoveryManager::new(config);
        assert!(!manager.recovery_strategies.is_empty());
    }

    #[tokio::test]
    async fn test_error_enhancement() {
        let config = ErrorRecoveryConfig::default();
        let manager = ErrorRecoveryManager::new(config);

        let original_error = StarError::ParseError {
            message: "Invalid quoted triple syntax".to_string(),
            line: Some(5),
            column: Some(10),
        };

        let enhanced_error = manager.enhance_error(original_error, None);

        match enhanced_error {
            AdvancedStarError::ParseErrorWithContext { message, line, column, recoverable, .. } => {
                assert_eq!(message, "Invalid quoted triple syntax");
                assert_eq!(line, Some(5));
                assert_eq!(column, Some(10));
                assert!(recoverable);
            }
            _ => panic!("Expected ParseErrorWithContext"),
        }
    }

    #[tokio::test]
    async fn test_retry_logic() {
        let config = ErrorRecoveryConfig::default();
        let manager = ErrorRecoveryManager::new(config);

        let operation_id = "test_operation";

        // Should allow retries initially
        assert!(manager.should_retry(operation_id).await);

        // Record failures up to max retries
        for _ in 0..3 {
            manager.record_failure(operation_id).await;
        }

        // Should not allow more retries
        assert!(!manager.should_retry(operation_id).await);
    }
}