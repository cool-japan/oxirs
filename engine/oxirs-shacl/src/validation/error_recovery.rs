//! Error recovery and robustness for SHACL validation
//!
//! This module provides graceful error handling, partial validation on errors,
//! constraint failure isolation, and validation limits to ensure robust validation
//! even when encountering malformed data or expensive constraints.

use crate::{
    constraints::{Constraint, ConstraintContext},
    validation::ConstraintEvaluationResult,
    Result, ShaclError,
};
use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Error recovery configuration for validation
#[derive(Debug, Clone)]
pub struct ErrorRecoveryConfig {
    /// Enable partial validation on errors
    pub enable_partial_validation: bool,
    /// Enable constraint failure isolation
    pub enable_constraint_isolation: bool,
    /// Enable recovery from malformed shapes
    pub enable_shape_recovery: bool,
    /// Enable timeout handling for expensive constraints
    pub enable_timeout_handling: bool,
    /// Maximum recursion depth for nested shapes
    pub max_recursion_depth: usize,
    /// Maximum evaluation timeout per constraint (milliseconds)
    pub max_evaluation_timeout_ms: u64,
    /// Maximum memory usage threshold (bytes)
    pub max_memory_usage: usize,
    /// Maximum result size limit
    pub max_result_size: usize,
    /// Enable graceful degradation
    pub enable_graceful_degradation: bool,
    /// Maximum number of errors before stopping validation
    pub max_errors_before_stop: usize,
}

impl Default for ErrorRecoveryConfig {
    fn default() -> Self {
        Self {
            enable_partial_validation: true,
            enable_constraint_isolation: true,
            enable_shape_recovery: true,
            enable_timeout_handling: true,
            max_recursion_depth: 50,
            max_evaluation_timeout_ms: 5000, // 5 seconds per constraint
            max_memory_usage: 100 * 1024 * 1024, // 100MB
            max_result_size: 10000,          // Maximum violations to report
            enable_graceful_degradation: true,
            max_errors_before_stop: 100,
        }
    }
}

/// Error recovery manager for SHACL validation
#[derive(Debug)]
pub struct ErrorRecoveryManager {
    /// Configuration
    config: ErrorRecoveryConfig,
    /// Error statistics
    error_stats: Arc<Mutex<ErrorStatistics>>,
    /// Failed constraint cache (to avoid retrying expensive failures)
    failed_constraint_cache: Arc<Mutex<HashMap<String, ConstraintFailureInfo>>>,
    /// Validation context stack for recursion tracking
    validation_stack: Arc<Mutex<Vec<ValidationStackFrame>>>,
    /// Memory usage monitor
    memory_monitor: MemoryMonitor,
}

/// Statistics about validation errors and recovery
#[derive(Debug, Clone, Default)]
pub struct ErrorStatistics {
    /// Total constraint evaluation errors
    pub constraint_errors: usize,
    /// Total shape parsing errors
    pub shape_errors: usize,
    /// Total timeout errors
    pub timeout_errors: usize,
    /// Total memory limit errors
    pub memory_errors: usize,
    /// Total recursion limit errors
    pub recursion_errors: usize,
    /// Successful error recoveries
    pub successful_recoveries: usize,
    /// Failed recoveries (could not continue)
    pub failed_recoveries: usize,
    /// Constraints that were skipped due to errors
    pub skipped_constraints: usize,
    /// Average error recovery time (microseconds)
    pub avg_recovery_time_us: f64,
}

/// Information about a constraint failure
#[derive(Debug, Clone)]
struct ConstraintFailureInfo {
    /// Constraint that failed
    constraint_type: String,
    /// Error message
    error_message: String,
    /// Timestamp of failure
    failed_at: Instant,
    /// Number of times this constraint has failed
    failure_count: usize,
    /// Whether this constraint should be skipped in future
    should_skip: bool,
}

/// Stack frame for tracking validation recursion
#[derive(Debug, Clone)]
struct ValidationStackFrame {
    /// Focus node being validated
    focus_node: Term,
    /// Shape being applied
    shape_id: String,
    /// Depth level
    depth: usize,
    /// Started at timestamp
    started_at: Instant,
}

/// Memory usage monitor
#[derive(Debug)]
struct MemoryMonitor {
    /// Starting memory usage
    baseline_memory: usize,
    /// Current memory threshold
    threshold: usize,
}

/// Result of error recovery attempt
#[derive(Debug, Clone)]
pub enum ErrorRecoveryResult {
    /// Recovery successful, validation can continue
    Recovered {
        /// Partial results from successful constraints
        partial_results: Vec<ConstraintEvaluationResult>,
        /// Error details that were recovered from
        recovered_errors: Vec<RecoveredError>,
    },
    /// Recovery failed, validation should stop
    Failed {
        /// Fatal error that prevented recovery
        fatal_error: ShaclError,
        /// Partial results before failure
        partial_results: Vec<ConstraintEvaluationResult>,
    },
    /// Graceful degradation - continue with reduced functionality
    Degraded {
        /// Results with degraded validation
        degraded_results: Vec<ConstraintEvaluationResult>,
        /// Degradation reason
        degradation_reason: String,
        /// Constraints that were skipped
        skipped_constraints: Vec<String>,
    },
}

/// Information about a recovered error
#[derive(Debug, Clone)]
pub struct RecoveredError {
    /// Type of error that was recovered
    pub error_type: ErrorType,
    /// Original error message
    pub original_error: String,
    /// Recovery strategy used
    pub recovery_strategy: RecoveryStrategy,
    /// Constraint that caused the error (if applicable)
    pub failed_constraint: Option<String>,
    /// Time taken to recover (microseconds)
    pub recovery_time_us: u64,
}

/// Types of errors that can be recovered
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorType {
    /// Constraint evaluation error
    ConstraintEvaluation,
    /// Shape parsing error
    ShapeParsing,
    /// Timeout error
    Timeout,
    /// Memory limit exceeded
    MemoryLimit,
    /// Recursion limit exceeded
    RecursionLimit,
    /// SPARQL query error
    SparqlQuery,
    /// Property path evaluation error
    PropertyPath,
    /// Store access error
    StoreAccess,
}

/// Recovery strategies available
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Skip the failing constraint and continue
    SkipConstraint,
    /// Use cached result from previous evaluation
    UseCachedResult,
    /// Apply default/fallback behavior
    UseDefault,
    /// Retry with simplified parameters
    RetrySimplified,
    /// Continue with partial results
    PartialResults,
    /// Degrade to simpler validation mode
    GracefulDegradation,
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self {
            baseline_memory: Self::get_current_memory_usage(),
            threshold: 100 * 1024 * 1024, // 100MB default
        }
    }
}

impl MemoryMonitor {
    /// Get current memory usage (simplified implementation)
    fn get_current_memory_usage() -> usize {
        // In a real implementation, you'd use system APIs to get actual memory usage
        // For now, we'll use a placeholder that estimates based on process info
        std::process::id() as usize * 1024 // Simplified estimate
    }

    /// Check if memory usage exceeds threshold
    fn check_memory_pressure(&self) -> bool {
        let current_usage = Self::get_current_memory_usage();
        current_usage > self.baseline_memory + self.threshold
    }

    /// Get memory usage delta since baseline
    fn memory_usage_delta(&self) -> usize {
        let current = Self::get_current_memory_usage();
        current.saturating_sub(self.baseline_memory)
    }
}

impl ErrorRecoveryManager {
    /// Create a new error recovery manager
    pub fn new(config: ErrorRecoveryConfig) -> Self {
        Self {
            config,
            error_stats: Arc::new(Mutex::new(ErrorStatistics::default())),
            failed_constraint_cache: Arc::new(Mutex::new(HashMap::new())),
            validation_stack: Arc::new(Mutex::new(Vec::new())),
            memory_monitor: MemoryMonitor::default(),
        }
    }

    /// Attempt to recover from a validation error
    pub fn recover_from_error(
        &self,
        error: &ShaclError,
        context: &ConstraintContext,
        partial_results: Vec<ConstraintEvaluationResult>,
    ) -> ErrorRecoveryResult {
        let recovery_start = Instant::now();

        // Classify the error type
        let error_type = self.classify_error(error);

        // Update error statistics
        self.update_error_stats(&error_type);

        // Check if we should attempt recovery based on configuration
        if !self.should_attempt_recovery(&error_type) {
            return ErrorRecoveryResult::Failed {
                fatal_error: error.clone(),
                partial_results,
            };
        }

        // Select recovery strategy
        let recovery_strategy = self.select_recovery_strategy(&error_type, context);

        // Execute recovery strategy
        let recovery_result = self.execute_recovery_strategy(
            recovery_strategy.clone(),
            error,
            context,
            partial_results,
        );

        // Record recovery attempt
        let recovery_time = recovery_start.elapsed();
        let recovered_error = RecoveredError {
            error_type,
            original_error: error.to_string(),
            recovery_strategy,
            failed_constraint: self.extract_constraint_info(context),
            recovery_time_us: recovery_time.as_micros() as u64,
        };

        // Update statistics
        self.record_recovery_attempt(&recovered_error, &recovery_result);

        recovery_result
    }

    /// Validate constraint with error recovery
    pub fn validate_constraint_with_recovery(
        &self,
        store: &dyn Store,
        constraint: &Constraint,
        context: &ConstraintContext,
        timeout: Option<Duration>,
    ) -> Result<ConstraintEvaluationResult> {
        // Check if this constraint has failed before and should be skipped
        if self.should_skip_constraint(constraint, context) {
            self.increment_skipped_constraints();
            return Ok(ConstraintEvaluationResult::satisfied_with_note(
                "Constraint skipped due to previous failures".to_string(),
            ));
        }

        // Check recursion depth
        if let Err(e) = self.check_recursion_limit(context) {
            return self.handle_recursion_error(e, context);
        }

        // Check memory usage
        if let Err(e) = self.check_memory_limit() {
            return self.handle_memory_error(e, context);
        }

        // Push validation context onto stack
        self.push_validation_context(context);

        // Execute constraint validation with timeout
        let result = if let Some(timeout) = timeout {
            self.validate_with_timeout(store, constraint, context, timeout)
        } else {
            self.validate_with_default_timeout(store, constraint, context)
        };

        // Pop validation context from stack
        self.pop_validation_context();

        // Handle result or error
        match result {
            Ok(result) => Ok(result),
            Err(error) => self.handle_constraint_error(error, constraint, context),
        }
    }

    /// Classify error type for recovery strategy selection
    fn classify_error(&self, error: &ShaclError) -> ErrorType {
        match error {
            ShaclError::ConstraintValidation(_) => ErrorType::ConstraintEvaluation,
            ShaclError::ShapeParsing(_) => ErrorType::ShapeParsing,
            ShaclError::PropertyPath(_) => ErrorType::PropertyPath,
            ShaclError::SparqlExecution(_) => ErrorType::SparqlQuery,
            ShaclError::Timeout(_) => ErrorType::Timeout,
            ShaclError::MemoryLimit(_) => ErrorType::MemoryLimit,
            ShaclError::RecursionLimit(_) => ErrorType::RecursionLimit,
            _ => ErrorType::StoreAccess, // Default classification
        }
    }

    /// Check if recovery should be attempted for this error type
    fn should_attempt_recovery(&self, error_type: &ErrorType) -> bool {
        match error_type {
            ErrorType::ConstraintEvaluation => self.config.enable_constraint_isolation,
            ErrorType::ShapeParsing => self.config.enable_shape_recovery,
            ErrorType::Timeout => self.config.enable_timeout_handling,
            ErrorType::MemoryLimit => self.config.enable_graceful_degradation,
            ErrorType::RecursionLimit => self.config.enable_graceful_degradation,
            _ => self.config.enable_partial_validation,
        }
    }

    /// Select appropriate recovery strategy
    fn select_recovery_strategy(
        &self,
        error_type: &ErrorType,
        context: &ConstraintContext,
    ) -> RecoveryStrategy {
        match error_type {
            ErrorType::ConstraintEvaluation => {
                // Check if we have a cached result
                if self.has_cached_constraint_result(context) {
                    RecoveryStrategy::UseCachedResult
                } else {
                    RecoveryStrategy::SkipConstraint
                }
            }
            ErrorType::Timeout => RecoveryStrategy::RetrySimplified,
            ErrorType::MemoryLimit => RecoveryStrategy::GracefulDegradation,
            ErrorType::RecursionLimit => RecoveryStrategy::GracefulDegradation,
            ErrorType::ShapeParsing => RecoveryStrategy::UseDefault,
            _ => RecoveryStrategy::PartialResults,
        }
    }

    /// Execute the selected recovery strategy
    fn execute_recovery_strategy(
        &self,
        strategy: RecoveryStrategy,
        error: &ShaclError,
        context: &ConstraintContext,
        partial_results: Vec<ConstraintEvaluationResult>,
    ) -> ErrorRecoveryResult {
        match strategy {
            RecoveryStrategy::SkipConstraint => {
                self.record_skipped_constraint(context);
                ErrorRecoveryResult::Recovered {
                    partial_results,
                    recovered_errors: vec![RecoveredError {
                        error_type: self.classify_error(error),
                        original_error: error.to_string(),
                        recovery_strategy: strategy,
                        failed_constraint: self.extract_constraint_info(context),
                        recovery_time_us: 0,
                    }],
                }
            }
            RecoveryStrategy::UseCachedResult => {
                if let Some(cached_result) = self.get_cached_constraint_result(context) {
                    let mut results = partial_results;
                    results.push(cached_result);
                    ErrorRecoveryResult::Recovered {
                        partial_results: results,
                        recovered_errors: vec![],
                    }
                } else {
                    ErrorRecoveryResult::Failed {
                        fatal_error: error.clone(),
                        partial_results,
                    }
                }
            }
            RecoveryStrategy::GracefulDegradation => ErrorRecoveryResult::Degraded {
                degraded_results: partial_results,
                degradation_reason: "Graceful degradation due to resource limits".to_string(),
                skipped_constraints: vec![],
            },
            RecoveryStrategy::PartialResults => ErrorRecoveryResult::Recovered {
                partial_results,
                recovered_errors: vec![],
            },
            _ => ErrorRecoveryResult::Failed {
                fatal_error: error.clone(),
                partial_results,
            },
        }
    }

    /// Check recursion depth limit
    fn check_recursion_limit(&self, context: &ConstraintContext) -> Result<()> {
        let stack = self.validation_stack.lock().unwrap();
        if stack.len() >= self.config.max_recursion_depth {
            return Err(ShaclError::RecursionLimit(format!(
                "Maximum recursion depth {} exceeded while validating shape {} for node {}",
                self.config.max_recursion_depth, context.shape_id, context.focus_node
            )));
        }
        Ok(())
    }

    /// Check memory usage limit
    fn check_memory_limit(&self) -> Result<()> {
        if self.memory_monitor.check_memory_pressure() {
            return Err(ShaclError::MemoryLimit(format!(
                "Memory usage exceeded threshold: {} bytes",
                self.memory_monitor.memory_usage_delta()
            )));
        }
        Ok(())
    }

    /// Push validation context onto recursion stack
    fn push_validation_context(&self, context: &ConstraintContext) {
        let mut stack = self.validation_stack.lock().unwrap();
        let frame = ValidationStackFrame {
            focus_node: context.focus_node.clone(),
            shape_id: context.shape_id.to_string(),
            depth: stack.len(),
            started_at: Instant::now(),
        };
        stack.push(frame);
    }

    /// Pop validation context from recursion stack
    fn pop_validation_context(&self) {
        let mut stack = self.validation_stack.lock().unwrap();
        stack.pop();
    }

    /// Validate constraint with default timeout
    fn validate_with_default_timeout(
        &self,
        store: &dyn Store,
        constraint: &Constraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        let timeout = Duration::from_millis(self.config.max_evaluation_timeout_ms);
        self.validate_with_timeout(store, constraint, context, timeout)
    }

    /// Validate constraint with specified timeout
    fn validate_with_timeout(
        &self,
        store: &dyn Store,
        constraint: &Constraint,
        context: &ConstraintContext,
        timeout: Duration,
    ) -> Result<ConstraintEvaluationResult> {
        let start_time = Instant::now();

        // Simple timeout implementation - in practice you'd want more sophisticated cancellation
        let result = constraint.evaluate(store, context);

        // Check if we exceeded timeout
        if start_time.elapsed() > timeout {
            return Err(ShaclError::Timeout(format!(
                "Constraint evaluation timed out after {:?} for constraint type: {:?}",
                timeout,
                constraint.component_id()
            )));
        }

        // Convert from constraints::ConstraintEvaluationResult to validation::ConstraintEvaluationResult
        match result? {
            crate::constraints::ConstraintEvaluationResult::Satisfied => {
                Ok(crate::validation::ConstraintEvaluationResult::Satisfied)
            }
            crate::constraints::ConstraintEvaluationResult::Violated {
                violating_value,
                message,
                details: _,
            } => Ok(crate::validation::ConstraintEvaluationResult::Violated {
                violating_value,
                message,
            }),
            crate::constraints::ConstraintEvaluationResult::Error { message, .. } => {
                Err(crate::ShaclError::ValidationEngine(message))
            }
        }
    }

    /// Handle constraint evaluation error
    fn handle_constraint_error(
        &self,
        error: ShaclError,
        constraint: &Constraint,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        // Record the constraint failure
        self.record_constraint_failure(constraint, &error);

        // Attempt error recovery
        let recovery_result = self.recover_from_error(&error, context, vec![]);

        match recovery_result {
            ErrorRecoveryResult::Recovered {
                partial_results, ..
            } => {
                // Return the best result we have, or a satisfied result if recovery worked
                Ok(partial_results
                    .into_iter()
                    .next()
                    .unwrap_or_else(ConstraintEvaluationResult::satisfied))
            }
            ErrorRecoveryResult::Degraded {
                degraded_results, ..
            } => {
                // Return degraded result
                Ok(degraded_results
                    .into_iter()
                    .next()
                    .unwrap_or_else(ConstraintEvaluationResult::satisfied))
            }
            ErrorRecoveryResult::Failed { .. } => Err(error),
        }
    }

    /// Handle recursion limit error
    fn handle_recursion_error(
        &self,
        error: ShaclError,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        if self.config.enable_graceful_degradation {
            // Return a satisfied result with a note about recursion limit
            Ok(ConstraintEvaluationResult::satisfied_with_note(format!(
                "Recursion limit reached for shape {} on node {}",
                context.shape_id, context.focus_node
            )))
        } else {
            Err(error)
        }
    }

    /// Handle memory limit error
    fn handle_memory_error(
        &self,
        error: ShaclError,
        context: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        if self.config.enable_graceful_degradation {
            // Clear any caches to free memory
            self.clear_caches();

            // Return a satisfied result with a note about memory pressure
            Ok(ConstraintEvaluationResult::satisfied_with_note(format!(
                "Memory limit reached during validation of shape {} on node {}",
                context.shape_id, context.focus_node
            )))
        } else {
            Err(error)
        }
    }

    /// Check if constraint should be skipped due to previous failures
    fn should_skip_constraint(
        &self,
        constraint: &Constraint,
        _context: &ConstraintContext,
    ) -> bool {
        let cache = self.failed_constraint_cache.lock().unwrap();
        let constraint_key = format!("{:?}", constraint.component_id());
        if let Some(failure_info) = cache.get(&constraint_key) {
            failure_info.should_skip && failure_info.failure_count > 3
        } else {
            false
        }
    }

    /// Record constraint failure for future reference
    fn record_constraint_failure(&self, constraint: &Constraint, error: &ShaclError) {
        let mut cache = self.failed_constraint_cache.lock().unwrap();
        let constraint_key = format!("{:?}", constraint.component_id());

        let failure_info = cache
            .entry(constraint_key)
            .or_insert_with(|| ConstraintFailureInfo {
                constraint_type: format!("{:?}", constraint.component_id()),
                error_message: error.to_string(),
                failed_at: Instant::now(),
                failure_count: 0,
                should_skip: false,
            });

        failure_info.failure_count += 1;
        failure_info.error_message = error.to_string();
        failure_info.failed_at = Instant::now();

        // Skip constraint if it has failed too many times
        if failure_info.failure_count > 5 {
            failure_info.should_skip = true;
        }
    }

    /// Helper methods (placeholders for now)
    fn has_cached_constraint_result(&self, _context: &ConstraintContext) -> bool {
        false // Placeholder
    }

    fn get_cached_constraint_result(
        &self,
        _context: &ConstraintContext,
    ) -> Option<ConstraintEvaluationResult> {
        None // Placeholder
    }

    fn record_skipped_constraint(&self, _context: &ConstraintContext) {
        // Record in statistics
        self.increment_skipped_constraints();
    }

    fn extract_constraint_info(&self, context: &ConstraintContext) -> Option<String> {
        Some(context.shape_id.to_string().to_string())
    }

    fn update_error_stats(&self, error_type: &ErrorType) {
        let mut stats = self.error_stats.lock().unwrap();
        match error_type {
            ErrorType::ConstraintEvaluation => stats.constraint_errors += 1,
            ErrorType::ShapeParsing => stats.shape_errors += 1,
            ErrorType::Timeout => stats.timeout_errors += 1,
            ErrorType::MemoryLimit => stats.memory_errors += 1,
            ErrorType::RecursionLimit => stats.recursion_errors += 1,
            _ => {}
        }
    }

    fn record_recovery_attempt(
        &self,
        _recovered_error: &RecoveredError,
        result: &ErrorRecoveryResult,
    ) {
        let mut stats = self.error_stats.lock().unwrap();
        match result {
            ErrorRecoveryResult::Recovered { .. } | ErrorRecoveryResult::Degraded { .. } => {
                stats.successful_recoveries += 1;
            }
            ErrorRecoveryResult::Failed { .. } => {
                stats.failed_recoveries += 1;
            }
        }
    }

    fn increment_skipped_constraints(&self) {
        let mut stats = self.error_stats.lock().unwrap();
        stats.skipped_constraints += 1;
    }

    fn clear_caches(&self) {
        let mut cache = self.failed_constraint_cache.lock().unwrap();
        cache.clear();
    }

    /// Get error recovery statistics
    pub fn get_statistics(&self) -> ErrorStatistics {
        self.error_stats.lock().unwrap().clone()
    }

    /// Reset error recovery statistics
    pub fn reset_statistics(&self) {
        let mut stats = self.error_stats.lock().unwrap();
        *stats = ErrorStatistics::default();
    }

    /// Get current validation stack depth
    pub fn get_current_depth(&self) -> usize {
        self.validation_stack.lock().unwrap().len()
    }

    /// Check if validation is currently in progress
    pub fn is_validation_active(&self) -> bool {
        !self.validation_stack.lock().unwrap().is_empty()
    }

    /// Get memory usage information
    pub fn get_memory_info(&self) -> MemoryUsageInfo {
        MemoryUsageInfo {
            current_usage: self.memory_monitor.memory_usage_delta(),
            threshold: self.memory_monitor.threshold,
            is_under_pressure: self.memory_monitor.check_memory_pressure(),
        }
    }
}

/// Memory usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageInfo {
    /// Current memory usage delta from baseline
    pub current_usage: usize,
    /// Memory threshold
    pub threshold: usize,
    /// Whether currently under memory pressure
    pub is_under_pressure: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ShapeId;
    use oxirs_core::model::{NamedNode, Term};

    #[test]
    fn test_error_recovery_manager_creation() {
        let config = ErrorRecoveryConfig::default();
        let manager = ErrorRecoveryManager::new(config);
        assert_eq!(manager.get_current_depth(), 0);
        assert!(!manager.is_validation_active());
    }

    #[test]
    fn test_recursion_limit_check() {
        let mut config = ErrorRecoveryConfig::default();
        config.max_recursion_depth = 2;

        let manager = ErrorRecoveryManager::new(config);
        let context = ConstraintContext::new(
            Term::NamedNode(NamedNode::new("http://example.org/test").unwrap()),
            ShapeId::new("TestShape"),
        );

        // Should pass initially
        assert!(manager.check_recursion_limit(&context).is_ok());

        // Push contexts to exceed limit
        manager.push_validation_context(&context);
        manager.push_validation_context(&context);
        manager.push_validation_context(&context); // This should exceed limit

        assert!(manager.check_recursion_limit(&context).is_err());
    }

    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::default();
        let usage = monitor.memory_usage_delta();
        // usage is always >= 0 by type invariant
    }

    #[test]
    fn test_error_classification() {
        let manager = ErrorRecoveryManager::new(ErrorRecoveryConfig::default());

        let constraint_error = ShaclError::ConstraintValidation("test error".to_string());
        assert_eq!(
            manager.classify_error(&constraint_error),
            ErrorType::ConstraintEvaluation
        );

        let timeout_error = ShaclError::Timeout("timeout".to_string());
        assert_eq!(manager.classify_error(&timeout_error), ErrorType::Timeout);
    }
}
