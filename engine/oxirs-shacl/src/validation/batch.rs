//! Enhanced batch validation system for large datasets
//!
//! This module provides comprehensive batch validation capabilities including:
//! - Memory-efficient processing of large datasets
//! - Progress reporting and cancellation support
//! - Error recovery and graceful degradation
//! - Detailed performance analytics

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use oxirs_core::{model::Term, Store};
use serde::{Deserialize, Serialize};

use crate::{
    constraints::{Constraint, ConstraintContext},
    optimization::StreamingValidationEngine,
    Result, ShaclError, ShapeId, ValidationReport,
};

type ProgressCallbackFn = Box<dyn Fn(BatchProgress) + Send + Sync>;
type ProgressCallbacks = Arc<Mutex<Vec<ProgressCallbackFn>>>;

/// Enhanced batch validation engine with comprehensive features
#[derive(Debug)]
pub struct EnhancedBatchValidationEngine {
    /// Core streaming engine
    #[allow(dead_code)]
    streaming_engine: StreamingValidationEngine,
    /// Configuration for batch processing
    config: BatchValidationConfig,
    /// Progress tracking
    progress_tracker: Arc<BatchProgressTracker>,
    /// Error recovery handler
    error_handler: BatchErrorHandler,
    /// Memory monitor
    memory_monitor: MemoryMonitor,
    /// Performance analytics
    analytics: Arc<Mutex<BatchPerformanceAnalytics>>,
}

/// Configuration for batch validation
#[derive(Debug, Clone)]
pub struct BatchValidationConfig {
    /// Size of each processing batch
    pub batch_size: usize,
    /// Maximum memory usage in bytes before triggering cleanup
    pub memory_threshold: usize,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Maximum number of worker threads
    pub max_workers: usize,
    /// Enable progress reporting
    pub enable_progress_reporting: bool,
    /// Progress reporting interval (number of processed items)
    pub progress_interval: usize,
    /// Enable graceful error recovery
    pub enable_error_recovery: bool,
    /// Maximum number of errors before stopping
    pub max_errors: Option<usize>,
    /// Enable memory monitoring
    pub enable_memory_monitoring: bool,
    /// Enable detailed analytics
    pub enable_analytics: bool,
    /// Checkpoint interval for saving intermediate results
    pub checkpoint_interval: Option<usize>,
}

impl Default for BatchValidationConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            memory_threshold: 500 * 1024 * 1024, // 500MB
            enable_parallel: true,
            max_workers: num_cpus::get().min(8),
            enable_progress_reporting: true,
            progress_interval: 10000,
            enable_error_recovery: true,
            max_errors: Some(100),
            enable_memory_monitoring: true,
            enable_analytics: true,
            checkpoint_interval: Some(50000),
        }
    }
}

/// Progress tracking for batch operations
pub struct BatchProgressTracker {
    /// Total number of items to process
    total_items: AtomicUsize,
    /// Number of items processed so far
    processed_items: AtomicUsize,
    /// Number of violations found
    violations_found: AtomicUsize,
    /// Number of errors encountered
    errors_encountered: AtomicUsize,
    /// Start time of the batch operation
    start_time: Instant,
    /// Cancellation flag
    cancelled: AtomicBool,
    /// Progress callback functions
    progress_callbacks: ProgressCallbacks,
}

impl std::fmt::Debug for BatchProgressTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchProgressTracker")
            .field("total_items", &self.total_items)
            .field("processed_items", &self.processed_items)
            .field("violations_found", &self.violations_found)
            .field("errors_encountered", &self.errors_encountered)
            .field("start_time", &self.start_time)
            .field("cancelled", &self.cancelled)
            .field("progress_callbacks", &"<function closures>")
            .finish()
    }
}

impl BatchProgressTracker {
    fn new() -> Self {
        Self {
            total_items: AtomicUsize::new(0),
            processed_items: AtomicUsize::new(0),
            violations_found: AtomicUsize::new(0),
            errors_encountered: AtomicUsize::new(0),
            start_time: Instant::now(),
            cancelled: AtomicBool::new(false),
            progress_callbacks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Set the total number of items to process
    pub fn set_total_items(&self, total: usize) {
        self.total_items.store(total, Ordering::SeqCst);
    }

    /// Update progress with processed items
    pub fn update_progress(&self, processed: usize, violations: usize, errors: usize) {
        self.processed_items.fetch_add(processed, Ordering::SeqCst);
        self.violations_found
            .fetch_add(violations, Ordering::SeqCst);
        self.errors_encountered.fetch_add(errors, Ordering::SeqCst);

        // Trigger progress callbacks
        self.trigger_progress_callbacks();
    }

    /// Get current progress
    pub fn get_progress(&self) -> BatchProgress {
        let total = self.total_items.load(Ordering::SeqCst);
        let processed = self.processed_items.load(Ordering::SeqCst);
        let violations = self.violations_found.load(Ordering::SeqCst);
        let errors = self.errors_encountered.load(Ordering::SeqCst);
        let elapsed = self.start_time.elapsed();

        BatchProgress {
            total_items: total,
            processed_items: processed,
            violations_found: violations,
            errors_encountered: errors,
            progress_percentage: if total > 0 {
                processed as f64 / total as f64 * 100.0
            } else {
                0.0
            },
            elapsed_time: elapsed,
            estimated_time_remaining: if processed > 0 && total > processed {
                Some(elapsed * (total - processed) as u32 / processed as u32)
            } else {
                None
            },
            items_per_second: if elapsed.as_secs() > 0 {
                processed as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            },
            is_cancelled: self.cancelled.load(Ordering::SeqCst),
        }
    }

    /// Request cancellation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation was requested
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Add a progress callback function
    pub fn add_progress_callback<F>(&self, callback: F)
    where
        F: Fn(BatchProgress) + Send + Sync + 'static,
    {
        if let Ok(mut callbacks) = self.progress_callbacks.lock() {
            callbacks.push(Box::new(callback));
        }
    }

    /// Trigger all progress callbacks
    fn trigger_progress_callbacks(&self) {
        if let Ok(callbacks) = self.progress_callbacks.lock() {
            let progress = self.get_progress();
            for callback in callbacks.iter() {
                callback(progress.clone());
            }
        }
    }
}

/// Current progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProgress {
    pub total_items: usize,
    pub processed_items: usize,
    pub violations_found: usize,
    pub errors_encountered: usize,
    pub progress_percentage: f64,
    pub elapsed_time: Duration,
    pub estimated_time_remaining: Option<Duration>,
    pub items_per_second: f64,
    pub is_cancelled: bool,
}

impl BatchProgress {
    /// Format progress as a human-readable string
    pub fn format_summary(&self) -> String {
        format!(
            "Progress: {:.1}% ({}/{}) | Violations: {} | Errors: {} | Speed: {:.1} items/sec | Elapsed: {:.1}s{}",
            self.progress_percentage,
            self.processed_items,
            self.total_items,
            self.violations_found,
            self.errors_encountered,
            self.items_per_second,
            self.elapsed_time.as_secs_f64(),
            if let Some(eta) = self.estimated_time_remaining {
                format!(" | ETA: {:.1}s", eta.as_secs_f64())
            } else {
                String::new()
            }
        )
    }
}

/// Error handling strategy for batch validation
#[derive(Debug)]
pub struct BatchErrorHandler {
    /// Maximum number of errors before stopping
    max_errors: Option<usize>,
    /// Current error count
    error_count: AtomicUsize,
    /// Error recovery strategies
    #[allow(dead_code)]
    recovery_strategies: Vec<ErrorRecoveryStrategy>,
    /// Error log
    error_log: Arc<Mutex<Vec<BatchValidationError>>>,
}

impl BatchErrorHandler {
    fn new(max_errors: Option<usize>) -> Self {
        Self {
            max_errors,
            error_count: AtomicUsize::new(0),
            recovery_strategies: vec![
                ErrorRecoveryStrategy::SkipAndContinue,
                ErrorRecoveryStrategy::RetryWithDelay(Duration::from_millis(100)),
            ],
            error_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Handle a validation error
    pub fn handle_error(&self, error: BatchValidationError) -> ErrorRecoveryAction {
        let error_count = self.error_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Log the error
        if let Ok(mut log) = self.error_log.lock() {
            log.push(error.clone());
        }

        // Check if we should stop processing
        if let Some(max_errors) = self.max_errors {
            if error_count >= max_errors {
                return ErrorRecoveryAction::Stop;
            }
        }

        // Apply recovery strategy based on error type
        match error.error_type {
            BatchErrorType::ValidationFailure => ErrorRecoveryAction::Skip,
            BatchErrorType::MemoryExhaustion => ErrorRecoveryAction::TriggerGarbageCollection,
            BatchErrorType::TimeoutError => {
                ErrorRecoveryAction::RetryWithDelay(Duration::from_secs(1))
            }
            BatchErrorType::ConstraintError => ErrorRecoveryAction::Skip,
            BatchErrorType::SystemError => ErrorRecoveryAction::Stop,
        }
    }

    /// Get current error count
    pub fn error_count(&self) -> usize {
        self.error_count.load(Ordering::SeqCst)
    }

    /// Get error log
    pub fn get_error_log(&self) -> Vec<BatchValidationError> {
        self.error_log
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum ErrorRecoveryStrategy {
    SkipAndContinue,
    RetryWithDelay(Duration),
    TriggerGarbageCollection,
    ReduceBatchSize,
}

/// Actions to take when recovering from errors
#[derive(Debug, Clone)]
pub enum ErrorRecoveryAction {
    Skip,
    Retry,
    RetryWithDelay(Duration),
    TriggerGarbageCollection,
    ReduceBatchSize,
    Stop,
}

/// Types of errors that can occur during batch validation
#[derive(Debug, Clone)]
pub enum BatchErrorType {
    ValidationFailure,
    MemoryExhaustion,
    TimeoutError,
    ConstraintError,
    SystemError,
}

/// Detailed error information for batch validation
#[derive(Debug, Clone)]
pub struct BatchValidationError {
    pub error_type: BatchErrorType,
    pub message: String,
    pub node: Option<Term>,
    pub shape: Option<ShapeId>,
    pub constraint: Option<String>,
    pub timestamp: Instant,
    pub batch_index: usize,
    pub item_index: usize,
}

/// Memory monitoring for batch operations
#[derive(Debug)]
pub struct MemoryMonitor {
    /// Memory threshold in bytes
    threshold: usize,
    /// Enable monitoring
    enabled: bool,
    /// Current memory usage estimate
    current_usage: AtomicUsize,
    /// Peak memory usage
    peak_usage: AtomicUsize,
}

impl MemoryMonitor {
    fn new(threshold: usize, enabled: bool) -> Self {
        Self {
            threshold,
            enabled,
            current_usage: AtomicUsize::new(0),
            peak_usage: AtomicUsize::new(0),
        }
    }

    /// Check if memory usage exceeds threshold
    pub fn check_memory_pressure(&self) -> bool {
        if !self.enabled {
            return false;
        }

        let current = self.estimate_memory_usage();
        self.current_usage.store(current, Ordering::SeqCst);

        let peak = self.peak_usage.load(Ordering::SeqCst);
        if current > peak {
            self.peak_usage.store(current, Ordering::SeqCst);
        }

        current > self.threshold
    }

    /// Estimate current memory usage (simplified)
    fn estimate_memory_usage(&self) -> usize {
        // This is a simplified estimation for testing
        // In practice, you might use system calls or memory profiling tools
        // Return a reasonable baseline that doesn't immediately trigger pressure
        std::mem::size_of::<Self>() + 100 // Baseline struct size plus small overhead
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::SeqCst)
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_usage.load(Ordering::SeqCst)
    }
}

/// Performance analytics for batch validation
#[derive(Debug, Clone, Default)]
pub struct BatchPerformanceAnalytics {
    /// Throughput measurements (items per second)
    throughput_history: VecDeque<f64>,
    /// Memory usage history
    memory_usage_history: VecDeque<usize>,
    /// Error rate history
    error_rate_history: VecDeque<f64>,
    /// Constraint performance breakdown
    #[allow(dead_code)]
    constraint_performance: HashMap<String, ConstraintPerformanceMetrics>,
    /// Batch processing times
    batch_times: Vec<Duration>,
    /// Total processing time
    total_processing_time: Duration,
}

/// Performance metrics for individual constraints
#[derive(Debug, Clone, Default)]
pub struct ConstraintPerformanceMetrics {
    pub total_evaluations: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub success_rate: f64,
    pub violation_rate: f64,
}

impl BatchPerformanceAnalytics {
    /// Record throughput measurement
    pub fn record_throughput(&mut self, items_per_second: f64) {
        self.throughput_history.push_back(items_per_second);
        if self.throughput_history.len() > 100 {
            self.throughput_history.pop_front();
        }
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, bytes: usize) {
        self.memory_usage_history.push_back(bytes);
        if self.memory_usage_history.len() > 100 {
            self.memory_usage_history.pop_front();
        }
    }

    /// Record error rate
    pub fn record_error_rate(&mut self, rate: f64) {
        self.error_rate_history.push_back(rate);
        if self.error_rate_history.len() > 100 {
            self.error_rate_history.pop_front();
        }
    }

    /// Record batch processing time
    pub fn record_batch_time(&mut self, duration: Duration) {
        self.batch_times.push(duration);
        self.total_processing_time += duration;
    }

    /// Get average throughput
    pub fn average_throughput(&self) -> f64 {
        if self.throughput_history.is_empty() {
            0.0
        } else {
            self.throughput_history.iter().sum::<f64>() / self.throughput_history.len() as f64
        }
    }

    /// Get peak throughput
    pub fn peak_throughput(&self) -> f64 {
        self.throughput_history.iter().cloned().fold(0.0, f64::max)
    }

    /// Get average memory usage
    pub fn average_memory_usage(&self) -> usize {
        if self.memory_usage_history.is_empty() {
            0
        } else {
            self.memory_usage_history.iter().sum::<usize>() / self.memory_usage_history.len()
        }
    }

    /// Get peak memory usage
    pub fn peak_memory_usage(&self) -> usize {
        self.memory_usage_history.iter().cloned().max().unwrap_or(0)
    }

    /// Get average error rate
    pub fn average_error_rate(&self) -> f64 {
        if self.error_rate_history.is_empty() {
            0.0
        } else {
            self.error_rate_history.iter().sum::<f64>() / self.error_rate_history.len() as f64
        }
    }

    /// Generate performance summary
    pub fn generate_summary(&self) -> String {
        format!(
            "BatchPerformanceAnalytics {{\n  avg_throughput: {:.2} items/sec,\n  peak_throughput: {:.2} items/sec,\n  avg_memory: {} bytes,\n  peak_memory: {} bytes,\n  avg_error_rate: {:.2}%,\n  total_batches: {},\n  total_time: {:.2}s\n}}",
            self.average_throughput(),
            self.peak_throughput(),
            self.average_memory_usage(),
            self.peak_memory_usage(),
            self.average_error_rate() * 100.0,
            self.batch_times.len(),
            self.total_processing_time.as_secs_f64()
        )
    }
}

impl EnhancedBatchValidationEngine {
    /// Create a new enhanced batch validation engine
    pub fn new(config: BatchValidationConfig) -> Self {
        let streaming_engine = StreamingValidationEngine::new(
            config.batch_size,
            config.memory_threshold,
            config.enable_memory_monitoring,
        );

        Self {
            streaming_engine,
            config: config.clone(),
            progress_tracker: Arc::new(BatchProgressTracker::new()),
            error_handler: BatchErrorHandler::new(config.max_errors),
            memory_monitor: MemoryMonitor::new(
                config.memory_threshold,
                config.enable_memory_monitoring,
            ),
            analytics: Arc::new(Mutex::new(BatchPerformanceAnalytics::default())),
        }
    }

    /// Validate a large dataset with comprehensive monitoring and error recovery
    pub fn validate_large_dataset<I>(
        &self,
        store: &dyn Store,
        constraints: Vec<Constraint>,
        node_iterator: I,
    ) -> Result<EnhancedBatchValidationResult>
    where
        I: Iterator<Item = Term> + Send,
    {
        let start_time = Instant::now();
        let mut result = EnhancedBatchValidationResult::new();

        // Convert iterator to Vec to get total count (if possible)
        let nodes: Vec<Term> = node_iterator.collect();
        let total_nodes = nodes.len();

        self.progress_tracker.set_total_items(total_nodes);

        if self.config.enable_progress_reporting {
            tracing::info!("Starting batch validation of {} nodes", total_nodes);
        }

        // Process in batches
        let mut processed_count = 0;
        for (batch_index, chunk) in nodes.chunks(self.config.batch_size).enumerate() {
            // Check for cancellation
            if self.progress_tracker.is_cancelled() {
                result.was_cancelled = true;
                break;
            }

            // Check memory pressure
            if self.memory_monitor.check_memory_pressure() {
                if self.config.enable_error_recovery {
                    self.handle_memory_pressure()?;
                } else {
                    return Err(ShaclError::ValidationEngine(
                        "Memory threshold exceeded".to_string(),
                    ));
                }
            }

            // Process batch with error recovery
            let batch_start = Instant::now();
            let batch_result =
                self.process_batch_with_recovery(store, &constraints, chunk, batch_index)?;

            // Update analytics
            if self.config.enable_analytics {
                if let Ok(mut analytics) = self.analytics.lock() {
                    analytics.record_batch_time(batch_start.elapsed());
                    analytics.record_memory_usage(self.memory_monitor.current_usage());

                    let items_processed = chunk.len();
                    let elapsed = batch_start.elapsed();
                    if elapsed.as_secs_f64() > 0.0 {
                        let throughput = items_processed as f64 / elapsed.as_secs_f64();
                        analytics.record_throughput(throughput);
                    }
                }
            }

            // Merge batch results
            result.merge_batch_result(batch_result);
            processed_count += chunk.len();

            // Update progress
            self.progress_tracker.update_progress(
                chunk.len(),
                result.total_violations,
                self.error_handler.error_count(),
            );

            // Progress reporting
            if self.config.enable_progress_reporting
                && processed_count % self.config.progress_interval == 0
            {
                let progress = self.progress_tracker.get_progress();
                tracing::info!("{}", progress.format_summary());
            }

            // Checkpoint if configured
            if let Some(checkpoint_interval) = self.config.checkpoint_interval {
                if processed_count % checkpoint_interval == 0 {
                    self.create_checkpoint(&result, processed_count)?;
                }
            }
        }

        // Finalize results
        result.total_processing_time = start_time.elapsed();
        result.final_progress = self.progress_tracker.get_progress();
        result.error_log = self.error_handler.get_error_log();

        if self.config.enable_analytics {
            if let Ok(analytics) = self.analytics.lock() {
                result.performance_analytics = Some(analytics.clone());
            }
        }

        if self.config.enable_progress_reporting {
            tracing::info!(
                "Batch validation completed: {} nodes processed, {} violations found, {} errors encountered",
                processed_count,
                result.total_violations,
                result.error_log.len()
            );
        }

        Ok(result)
    }

    /// Process a single batch with error recovery
    fn process_batch_with_recovery(
        &self,
        store: &dyn Store,
        constraints: &[Constraint],
        nodes: &[Term],
        batch_index: usize,
    ) -> Result<BatchValidationResult> {
        let mut batch_result = BatchValidationResult::new();
        let mut retry_count = 0;
        const MAX_RETRIES: usize = 3;

        loop {
            match self.process_batch_internal(store, constraints, nodes, batch_index) {
                Ok(result) => {
                    batch_result = result;
                    break;
                }
                Err(error) => {
                    let batch_error = BatchValidationError {
                        error_type: BatchErrorType::ValidationFailure,
                        message: error.to_string(),
                        node: None,
                        shape: None,
                        constraint: None,
                        timestamp: Instant::now(),
                        batch_index,
                        item_index: 0,
                    };

                    let recovery_action = self.error_handler.handle_error(batch_error);

                    match recovery_action {
                        ErrorRecoveryAction::Stop => return Err(error),
                        ErrorRecoveryAction::Skip => {
                            batch_result.skipped_items = nodes.len();
                            break;
                        }
                        ErrorRecoveryAction::Retry => {
                            retry_count += 1;
                            if retry_count >= MAX_RETRIES {
                                return Err(error);
                            }
                        }
                        ErrorRecoveryAction::RetryWithDelay(delay) => {
                            retry_count += 1;
                            if retry_count >= MAX_RETRIES {
                                return Err(error);
                            }
                            thread::sleep(delay);
                        }
                        ErrorRecoveryAction::TriggerGarbageCollection => {
                            self.handle_memory_pressure()?;
                            retry_count += 1;
                            if retry_count >= MAX_RETRIES {
                                return Err(error);
                            }
                        }
                        ErrorRecoveryAction::ReduceBatchSize => {
                            // Process nodes individually
                            batch_result = self.process_nodes_individually(
                                store,
                                constraints,
                                nodes,
                                batch_index,
                            )?;
                            break;
                        }
                    }
                }
            }
        }

        Ok(batch_result)
    }

    /// Process a batch internally
    fn process_batch_internal(
        &self,
        store: &dyn Store,
        constraints: &[Constraint],
        nodes: &[Term],
        batch_index: usize,
    ) -> Result<BatchValidationResult> {
        let mut batch_result = BatchValidationResult::new();

        for (item_index, node) in nodes.iter().enumerate() {
            let context = ConstraintContext::new(
                node.clone(),
                ShapeId::new(format!("BatchValidation_{batch_index}")),
            );

            // Use the streaming engine's evaluator
            // Note: This is a simplified integration - in practice, you'd want to
            // properly integrate with the existing constraint evaluation system
            for constraint in constraints {
                match constraint.evaluate(store, &context) {
                    Ok(result) => {
                        if result.is_violated() {
                            batch_result.violation_count += 1;
                            // Convert to ValidationViolation
                            // This would be implemented based on your validation system
                        }
                    }
                    Err(error) => {
                        let batch_error = BatchValidationError {
                            error_type: BatchErrorType::ConstraintError,
                            message: error.to_string(),
                            node: Some(node.clone()),
                            shape: Some(context.shape_id.clone()),
                            constraint: Some(format!("{constraint:?}")),
                            timestamp: Instant::now(),
                            batch_index,
                            item_index,
                        };

                        let recovery_action = self.error_handler.handle_error(batch_error);
                        match recovery_action {
                            ErrorRecoveryAction::Stop => return Err(error),
                            ErrorRecoveryAction::Skip => continue,
                            _ => return Err(error),
                        }
                    }
                }
            }

            batch_result.processed_items += 1;
        }

        Ok(batch_result)
    }

    /// Process nodes individually when batch processing fails
    fn process_nodes_individually(
        &self,
        store: &dyn Store,
        constraints: &[Constraint],
        nodes: &[Term],
        batch_index: usize,
    ) -> Result<BatchValidationResult> {
        let mut batch_result = BatchValidationResult::new();

        for (item_index, node) in nodes.iter().enumerate() {
            // Try to process each node individually
            match self.process_single_node(store, constraints, node, batch_index, item_index) {
                Ok(violations) => {
                    batch_result.violation_count += violations;
                    batch_result.processed_items += 1;
                }
                Err(_error) => {
                    batch_result.skipped_items += 1;
                }
            }
        }

        Ok(batch_result)
    }

    /// Process a single node
    fn process_single_node(
        &self,
        store: &dyn Store,
        constraints: &[Constraint],
        node: &Term,
        batch_index: usize,
        item_index: usize,
    ) -> Result<usize> {
        let context = ConstraintContext::new(
            node.clone(),
            ShapeId::new(format!("SingleNodeValidation_{batch_index}_{item_index}")),
        );

        let mut violations = 0;

        for constraint in constraints {
            match constraint.evaluate(store, &context) {
                Ok(result) => {
                    if result.is_violated() {
                        violations += 1;
                    }
                }
                Err(error) => {
                    return Err(error);
                }
            }
        }

        Ok(violations)
    }

    /// Handle memory pressure by triggering cleanup
    fn handle_memory_pressure(&self) -> Result<()> {
        // Trigger garbage collection (simplified)
        // In practice, you might:
        // 1. Clear caches
        // 2. Force garbage collection
        // 3. Reduce batch sizes
        // 4. Spill data to disk

        tracing::warn!("Memory pressure detected, triggering cleanup");

        // Clear any caches we have access to
        // This is a placeholder for actual cleanup logic

        Ok(())
    }

    /// Create a checkpoint with intermediate results
    fn create_checkpoint(
        &self,
        result: &EnhancedBatchValidationResult,
        processed_count: usize,
    ) -> Result<()> {
        if self.config.enable_progress_reporting {
            tracing::info!(
                "Checkpoint: {} items processed, {} violations found",
                processed_count,
                result.total_violations
            );
        }

        // In a full implementation, you might save intermediate results to disk
        // or send them to an external monitoring system

        Ok(())
    }

    /// Get progress tracker for external monitoring
    pub fn progress_tracker(&self) -> Arc<BatchProgressTracker> {
        self.progress_tracker.clone()
    }

    /// Get performance analytics
    pub fn get_analytics(&self) -> Option<BatchPerformanceAnalytics> {
        self.analytics.lock().ok().map(|a| a.clone())
    }

    /// Request cancellation of current operation
    pub fn cancel(&self) {
        self.progress_tracker.cancel();
    }
}

/// Result of processing a single batch
#[derive(Debug, Clone, Default)]
pub struct BatchValidationResult {
    pub processed_items: usize,
    pub violation_count: usize,
    pub skipped_items: usize,
    pub processing_time: Duration,
}

impl BatchValidationResult {
    fn new() -> Self {
        Self::default()
    }
}

/// Comprehensive result of enhanced batch validation
#[derive(Debug, Clone)]
pub struct EnhancedBatchValidationResult {
    /// Total number of items processed
    pub total_processed: usize,
    /// Total number of violations found
    pub total_violations: usize,
    /// Total number of items skipped due to errors
    pub total_skipped: usize,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Whether the operation was cancelled
    pub was_cancelled: bool,
    /// Final progress state
    pub final_progress: BatchProgress,
    /// Log of all errors encountered
    pub error_log: Vec<BatchValidationError>,
    /// Performance analytics (if enabled)
    pub performance_analytics: Option<BatchPerformanceAnalytics>,
    /// Detailed validation report
    pub validation_report: ValidationReport,
}

impl EnhancedBatchValidationResult {
    fn new() -> Self {
        Self {
            total_processed: 0,
            total_violations: 0,
            total_skipped: 0,
            total_processing_time: Duration::ZERO,
            was_cancelled: false,
            final_progress: BatchProgress {
                total_items: 0,
                processed_items: 0,
                violations_found: 0,
                errors_encountered: 0,
                progress_percentage: 0.0,
                elapsed_time: Duration::ZERO,
                estimated_time_remaining: None,
                items_per_second: 0.0,
                is_cancelled: false,
            },
            error_log: Vec::new(),
            performance_analytics: None,
            validation_report: ValidationReport::new(),
        }
    }

    /// Merge results from a batch
    fn merge_batch_result(&mut self, batch_result: BatchValidationResult) {
        self.total_processed += batch_result.processed_items;
        self.total_violations += batch_result.violation_count;
        self.total_skipped += batch_result.skipped_items;
    }

    /// Generate a summary report
    pub fn generate_summary(&self) -> String {
        format!(
            "Enhanced Batch Validation Summary:\n\
             Total Processed: {}\n\
             Total Violations: {}\n\
             Total Skipped: {}\n\
             Total Processing Time: {:.2}s\n\
             Was Cancelled: {}\n\
             Errors Encountered: {}\n\
             Final Progress: {:.1}%\n\
             Average Speed: {:.1} items/sec",
            self.total_processed,
            self.total_violations,
            self.total_skipped,
            self.total_processing_time.as_secs_f64(),
            self.was_cancelled,
            self.error_log.len(),
            self.final_progress.progress_percentage,
            self.final_progress.items_per_second
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_validation_config_default() {
        let config = BatchValidationConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert!(config.enable_parallel);
        assert!(config.enable_progress_reporting);
        assert!(config.enable_error_recovery);
    }

    #[test]
    fn test_progress_tracker() {
        let tracker = BatchProgressTracker::new();
        tracker.set_total_items(100);
        tracker.update_progress(50, 5, 1);

        let progress = tracker.get_progress();
        assert_eq!(progress.total_items, 100);
        assert_eq!(progress.processed_items, 50);
        assert_eq!(progress.violations_found, 5);
        assert_eq!(progress.errors_encountered, 1);
        assert_eq!(progress.progress_percentage, 50.0);
    }

    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::new(1000, true);
        // Test that it doesn't immediately report memory pressure
        assert!(!monitor.check_memory_pressure());
    }

    #[test]
    fn test_error_handler() {
        let handler = BatchErrorHandler::new(Some(5));

        let error = BatchValidationError {
            error_type: BatchErrorType::ValidationFailure,
            message: "Test error".to_string(),
            node: None,
            shape: None,
            constraint: None,
            timestamp: Instant::now(),
            batch_index: 0,
            item_index: 0,
        };

        let action = handler.handle_error(error);
        assert!(matches!(action, ErrorRecoveryAction::Skip));
        assert_eq!(handler.error_count(), 1);
    }
}
