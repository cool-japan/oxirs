//! Async validation engine for I/O-bound and concurrent validation operations
//!
//! This module provides comprehensive async/await support for SHACL validation,
//! enabling efficient concurrent validation, streaming operations, and integration
//! with async I/O systems.

#[cfg(feature = "async")]
use std::collections::HashMap;
#[cfg(feature = "async")]
use std::future::Future;
#[cfg(feature = "async")]
use std::pin::Pin;
#[cfg(feature = "async")]
use std::sync::Arc;
#[cfg(feature = "async")]
use std::time::Duration;

#[cfg(feature = "async")]
use indexmap::IndexMap;
#[cfg(feature = "async")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "async")]
use tokio::sync::{mpsc, RwLock, Semaphore};
#[cfg(feature = "async")]
use tokio::time::{sleep, timeout};

#[cfg(feature = "async")]
use oxirs_core::{model::Term, Store};

#[cfg(feature = "async")]
use crate::{
    builders::{EnhancedValidatorBuilder, ValidationConfigBuilder},
    validation::{ValidationEngine, ValidationStats},
    ConstraintComponentId, Result, ShaclError, Shape, ShapeId, ValidationConfig, ValidationReport,
    ValidationViolation,
};

/// Async validation engine for concurrent and I/O-bound operations
#[derive(Debug)]
pub struct AsyncValidationEngine {
    /// Shared validation engine
    engine: Arc<RwLock<ValidationEngine<'static>>>,
    
    /// Async execution configuration
    async_config: AsyncValidationConfig,
    
    /// Semaphore for controlling concurrent operations
    concurrency_limiter: Arc<Semaphore>,
    
    /// Channel for streaming validation results
    result_sender: Option<mpsc::UnboundedSender<ValidationEvent>>,
}

/// Configuration for async validation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncValidationConfig {
    /// Maximum number of concurrent validation tasks
    pub max_concurrent_tasks: usize,
    
    /// Timeout for individual validation operations
    pub validation_timeout: Duration,
    
    /// Batch size for streaming validation
    pub stream_batch_size: usize,
    
    /// Enable progress reporting
    pub enable_progress_reporting: bool,
    
    /// Buffer size for async channels
    pub channel_buffer_size: usize,
    
    /// Enable validation result streaming
    pub enable_streaming: bool,
    
    /// Retry configuration for failed operations
    pub retry_config: AsyncRetryConfig,
}

impl Default for AsyncValidationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: num_cpus::get(),
            validation_timeout: Duration::from_secs(30),
            stream_batch_size: 100,
            enable_progress_reporting: true,
            channel_buffer_size: 1000,
            enable_streaming: false,
            retry_config: AsyncRetryConfig::default(),
        }
    }
}

/// Retry configuration for async operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncRetryConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    
    /// Base delay between retries
    pub base_delay: Duration,
    
    /// Maximum delay between retries
    pub max_delay: Duration,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for AsyncRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            backoff_multiplier: 2.0,
        }
    }
}

/// Events emitted during async validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationEvent {
    /// Validation started
    Started {
        total_shapes: usize,
        estimated_duration: Option<Duration>,
    },
    
    /// Progress update
    Progress {
        completed_shapes: usize,
        total_shapes: usize,
        current_shape: Option<ShapeId>,
        elapsed: Duration,
    },
    
    /// Violation found
    Violation {
        violation: ValidationViolation,
        shape_id: ShapeId,
    },
    
    /// Shape validation completed
    ShapeCompleted {
        shape_id: ShapeId,
        violations: usize,
        duration: Duration,
    },
    
    /// Validation completed
    Completed {
        report: ValidationReport,
        total_duration: Duration,
        statistics: ValidationStats,
    },
    
    /// Error occurred
    Error {
        error: String,
        shape_id: Option<ShapeId>,
        retry_attempt: Option<usize>,
    },
    
    /// Batch processed (for streaming)
    BatchProcessed {
        batch_size: usize,
        violations: usize,
        elapsed: Duration,
    },
}

/// Result of async validation operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncValidationResult {
    /// Main validation report
    pub report: ValidationReport,
    
    /// Detailed execution statistics
    pub stats: AsyncValidationStats,
    
    /// Events that occurred during validation
    pub events: Vec<ValidationEvent>,
}

/// Statistics for async validation operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncValidationStats {
    /// Total execution time
    pub total_duration: Duration,
    
    /// Time spent on actual validation
    pub validation_duration: Duration,
    
    /// Time spent on I/O operations
    pub io_duration: Duration,
    
    /// Number of concurrent tasks executed
    pub concurrent_tasks: usize,
    
    /// Number of retry attempts
    pub retry_attempts: usize,
    
    /// Peak memory usage (if available)
    pub peak_memory_usage: Option<usize>,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Average task duration
    pub avg_task_duration: Duration,
}

impl Default for AsyncValidationStats {
    fn default() -> Self {
        Self {
            total_duration: Duration::ZERO,
            validation_duration: Duration::ZERO,
            io_duration: Duration::ZERO,
            concurrent_tasks: 0,
            retry_attempts: 0,
            peak_memory_usage: None,
            cache_hit_rate: 0.0,
            avg_task_duration: Duration::ZERO,
        }
    }
}

/// Builder for async validation engine
#[derive(Debug)]
pub struct AsyncValidationEngineBuilder {
    shapes: Option<Arc<IndexMap<ShapeId, Shape>>>,
    validation_config: ValidationConfig,
    async_config: AsyncValidationConfig,
}

impl AsyncValidationEngineBuilder {
    /// Create a new async validation engine builder
    pub fn new() -> Self {
        Self {
            shapes: None,
            validation_config: ValidationConfig::default(),
            async_config: AsyncValidationConfig::default(),
        }
    }
    
    /// Set shapes for validation
    pub fn shapes(mut self, shapes: IndexMap<ShapeId, Shape>) -> Self {
        self.shapes = Some(Arc::new(shapes));
        self
    }
    
    /// Configure validation settings
    pub fn validation_config(mut self, config: ValidationConfig) -> Self {
        self.validation_config = config;
        self
    }
    
    /// Configure async settings
    pub fn async_config(mut self, config: AsyncValidationConfig) -> Self {
        self.async_config = config;
        self
    }
    
    /// Configure validation using fluent builder
    pub fn configure_validation<F>(mut self, configurator: F) -> Self
    where
        F: FnOnce(ValidationConfigBuilder) -> ValidationConfigBuilder,
    {
        let builder = ValidationConfigBuilder::new();
        self.validation_config = configurator(builder).build();
        self
    }
    
    /// Set maximum concurrent tasks
    pub fn max_concurrent_tasks(mut self, max_tasks: usize) -> Self {
        self.async_config.max_concurrent_tasks = max_tasks;
        self
    }
    
    /// Set validation timeout
    pub fn validation_timeout(mut self, timeout: Duration) -> Self {
        self.async_config.validation_timeout = timeout;
        self
    }
    
    /// Enable streaming validation
    pub fn enable_streaming(mut self, enable: bool) -> Self {
        self.async_config.enable_streaming = enable;
        self
    }
    
    /// Set stream batch size
    pub fn stream_batch_size(mut self, batch_size: usize) -> Self {
        self.async_config.stream_batch_size = batch_size;
        self
    }
    
    /// Build the async validation engine
    pub async fn build(self) -> Result<AsyncValidationEngine> {
        let shapes = self.shapes.ok_or_else(|| {
            ShaclError::ValidationEngine("No shapes provided for validation".to_string())
        })?;
        
        // Create sync validation engine with owned shapes
        let static_shapes: &'static IndexMap<ShapeId, Shape> = Box::leak(Box::new((*shapes).clone()));
        let engine = ValidationEngine::new(static_shapes, self.validation_config);
        
        let concurrency_limiter = Arc::new(Semaphore::new(self.async_config.max_concurrent_tasks));
        
        Ok(AsyncValidationEngine {
            engine: Arc::new(RwLock::new(engine)),
            async_config: self.async_config,
            concurrency_limiter,
            result_sender: None,
        })
    }
}

impl Default for AsyncValidationEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncValidationEngine {
    /// Create a new async validation engine builder
    pub fn builder() -> AsyncValidationEngineBuilder {
        AsyncValidationEngineBuilder::new()
    }
    
    /// Validate a store asynchronously
    pub async fn validate_store(&self, store: &Store) -> Result<AsyncValidationResult> {
        let start_time = tokio::time::Instant::now();
        let mut events = Vec::new();
        let mut stats = AsyncValidationStats::default();
        
        // Get shapes for validation
        let engine = self.engine.read().await;
        let shape_count = engine.shapes.len();
        drop(engine);
        
        events.push(ValidationEvent::Started {
            total_shapes: shape_count,
            estimated_duration: None,
        });
        
        // Validate with timeout
        let result = timeout(
            self.async_config.validation_timeout,
            self.validate_store_internal(store, &mut events, &mut stats)
        ).await;
        
        let report = match result {
            Ok(Ok(report)) => report,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(ShaclError::AsyncOperation(
                    "Validation timed out".to_string()
                ));
            }
        };
        
        stats.total_duration = start_time.elapsed();
        
        events.push(ValidationEvent::Completed {
            report: report.clone(),
            total_duration: stats.total_duration,
            statistics: engine.get_statistics().clone(),
        });
        
        Ok(AsyncValidationResult {
            report,
            stats,
            events,
        })
    }
    
    /// Validate specific shapes asynchronously
    pub async fn validate_shapes(
        &self,
        store: &Store,
        shape_ids: &[ShapeId],
    ) -> Result<AsyncValidationResult> {
        let start_time = tokio::time::Instant::now();
        let mut events = Vec::new();
        let mut stats = AsyncValidationStats::default();
        
        events.push(ValidationEvent::Started {
            total_shapes: shape_ids.len(),
            estimated_duration: None,
        });
        
        let mut final_report = ValidationReport::new();
        let mut completed_shapes = 0;
        
        // Process shapes concurrently with semaphore limiting
        let tasks: Vec<_> = shape_ids.iter().map(|shape_id| {
            let engine = Arc::clone(&self.engine);
            let shape_id = shape_id.clone();
            let semaphore = Arc::clone(&self.concurrency_limiter);
            
            async move {
                let _permit = semaphore.acquire().await.unwrap();
                let shape_start = tokio::time::Instant::now();
                
                let engine_guard = engine.read().await;
                let shape = engine_guard.shapes.get(&shape_id);
                if let Some(shape) = shape {
                    drop(engine_guard);
                    let mut engine_guard = engine.write().await;
                    let result = engine_guard.validate_shape(store, shape, None);
                    drop(engine_guard);
                    
                    match result {
                        Ok(report) => Ok((shape_id, report, shape_start.elapsed())),
                        Err(e) => Err((shape_id, e)),
                    }
                } else {
                    Err((shape_id, ShaclError::ValidationEngine(
                        format!("Shape not found: {}", shape_id)
                    )))
                }
            }
        }).collect();
        
        // Execute tasks concurrently
        let results = futures::future::join_all(tasks).await;
        
        for result in results {
            match result {
                Ok((shape_id, report, duration)) => {
                    completed_shapes += 1;
                    let violation_count = report.violation_count();
                    final_report.merge_result(report);
                    
                    events.push(ValidationEvent::ShapeCompleted {
                        shape_id: shape_id.clone(),
                        violations: violation_count,
                        duration,
                    });
                    
                    events.push(ValidationEvent::Progress {
                        completed_shapes,
                        total_shapes: shape_ids.len(),
                        current_shape: Some(shape_id),
                        elapsed: start_time.elapsed(),
                    });
                    
                    stats.concurrent_tasks += 1;
                }
                Err((shape_id, error)) => {
                    events.push(ValidationEvent::Error {
                        error: error.to_string(),
                        shape_id: Some(shape_id),
                        retry_attempt: None,
                    });
                }
            }
        }
        
        stats.total_duration = start_time.elapsed();
        stats.validation_duration = stats.total_duration; // Approximation
        stats.avg_task_duration = if stats.concurrent_tasks > 0 {
            stats.total_duration / stats.concurrent_tasks as u32
        } else {
            Duration::ZERO
        };
        
        let engine = self.engine.read().await;
        stats.cache_hit_rate = engine.get_cache_hit_rate();
        
        events.push(ValidationEvent::Completed {
            report: final_report.clone(),
            total_duration: stats.total_duration,
            statistics: engine.get_statistics().clone(),
        });
        
        Ok(AsyncValidationResult {
            report: final_report,
            stats,
            events,
        })
    }
    
    /// Validate a store with streaming results
    pub async fn validate_store_streaming(
        &self,
        store: &Store,
    ) -> Result<(mpsc::UnboundedReceiver<ValidationEvent>, impl Future<Output = Result<AsyncValidationResult>>)> {
        let (tx, rx) = mpsc::unbounded_channel();
        
        let engine = Arc::clone(&self.engine);
        let async_config = self.async_config.clone();
        let semaphore = Arc::clone(&self.concurrency_limiter);
        
        let validation_future = async move {
            let start_time = tokio::time::Instant::now();
            let mut stats = AsyncValidationStats::default();
            let mut events = Vec::new();
            
            let engine_guard = engine.read().await;
            let shape_count = engine_guard.shapes.len();
            let shapes: Vec<_> = engine_guard.shapes.iter().collect();
            drop(engine_guard);
            
            let _ = tx.send(ValidationEvent::Started {
                total_shapes: shape_count,
                estimated_duration: None,
            });
            
            let mut final_report = ValidationReport::new();
            let mut completed_shapes = 0;
            
            // Process shapes in batches for streaming
            for batch in shapes.chunks(async_config.stream_batch_size) {
                let batch_start = tokio::time::Instant::now();
                let tasks: Vec<_> = batch.iter().map(|(shape_id, shape)| {
                    let engine = Arc::clone(&engine);
                    let shape_id = (*shape_id).clone();
                    let shape = (*shape).clone();
                    let semaphore = Arc::clone(&semaphore);
                    let tx = tx.clone();
                    
                    async move {
                        let _permit = semaphore.acquire().await.unwrap();
                        let shape_start = tokio::time::Instant::now();
                        
                        let mut engine_guard = engine.write().await;
                        let result = engine_guard.validate_shape(store, &shape, None);
                        drop(engine_guard);
                        
                        match result {
                            Ok(report) => {
                                let violation_count = report.violation_count();
                                
                                // Send individual violations
                                for violation in &report.violations {
                                    let _ = tx.send(ValidationEvent::Violation {
                                        violation: violation.clone(),
                                        shape_id: shape_id.clone(),
                                    });
                                }
                                
                                let _ = tx.send(ValidationEvent::ShapeCompleted {
                                    shape_id: shape_id.clone(),
                                    violations: violation_count,
                                    duration: shape_start.elapsed(),
                                });
                                
                                Ok((shape_id, report))
                            }
                            Err(e) => {
                                let _ = tx.send(ValidationEvent::Error {
                                    error: e.to_string(),
                                    shape_id: Some(shape_id.clone()),
                                    retry_attempt: None,
                                });
                                Err((shape_id, e))
                            }
                        }
                    }
                }).collect();
                
                let batch_results = futures::future::join_all(tasks).await;
                let mut batch_violations = 0;
                
                for result in batch_results {
                    match result {
                        Ok((shape_id, report)) => {
                            completed_shapes += 1;
                            batch_violations += report.violation_count();
                            final_report.merge_result(report);
                            
                            let _ = tx.send(ValidationEvent::Progress {
                                completed_shapes,
                                total_shapes: shape_count,
                                current_shape: Some(shape_id),
                                elapsed: start_time.elapsed(),
                            });
                        }
                        Err(_) => {
                            // Error already sent in task
                        }
                    }
                }
                
                let _ = tx.send(ValidationEvent::BatchProcessed {
                    batch_size: batch.len(),
                    violations: batch_violations,
                    elapsed: batch_start.elapsed(),
                });
                
                stats.concurrent_tasks += batch.len();
            }
            
            stats.total_duration = start_time.elapsed();
            stats.validation_duration = stats.total_duration;
            
            let engine_guard = engine.read().await;
            stats.cache_hit_rate = engine_guard.get_cache_hit_rate();
            
            let _ = tx.send(ValidationEvent::Completed {
                report: final_report.clone(),
                total_duration: stats.total_duration,
                statistics: engine_guard.get_statistics().clone(),
            });
            
            Ok(AsyncValidationResult {
                report: final_report,
                stats,
                events,
            })
        };
        
        Ok((rx, validation_future))
    }
    
    /// Internal validation implementation
    async fn validate_store_internal(
        &self,
        store: &Store,
        events: &mut Vec<ValidationEvent>,
        stats: &mut AsyncValidationStats,
    ) -> Result<ValidationReport> {
        let mut engine = self.engine.write().await;
        let validation_start = tokio::time::Instant::now();
        
        let result = engine.validate_store(store);
        
        stats.validation_duration = validation_start.elapsed();
        stats.cache_hit_rate = engine.get_cache_hit_rate();
        
        result
    }
    
    /// Validate nodes with retry logic
    pub async fn validate_nodes_with_retry(
        &self,
        store: &Store,
        shape_id: &ShapeId,
        nodes: &[Term],
    ) -> Result<ValidationReport> {
        let mut attempts = 0;
        let mut delay = self.async_config.retry_config.base_delay;
        
        loop {
            let result = self.validate_nodes_async(store, shape_id, nodes).await;
            
            match result {
                Ok(report) => return Ok(report),
                Err(e) if attempts < self.async_config.retry_config.max_retries => {
                    attempts += 1;
                    
                    // Wait before retry
                    sleep(delay).await;
                    
                    // Exponential backoff
                    delay = std::cmp::min(
                        Duration::from_millis(
                            (delay.as_millis() as f64 * self.async_config.retry_config.backoff_multiplier) as u64
                        ),
                        self.async_config.retry_config.max_delay,
                    );
                }
                Err(e) => return Err(e),
            }
        }
    }
    
    /// Validate specific nodes asynchronously
    async fn validate_nodes_async(
        &self,
        store: &Store,
        shape_id: &ShapeId,
        nodes: &[Term],
    ) -> Result<ValidationReport> {
        let mut engine = self.engine.write().await;
        
        // Get the shape
        let shape = engine.shapes.get(shape_id)
            .ok_or_else(|| ShaclError::ValidationEngine(
                format!("Shape not found: {}", shape_id)
            ))?
            .clone();
        
        engine.validate_nodes(store, &shape, nodes)
    }
    
    /// Get current async configuration
    pub fn async_config(&self) -> &AsyncValidationConfig {
        &self.async_config
    }
    
    /// Update async configuration
    pub async fn update_async_config(&mut self, config: AsyncValidationConfig) {
        self.async_config = config;
        
        // Update semaphore if max concurrent tasks changed
        self.concurrency_limiter = Arc::new(Semaphore::new(self.async_config.max_concurrent_tasks));
    }
    
    /// Get validation statistics
    pub async fn get_statistics(&self) -> ValidationStats {
        let engine = self.engine.read().await;
        engine.get_statistics().clone()
    }
    
    /// Clear validation caches
    pub async fn clear_caches(&self) {
        let mut engine = self.engine.write().await;
        engine.clear_caches();
    }
}

/// Async validation utilities
pub mod utils {
    use super::*;
    
    /// Create an async validation engine from enhanced validator builder
    pub async fn from_enhanced_builder(
        builder: EnhancedValidatorBuilder,
        async_config: AsyncValidationConfig,
    ) -> Result<AsyncValidationEngine> {
        let validator = builder.build()?;
        
        // Extract shapes from validator (this would need to be implemented in the validator)
        // For now, create empty shapes
        let shapes = IndexMap::new();
        
        AsyncValidationEngine::builder()
            .shapes(shapes)
            .async_config(async_config)
            .build()
            .await
    }
    
    /// Validate multiple stores concurrently
    pub async fn validate_stores_concurrent(
        engines: Vec<&AsyncValidationEngine>,
        stores: Vec<&Store>,
    ) -> Result<Vec<AsyncValidationResult>> {
        if engines.len() != stores.len() {
            return Err(ShaclError::AsyncOperation(
                "Number of engines must match number of stores".to_string()
            ));
        }
        
        let tasks: Vec<_> = engines.into_iter().zip(stores).map(|(engine, store)| {
            engine.validate_store(store)
        }).collect();
        
        let results = futures::future::try_join_all(tasks).await?;
        Ok(results)
    }
    
    /// Monitor validation progress
    pub async fn monitor_validation(
        mut event_receiver: mpsc::UnboundedReceiver<ValidationEvent>,
        progress_callback: impl Fn(ValidationEvent) + Send + Sync + 'static,
    ) {
        while let Some(event) = event_receiver.recv().await {
            progress_callback(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::Store;
    
    #[tokio::test]
    async fn test_async_validation_engine_builder() {
        let shapes = IndexMap::new();
        let config = AsyncValidationConfig::default();
        
        let engine = AsyncValidationEngine::builder()
            .shapes(shapes)
            .async_config(config)
            .max_concurrent_tasks(4)
            .validation_timeout(Duration::from_secs(10))
            .build()
            .await;
        
        assert!(engine.is_ok());
        let engine = engine.unwrap();
        assert_eq!(engine.async_config.max_concurrent_tasks, 4);
        assert_eq!(engine.async_config.validation_timeout, Duration::from_secs(10));
    }
    
    #[tokio::test]
    async fn test_async_validation_config() {
        let config = AsyncValidationConfig::default();
        assert!(config.max_concurrent_tasks > 0);
        assert!(config.validation_timeout > Duration::ZERO);
        assert!(config.stream_batch_size > 0);
    }
    
    #[tokio::test]
    async fn test_retry_config() {
        let config = AsyncRetryConfig::default();
        assert!(config.max_retries > 0);
        assert!(config.base_delay > Duration::ZERO);
        assert!(config.max_delay > config.base_delay);
        assert!(config.backoff_multiplier > 1.0);
    }
}