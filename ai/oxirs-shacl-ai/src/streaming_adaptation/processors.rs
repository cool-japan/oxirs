//! Stream processors for different data types

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, RwLock};

use oxirs_core::model::Triple;
use oxirs_shacl::ValidationReport;

use crate::neural_patterns::NeuralPattern;
use crate::self_adaptive_ai::PerformanceMetrics;
use crate::{Result, ShaclAiError};

use super::core::StreamData;

/// Trait for stream processors
#[async_trait::async_trait]
pub trait StreamProcessor: Send + Sync + std::fmt::Debug {
    async fn initialize(&self) -> Result<()>;
    async fn process(&self, data: Box<dyn StreamData>) -> Result<()>;
    async fn shutdown(&self) -> Result<()>;
}

/// RDF stream processor for handling triple streams
#[derive(Debug)]
pub struct RdfStreamProcessor {
    processed_count: Arc<Mutex<u64>>,
    triple_buffer: Arc<RwLock<Vec<Triple>>>,
}

impl RdfStreamProcessor {
    /// Create new RDF stream processor
    pub fn new() -> Self {
        Self {
            processed_count: Arc::new(Mutex::new(0)),
            triple_buffer: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get number of processed triples
    pub async fn get_processed_count(&self) -> u64 {
        *self.processed_count.lock().await
    }

    /// Get current buffer size
    pub async fn get_buffer_size(&self) -> usize {
        self.triple_buffer.read().await.len()
    }

    /// Process RDF triple
    async fn process_triple(&self, triple: &Triple) -> Result<()> {
        // Add to buffer
        {
            let mut buffer = self.triple_buffer.write().await;
            buffer.push(triple.clone());

            // Keep buffer size manageable
            if buffer.len() > 1000 {
                buffer.drain(0..500);
            }
        }

        // Update count
        {
            let mut count = self.processed_count.lock().await;
            *count += 1;
        }

        tracing::debug!("Processed RDF triple: {:?}", triple);
        Ok(())
    }
}

#[async_trait::async_trait]
impl StreamProcessor for RdfStreamProcessor {
    async fn initialize(&self) -> Result<()> {
        tracing::info!("Initialized RDF stream processor");
        Ok(())
    }

    async fn process(&self, data: Box<dyn StreamData>) -> Result<()> {
        // Extract and own the triple completely before any async operations
        let any_data = data as Box<dyn std::any::Any>;
        let triple_owned = match any_data.downcast::<Triple>() {
            Ok(triple) => *triple,
            Err(_) => {
                return Err(ShaclAiError::StreamingAdaptation(
                    "Invalid data type for RDF processor".to_string(),
                ));
            }
        };

        // Now we can safely await without holding Any references
        self.process_triple(&triple_owned).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutdown RDF stream processor");
        // Clear buffer
        self.triple_buffer.write().await.clear();
        Ok(())
    }
}

/// Validation stream processor for handling validation reports
#[derive(Debug)]
pub struct ValidationStreamProcessor {
    processed_reports: Arc<Mutex<u64>>,
    validation_cache: Arc<RwLock<HashMap<String, ValidationReport>>>,
}

impl ValidationStreamProcessor {
    /// Create new validation stream processor
    pub fn new() -> Self {
        Self {
            processed_reports: Arc::new(Mutex::new(0)),
            validation_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get number of processed reports
    pub async fn get_processed_count(&self) -> u64 {
        *self.processed_reports.lock().await
    }

    /// Process validation report
    async fn process_validation_report(&self, report: &ValidationReport) -> Result<()> {
        // Cache the report
        {
            let mut cache = self.validation_cache.write().await;
            let report_id = format!("report_{}", chrono::Utc::now().timestamp_millis());
            cache.insert(report_id, report.clone());

            // Keep cache size manageable
            if cache.len() > 500 {
                let keys_to_remove: Vec<_> = cache.keys().take(250).cloned().collect();
                for key in keys_to_remove {
                    cache.remove(&key);
                }
            }
        }

        // Update count
        {
            let mut count = self.processed_reports.lock().await;
            *count += 1;
        }

        tracing::debug!(
            "Processed validation report with {} violations",
            report.violations.len()
        );
        Ok(())
    }
}

#[async_trait::async_trait]
impl StreamProcessor for ValidationStreamProcessor {
    async fn initialize(&self) -> Result<()> {
        tracing::info!("Initialized validation stream processor");
        Ok(())
    }

    async fn process(&self, data: Box<dyn StreamData>) -> Result<()> {
        // Extract and own the report completely before any async operations
        let report_owned = match (data as Box<dyn std::any::Any>).downcast::<ValidationReport>() {
            Ok(report) => *report,
            Err(_) => {
                return Err(ShaclAiError::StreamingAdaptation(
                    "Invalid data type for validation processor".to_string(),
                ));
            }
        };

        // Now we can safely await without holding Any references
        self.process_validation_report(&report_owned).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutdown validation stream processor");
        // Clear cache
        self.validation_cache.write().await.clear();
        Ok(())
    }
}

/// Metrics stream processor for handling performance metrics
#[derive(Debug)]
pub struct MetricsStreamProcessor {
    processed_metrics: Arc<Mutex<u64>>,
    metrics_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
}

impl MetricsStreamProcessor {
    /// Create new metrics stream processor
    pub fn new() -> Self {
        Self {
            processed_metrics: Arc::new(Mutex::new(0)),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get number of processed metrics
    pub async fn get_processed_count(&self) -> u64 {
        *self.processed_metrics.lock().await
    }

    /// Get latest metrics
    pub async fn get_latest_metrics(&self) -> Option<PerformanceMetrics> {
        self.metrics_history.read().await.last().cloned()
    }

    /// Process performance metrics
    async fn process_metrics(&self, metrics: &PerformanceMetrics) -> Result<()> {
        // Add to history
        {
            let mut history = self.metrics_history.write().await;
            history.push(metrics.clone());

            // Keep history size manageable (last 1000 entries)
            if history.len() > 1000 {
                history.drain(0..500);
            }
        }

        // Update count
        {
            let mut count = self.processed_metrics.lock().await;
            *count += 1;
        }

        tracing::debug!(
            "Processed performance metrics: accuracy={:.3}, throughput={:.1}",
            metrics.accuracy,
            metrics.throughput
        );
        Ok(())
    }
}

#[async_trait::async_trait]
impl StreamProcessor for MetricsStreamProcessor {
    async fn initialize(&self) -> Result<()> {
        tracing::info!("Initialized metrics stream processor");
        Ok(())
    }

    async fn process(&self, data: Box<dyn StreamData>) -> Result<()> {
        // Extract and own the metrics completely before any async operations
        let metrics_owned = match (data as Box<dyn std::any::Any>).downcast::<PerformanceMetrics>()
        {
            Ok(metrics) => *metrics,
            Err(_) => {
                return Err(ShaclAiError::StreamingAdaptation(
                    "Invalid data type for metrics processor".to_string(),
                ));
            }
        };

        // Now we can safely await without holding Any references
        self.process_metrics(&metrics_owned).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutdown metrics stream processor");
        // Clear history
        self.metrics_history.write().await.clear();
        Ok(())
    }
}

/// Pattern stream processor for handling neural patterns
#[derive(Debug)]
pub struct PatternStreamProcessor {
    processed_patterns: Arc<Mutex<u64>>,
    pattern_library: Arc<RwLock<HashMap<String, NeuralPattern>>>,
}

impl PatternStreamProcessor {
    /// Create new pattern stream processor
    pub fn new() -> Self {
        Self {
            processed_patterns: Arc::new(Mutex::new(0)),
            pattern_library: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get number of processed patterns
    pub async fn get_processed_count(&self) -> u64 {
        *self.processed_patterns.lock().await
    }

    /// Get pattern library size
    pub async fn get_library_size(&self) -> usize {
        self.pattern_library.read().await.len()
    }

    /// Process neural pattern
    async fn process_neural_pattern(&self, pattern: &NeuralPattern) -> Result<()> {
        // Add to library
        {
            let mut library = self.pattern_library.write().await;
            let pattern_id = format!("pattern_{}", chrono::Utc::now().timestamp_millis());
            library.insert(pattern_id, pattern.clone());

            // Keep library size manageable
            if library.len() > 200 {
                let keys_to_remove: Vec<_> = library.keys().take(100).cloned().collect();
                for key in keys_to_remove {
                    library.remove(&key);
                }
            }
        }

        // Update count
        {
            let mut count = self.processed_patterns.lock().await;
            *count += 1;
        }

        tracing::debug!(
            "Processed neural pattern with {} features",
            pattern.features.len()
        );
        Ok(())
    }
}

#[async_trait::async_trait]
impl StreamProcessor for PatternStreamProcessor {
    async fn initialize(&self) -> Result<()> {
        tracing::info!("Initialized pattern stream processor");
        Ok(())
    }

    async fn process(&self, data: Box<dyn StreamData>) -> Result<()> {
        // Extract and own the pattern completely before any async operations
        let pattern_owned = match (data as Box<dyn std::any::Any>).downcast::<NeuralPattern>() {
            Ok(pattern) => *pattern,
            Err(_) => {
                return Err(ShaclAiError::StreamingAdaptation(
                    "Invalid data type for pattern processor".to_string(),
                ));
            }
        };

        // Now we can safely await without holding Any references
        self.process_neural_pattern(&pattern_owned).await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutdown pattern stream processor");
        // Clear library
        self.pattern_library.write().await.clear();
        Ok(())
    }
}

impl Default for RdfStreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ValidationStreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MetricsStreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PatternStreamProcessor {
    fn default() -> Self {
        Self::new()
    }
}
