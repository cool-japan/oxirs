//! SHACL-AI assistant: core error types, main assistant struct, and builder.
//!
//! This module contains the primary entry point [`ShaclAiAssistant`] along with
//! [`ShaclAiAssistantBuilder`], the crate-level [`ShaclAiError`] enum, and the
//! `init` / `VERSION` public surface.

use std::sync::{Arc, Mutex};
use thiserror::Error;

use oxirs_core::{OxirsError, Store};

use oxirs_shacl::{ValidationConfig, ValidationReport};

use crate::{
    analytics::{AnalyticsEngine, ValidationInsights},
    config::{ShaclAiConfig, ShaclAiStatistics},
    data_types::{ModelTrainingResult, TrainingDataset, TrainingResult},
    error_handling::{ErrorHandlingConfig, IntelligentErrorHandler, SmartErrorAnalysis},
    optimization::{OptimizationEngine, OptimizedValidationStrategy},
    patterns::PatternAnalyzer,
    prediction::{ValidationPrediction, ValidationPredictor},
    quality::{QualityAssessor, QualityReport},
    ai_orchestrator::{AiOrchestrator, ComprehensiveLearningResult},
    learning::{LearningConfig, ShapeLearner},
    prediction::PredictionConfig,
    quality::QualityConfig,
};

/// Core error type for SHACL-AI operations
#[derive(Debug, Error)]
pub enum ShaclAiError {
    #[error("Shape learning error: {0}")]
    ShapeLearning(String),

    #[error("Quality assessment error: {0}")]
    QualityAssessment(String),

    #[error("Validation prediction error: {0}")]
    ValidationPrediction(String),

    #[error("Pattern recognition error: {0}")]
    PatternRecognition(String),

    #[error("Optimization error: {0}")]
    Optimization(String),

    #[error("Performance error: {0}")]
    Performance(String),

    #[error("Analytics error: {0}")]
    Analytics(String),

    #[error("Visualization error: {0}")]
    Visualization(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Model training error: {0}")]
    ModelTraining(String),

    #[error("Meta-learning error: {0}")]
    MetaLearning(String),

    #[error("Data processing error: {0}")]
    DataProcessing(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("Shape management error: {0}")]
    ShapeManagement(String),

    #[error("Predictive analytics error: {0}")]
    PredictiveAnalytics(String),

    #[error("Streaming adaptation error: {0}")]
    StreamingAdaptation(String),

    #[error("Performance analytics error: {0}")]
    PerformanceAnalytics(String),

    #[error("Version not found: {0}")]
    VersionNotFound(String),

    #[error("SHACL error: {0}")]
    Shacl(#[from] oxirs_shacl::ShaclError),

    #[error("OxiRS core error: {0}")]
    Core(#[from] OxirsError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Anyhow error: {0}")]
    Anyhow(#[from] anyhow::Error),

    #[error("Join error: {0}")]
    Join(#[from] tokio::task::JoinError),

    #[error("Model error: {0}")]
    Model(#[from] crate::ml::ModelError),

    #[error("Photonic computing error: {0}")]
    PhotonicComputing(String),

    #[error("Array shape error: {0}")]
    Shape(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Integration testing error: {0}")]
    Integration(String),

    #[error("Benchmark error: {0}")]
    Benchmark(String),
}

/// Result type alias for SHACL-AI operations
pub type Result<T> = std::result::Result<T, ShaclAiError>;

impl From<scirs2_core::ndarray_ext::ShapeError> for ShaclAiError {
    fn from(err: scirs2_core::ndarray_ext::ShapeError) -> Self {
        ShaclAiError::Shape(err.to_string())
    }
}

/// AI-powered SHACL assistant for comprehensive validation enhancement
#[derive(Debug)]
pub struct ShaclAiAssistant {
    /// Shape learning component
    shape_learner: Arc<Mutex<ShapeLearner>>,

    /// Quality assessment component
    quality_assessor: Arc<Mutex<QualityAssessor>>,

    /// Validation predictor component
    validation_predictor: Arc<Mutex<ValidationPredictor>>,

    /// Optimization engine
    optimization_engine: Arc<Mutex<OptimizationEngine>>,

    /// Pattern analyzer
    pattern_analyzer: Arc<Mutex<PatternAnalyzer>>,

    /// Analytics engine
    analytics_engine: Arc<Mutex<AnalyticsEngine>>,

    /// AI orchestrator for comprehensive learning
    ai_orchestrator: Arc<Mutex<AiOrchestrator>>,

    /// Intelligent error handler
    error_handler: Arc<Mutex<IntelligentErrorHandler>>,

    /// Configuration
    pub(crate) config: ShaclAiConfig,
}

impl ShaclAiAssistant {
    /// Create a new SHACL-AI assistant with default configuration
    pub fn new() -> Self {
        let config = ShaclAiConfig::default();
        Self::with_config(config)
    }

    /// Create a new SHACL-AI assistant with custom configuration
    pub fn with_config(config: ShaclAiConfig) -> Self {
        Self {
            shape_learner: Arc::new(Mutex::new(ShapeLearner::with_config(
                config.learning.clone(),
            ))),
            quality_assessor: Arc::new(Mutex::new(QualityAssessor::with_config(
                config.quality.clone(),
            ))),
            validation_predictor: Arc::new(Mutex::new(ValidationPredictor::with_config(
                config.prediction.clone(),
            ))),
            optimization_engine: Arc::new(Mutex::new(OptimizationEngine::with_config(
                config.optimization.clone(),
            ))),
            pattern_analyzer: Arc::new(Mutex::new(PatternAnalyzer::with_config(
                config.patterns.clone(),
            ))),
            analytics_engine: Arc::new(Mutex::new(AnalyticsEngine::with_config(
                config.analytics.clone(),
            ))),
            ai_orchestrator: Arc::new(Mutex::new(AiOrchestrator::new())),
            error_handler: Arc::new(Mutex::new(IntelligentErrorHandler::with_config(
                ErrorHandlingConfig::default(),
            ))),
            config,
        }
    }

    /// Learn shapes from RDF data with AI assistance
    pub fn learn_shapes(
        &mut self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<Vec<oxirs_shacl::Shape>> {
        tracing::info!("Starting AI-powered shape learning");

        // Analyze patterns in the data
        let patterns = self
            .pattern_analyzer
            .lock()
            .map_err(|e| {
                ShaclAiError::ShapeLearning(format!("Failed to lock pattern analyzer: {e}"))
            })?
            .analyze_graph_patterns(store, graph_name)?;
        tracing::debug!("Discovered {} patterns in graph", patterns.len());

        // Learn shapes based on patterns
        let learned_shapes = self
            .shape_learner
            .lock()
            .map_err(|e| ShaclAiError::ShapeLearning(format!("Failed to lock shape learner: {e}")))?
            .learn_shapes_from_patterns(store, &patterns, graph_name)?;
        tracing::info!("Learned {} shapes from data", learned_shapes.len());

        // Optimize learned shapes
        let optimized_shapes = self
            .optimization_engine
            .lock()
            .map_err(|e| {
                ShaclAiError::Optimization(format!("Failed to lock optimization engine: {e}"))
            })?
            .optimize_shapes(&learned_shapes, store)?;
        tracing::info!("Optimized shapes using AI recommendations");

        Ok(optimized_shapes)
    }

    /// Comprehensive AI-powered shape learning using orchestrator (Ultrathink Mode)
    pub fn learn_shapes_comprehensive(
        &mut self,
        store: &dyn Store,
        graph_name: Option<&str>,
    ) -> Result<ComprehensiveLearningResult> {
        tracing::info!("Starting comprehensive AI-powered shape learning (Ultrathink Mode)");

        self.ai_orchestrator
            .lock()
            .map_err(|e| {
                ShaclAiError::ShapeLearning(format!("Failed to lock AI orchestrator: {e}"))
            })?
            .comprehensive_learning(store, graph_name)
    }

    /// Extract high-quality shapes from comprehensive learning result
    pub fn extract_shapes_from_comprehensive_result(
        &self,
        result: &ComprehensiveLearningResult,
    ) -> Vec<oxirs_shacl::Shape> {
        result.shapes.to_vec()
    }

    /// Assess data quality with AI insights
    pub fn assess_quality(
        &self,
        store: &dyn Store,
        shapes: &[oxirs_shacl::Shape],
    ) -> Result<QualityReport> {
        tracing::info!("Starting AI-powered quality assessment");

        let report = self
            .quality_assessor
            .lock()
            .map_err(|e| {
                ShaclAiError::QualityAssessment(format!("Failed to lock quality assessor: {e}"))
            })?
            .assess_comprehensive_quality(store, shapes)?;

        // Add AI insights to the report
        let insights = self
            .analytics_engine
            .lock()
            .map_err(|e| ShaclAiError::Analytics(format!("Failed to lock analytics engine: {e}")))?
            .generate_quality_insights(store, shapes, &report)?;

        Ok(QualityReport {
            ai_insights: Some(insights),
            ..report
        })
    }

    /// Predict validation outcomes before execution
    pub fn predict_validation(
        &self,
        store: &dyn Store,
        shapes: &[oxirs_shacl::Shape],
        config: &ValidationConfig,
    ) -> Result<ValidationPrediction> {
        tracing::info!("Predicting validation outcomes using AI");

        self.validation_predictor
            .lock()
            .map_err(|e| {
                ShaclAiError::ValidationPrediction(format!(
                    "Failed to lock validation predictor: {e}"
                ))
            })?
            .predict_validation_outcome(store, shapes, config)
    }

    /// Optimize validation strategy using AI recommendations
    pub fn optimize_validation(
        &self,
        store: &dyn Store,
        shapes: &[oxirs_shacl::Shape],
    ) -> Result<OptimizedValidationStrategy> {
        tracing::info!("Optimizing validation strategy with AI");

        self.optimization_engine
            .lock()
            .map_err(|e| {
                ShaclAiError::Optimization(format!("Failed to lock optimization engine: {e}"))
            })?
            .optimize_validation_strategy(store, shapes)
    }

    /// Generate comprehensive analytics and insights
    pub fn generate_insights(
        &self,
        store: &dyn Store,
        shapes: &[oxirs_shacl::Shape],
        validation_history: &[ValidationReport],
    ) -> Result<ValidationInsights> {
        tracing::info!("Generating AI-powered validation insights");

        self.analytics_engine
            .lock()
            .map_err(|e| ShaclAiError::Analytics(format!("Failed to lock analytics engine: {e}")))?
            .generate_comprehensive_insights(store, shapes, validation_history)
    }

    /// Process validation errors with intelligent error handling and repair suggestions
    pub fn process_validation_errors(
        &self,
        validation_report: &ValidationReport,
        store: &dyn Store,
        shapes: &[oxirs_shacl::Shape],
    ) -> Result<SmartErrorAnalysis> {
        tracing::info!(
            "Processing validation errors with intelligent analysis and repair suggestions"
        );

        self.error_handler
            .lock()
            .map_err(|e| {
                ShaclAiError::ValidationPrediction(format!("Failed to lock error handler: {e}"))
            })?
            .process_validation_errors(validation_report, store, shapes)
    }

    /// Train models on validation data for improved predictions
    pub fn train_models(&mut self, training_data: &TrainingDataset) -> Result<TrainingResult> {
        tracing::info!("Training AI models on validation data");
        let training_start = std::time::Instant::now();

        let mut results = Vec::new();
        let mut all_successful = true;

        // Train shape learning model
        if self.config.learning.enable_training {
            let model_start = std::time::Instant::now();
            match self
                .shape_learner
                .lock()
                .map_err(|e| {
                    ShaclAiError::ModelTraining(format!("Failed to lock shape learner: {e}"))
                })?
                .train_model(&training_data.shape_data)
            {
                Ok(shape_result) => {
                    tracing::info!("Shape learning model trained successfully");
                    results.push(("shape_learning".to_string(), shape_result));
                }
                Err(e) => {
                    tracing::error!("Shape learning model training failed: {}", e);
                    all_successful = false;
                    results.push((
                        "shape_learning".to_string(),
                        ModelTrainingResult {
                            success: false,
                            accuracy: 0.0,
                            loss: f64::INFINITY,
                            epochs_trained: 0,
                            training_time: model_start.elapsed(),
                        },
                    ));
                }
            }
        }

        // Train quality assessment model
        if self.config.quality.enable_training {
            let model_start = std::time::Instant::now();
            match self
                .quality_assessor
                .lock()
                .map_err(|e| {
                    ShaclAiError::ModelTraining(format!("Failed to lock quality assessor: {e}"))
                })?
                .train_model(&training_data.quality_data)
            {
                Ok(quality_result) => {
                    tracing::info!("Quality assessment model trained successfully");
                    results.push(("quality_assessment".to_string(), quality_result));
                }
                Err(e) => {
                    tracing::error!("Quality assessment model training failed: {}", e);
                    all_successful = false;
                    results.push((
                        "quality_assessment".to_string(),
                        ModelTrainingResult {
                            success: false,
                            accuracy: 0.0,
                            loss: f64::INFINITY,
                            epochs_trained: 0,
                            training_time: model_start.elapsed(),
                        },
                    ));
                }
            }
        }

        // Train validation prediction model
        if self.config.prediction.enable_training {
            let model_start = std::time::Instant::now();
            match self
                .validation_predictor
                .lock()
                .map_err(|e| {
                    ShaclAiError::ModelTraining(format!("Failed to lock validation predictor: {e}"))
                })?
                .train_model(&training_data.prediction_data)
            {
                Ok(prediction_result) => {
                    tracing::info!("Validation prediction model trained successfully");
                    results.push(("validation_prediction".to_string(), prediction_result));
                }
                Err(e) => {
                    tracing::error!("Validation prediction model training failed: {}", e);
                    all_successful = false;
                    results.push((
                        "validation_prediction".to_string(),
                        ModelTrainingResult {
                            success: false,
                            accuracy: 0.0,
                            loss: f64::INFINITY,
                            epochs_trained: 0,
                            training_time: model_start.elapsed(),
                        },
                    ));
                }
            }
        }

        let total_training_time = training_start.elapsed();
        tracing::info!(
            "Model training completed in {:?} with {} successful models out of {}",
            total_training_time,
            results.iter().filter(|(_, r)| r.success).count(),
            results.len()
        );

        Ok(TrainingResult {
            model_results: results,
            overall_success: all_successful,
            training_time: total_training_time,
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &ShaclAiConfig {
        &self.config
    }

    /// Get comprehensive statistics about AI operations
    pub fn get_ai_statistics(&self) -> Result<ShaclAiStatistics> {
        Ok(ShaclAiStatistics {
            shapes_learned: self
                .shape_learner
                .lock()
                .map_err(|e| ShaclAiError::Analytics(format!("Failed to lock shape learner: {e}")))?
                .get_statistics()
                .total_shapes_learned,
            quality_assessments: self
                .quality_assessor
                .lock()
                .map_err(|e| {
                    ShaclAiError::Analytics(format!("Failed to lock quality assessor: {e}"))
                })?
                .get_statistics()
                .total_assessments,
            predictions_made: self
                .validation_predictor
                .lock()
                .map_err(|e| {
                    ShaclAiError::Analytics(format!("Failed to lock validation predictor: {e}"))
                })?
                .get_statistics()
                .total_predictions,
            optimizations_performed: self
                .optimization_engine
                .lock()
                .map_err(|e| {
                    ShaclAiError::Analytics(format!("Failed to lock optimization engine: {e}"))
                })?
                .get_statistics()
                .total_optimizations,
            patterns_analyzed: self
                .pattern_analyzer
                .lock()
                .map_err(|e| {
                    ShaclAiError::Analytics(format!("Failed to lock pattern analyzer: {e}"))
                })?
                .get_statistics()
                .total_analyses,
            insights_generated: self
                .analytics_engine
                .lock()
                .map_err(|e| {
                    ShaclAiError::Analytics(format!("Failed to lock analytics engine: {e}"))
                })?
                .get_statistics()
                .total_insights_generated,
        })
    }
}

impl Default for ShaclAiAssistant {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating SHACL-AI assistant with custom configuration
#[derive(Debug)]
pub struct ShaclAiAssistantBuilder {
    pub(crate) config: ShaclAiConfig,
}

impl ShaclAiAssistantBuilder {
    pub fn new() -> Self {
        Self {
            config: ShaclAiConfig::default(),
        }
    }

    pub fn with_learning_config(mut self, config: LearningConfig) -> Self {
        self.config.learning = config;
        self
    }

    pub fn with_quality_config(mut self, config: QualityConfig) -> Self {
        self.config.quality = config;
        self
    }

    pub fn with_prediction_config(mut self, config: PredictionConfig) -> Self {
        self.config.prediction = config;
        self
    }

    pub fn with_optimization_config(mut self, config: crate::optimization::OptimizationConfig) -> Self {
        self.config.optimization = config;
        self
    }

    pub fn enable_parallel_processing(mut self, enable: bool) -> Self {
        self.config.global.enable_parallel_processing = enable;
        self
    }

    pub fn max_memory_mb(mut self, max_memory: usize) -> Self {
        self.config.global.max_memory_mb = max_memory;
        self
    }

    pub fn enable_caching(mut self, enable: bool) -> Self {
        self.config.global.enable_caching = enable;
        self
    }

    pub fn build(self) -> ShaclAiAssistant {
        ShaclAiAssistant::with_config(self.config)
    }
}

impl Default for ShaclAiAssistantBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Version information for OxiRS SHACL-AI
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize OxiRS SHACL-AI with default configuration
pub fn init() -> Result<()> {
    tracing::info!("Initializing OxiRS SHACL-AI v{}", VERSION);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shacl_ai_assistant_creation() {
        let assistant = ShaclAiAssistant::new();
        let stats = assistant.get_ai_statistics().expect("should succeed");

        assert_eq!(stats.shapes_learned, 0);
        assert_eq!(stats.quality_assessments, 0);
        assert_eq!(stats.predictions_made, 0);
    }

    #[test]
    fn test_shacl_ai_config_default() {
        let config = ShaclAiConfig::default();

        assert!(config.global.enable_parallel_processing);
        assert_eq!(config.global.max_memory_mb, 1024);
        assert!(config.global.enable_caching);
        assert_eq!(config.global.cache_size_limit, 10000);
    }

    #[test]
    fn test_shacl_ai_builder() {
        let assistant = ShaclAiAssistantBuilder::new()
            .enable_parallel_processing(false)
            .max_memory_mb(512)
            .enable_caching(false)
            .build();

        assert!(!assistant.config.global.enable_parallel_processing);
        assert_eq!(assistant.config.global.max_memory_mb, 512);
        assert!(!assistant.config.global.enable_caching);
    }
}
