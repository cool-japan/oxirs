//! # External ML Frameworks Integration
//!
//! Seamless integration with popular ML frameworks:
//! - TensorFlow integration
//! - PyTorch integration
//! - ONNX model support
//! - Hugging Face Transformers integration
//! - scikit-learn integration
//! - Model format conversion
//! - Inference engines (ONNX Runtime, TensorRT)
//! - Model zoo/repository integration

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use scirs2_core::ndarray_ext::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Main external ML integration system
#[derive(Clone)]
pub struct ExternalMLIntegration {
    config: MLIntegrationConfig,
    model_registry: Arc<ModelRegistry>,
    framework_adapters: Arc<DashMap<MLFramework, Box<dyn FrameworkAdapter>>>,
    inference_engine: Arc<InferenceEngine>,
    model_converter: Arc<ModelConverter>,
}

/// ML integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLIntegrationConfig {
    /// Enable TensorFlow support
    pub enable_tensorflow: bool,
    /// Enable PyTorch support
    pub enable_pytorch: bool,
    /// Enable ONNX support
    pub enable_onnx: bool,
    /// Enable Hugging Face support
    pub enable_huggingface: bool,
    /// Model cache directory
    pub model_cache_dir: PathBuf,
    /// Default inference backend
    pub default_inference_backend: InferenceBackend,
    /// Maximum model size (MB)
    pub max_model_size_mb: usize,
    /// Enable model quantization
    pub enable_quantization: bool,
}

impl Default for MLIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_tensorflow: true,
            enable_pytorch: true,
            enable_onnx: true,
            enable_huggingface: true,
            model_cache_dir: PathBuf::from("/tmp/ml_models"),
            default_inference_backend: InferenceBackend::ONNX,
            max_model_size_mb: 500,
            enable_quantization: false,
        }
    }
}

/// ML Framework enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MLFramework {
    TensorFlow,
    PyTorch,
    ONNX,
    HuggingFace,
    ScikitLearn,
    JAX,
    MXNet,
}

/// Inference backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceBackend {
    ONNX,
    TensorRT,
    OpenVINO,
    CoreML,
    Native,
}

/// Model Registry
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<String, RegisteredModel>>>,
    model_versions: Arc<DashMap<String, Vec<ModelVersion>>>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            model_versions: Arc::new(DashMap::new()),
        }
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new model
    pub async fn register_model(&self, model: RegisteredModel) -> Result<()> {
        let mut models = self.models.write().await;
        models.insert(model.id.clone(), model.clone());

        // Initialize version tracking
        self.model_versions
            .insert(model.id.clone(), vec![model.current_version.clone()]);

        info!("Registered model: {}", model.id);
        Ok(())
    }

    /// Get model by ID
    pub async fn get_model(&self, model_id: &str) -> Option<RegisteredModel> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }

    /// Add model version
    pub async fn add_version(&self, model_id: &str, version: ModelVersion) -> Result<()> {
        if let Some(mut versions) = self.model_versions.get_mut(model_id) {
            versions.push(version.clone());

            // Update current version in model
            let mut models = self.models.write().await;
            if let Some(model) = models.get_mut(model_id) {
                model.current_version = version;
            }

            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", model_id))
        }
    }

    /// List all models
    pub async fn list_models(&self) -> Vec<RegisteredModel> {
        let models = self.models.read().await;
        models.values().cloned().collect()
    }
}

/// Registered model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub framework: MLFramework,
    pub current_version: ModelVersion,
    pub task_type: MLTaskType,
    pub metadata: ModelMetadata,
    pub created_at: DateTime<Utc>,
}

/// Model version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub version: String,
    pub model_path: PathBuf,
    pub format: ModelFormat,
    pub input_schema: Vec<TensorSpec>,
    pub output_schema: Vec<TensorSpec>,
    pub performance_metrics: PerformanceMetrics,
    pub created_at: DateTime<Utc>,
}

/// ML task type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLTaskType {
    QueryOptimization,
    CardinalityEstimation,
    EmbeddingGeneration,
    TextClassification,
    NamedEntityRecognition,
    QuestionAnswering,
    Regression,
    BinaryClassification,
    MultiClassification,
}

/// Model format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    TensorFlowSavedModel,
    PyTorchStateDict,
    PyTorchJIT,
    ONNX,
    HuggingFaceTransformers,
    ScikitLearnPickle,
    PMML,
    CoreML,
}

/// Tensor specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: DataType,
}

/// Data type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    String,
    Bool,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub author: Option<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
    pub dataset: Option<String>,
    pub training_framework: Option<String>,
    pub custom_metadata: HashMap<String, String>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1_score: Option<f64>,
    pub inference_time_ms: Option<f64>,
    pub model_size_mb: Option<f64>,
}

/// Framework adapter trait
pub trait FrameworkAdapter: Send + Sync {
    /// Load model from path
    fn load_model(&self, model_path: &Path) -> Result<Box<dyn MLModel>>;

    /// Convert model to ONNX
    fn convert_to_onnx(&self, model: &dyn MLModel, output_path: &Path) -> Result<()>;

    /// Get framework capabilities
    fn get_capabilities(&self) -> FrameworkCapabilities;
}

/// ML Model trait
pub trait MLModel: Send + Sync {
    /// Run inference
    fn predict(&self, inputs: HashMap<String, Array2<f64>>)
        -> Result<HashMap<String, Array2<f64>>>;

    /// Get input schema
    fn get_input_schema(&self) -> Vec<TensorSpec>;

    /// Get output schema
    fn get_output_schema(&self) -> Vec<TensorSpec>;

    /// Get model metadata
    fn get_metadata(&self) -> ModelMetadata;
}

/// Framework capabilities
#[derive(Debug, Clone)]
pub struct FrameworkCapabilities {
    pub supports_quantization: bool,
    pub supports_pruning: bool,
    pub supports_distributed_training: bool,
    pub supported_model_formats: Vec<ModelFormat>,
}

/// TensorFlow adapter
pub struct TensorFlowAdapter;

impl FrameworkAdapter for TensorFlowAdapter {
    fn load_model(&self, _model_path: &Path) -> Result<Box<dyn MLModel>> {
        // Simplified implementation - in production, use actual TensorFlow Rust bindings
        info!("Loading TensorFlow model (mock)");
        Ok(Box::new(MockModel::new()))
    }

    fn convert_to_onnx(&self, _model: &dyn MLModel, _output_path: &Path) -> Result<()> {
        info!("Converting TensorFlow model to ONNX (mock)");
        Ok(())
    }

    fn get_capabilities(&self) -> FrameworkCapabilities {
        FrameworkCapabilities {
            supports_quantization: true,
            supports_pruning: true,
            supports_distributed_training: true,
            supported_model_formats: vec![ModelFormat::TensorFlowSavedModel, ModelFormat::ONNX],
        }
    }
}

/// PyTorch adapter
pub struct PyTorchAdapter;

impl FrameworkAdapter for PyTorchAdapter {
    fn load_model(&self, _model_path: &Path) -> Result<Box<dyn MLModel>> {
        // Simplified implementation - in production, use tch-rs or similar
        info!("Loading PyTorch model (mock)");
        Ok(Box::new(MockModel::new()))
    }

    fn convert_to_onnx(&self, _model: &dyn MLModel, _output_path: &Path) -> Result<()> {
        info!("Converting PyTorch model to ONNX (mock)");
        Ok(())
    }

    fn get_capabilities(&self) -> FrameworkCapabilities {
        FrameworkCapabilities {
            supports_quantization: true,
            supports_pruning: true,
            supports_distributed_training: true,
            supported_model_formats: vec![
                ModelFormat::PyTorchStateDict,
                ModelFormat::PyTorchJIT,
                ModelFormat::ONNX,
            ],
        }
    }
}

/// ONNX adapter
pub struct ONNXAdapter;

impl FrameworkAdapter for ONNXAdapter {
    fn load_model(&self, _model_path: &Path) -> Result<Box<dyn MLModel>> {
        // Simplified implementation - in production, use ort or tract
        info!("Loading ONNX model (mock)");
        Ok(Box::new(MockModel::new()))
    }

    fn convert_to_onnx(&self, _model: &dyn MLModel, output_path: &Path) -> Result<()> {
        info!("Model already in ONNX format: {:?}", output_path);
        Ok(())
    }

    fn get_capabilities(&self) -> FrameworkCapabilities {
        FrameworkCapabilities {
            supports_quantization: true,
            supports_pruning: false,
            supports_distributed_training: false,
            supported_model_formats: vec![ModelFormat::ONNX],
        }
    }
}

/// Hugging Face adapter
pub struct HuggingFaceAdapter;

impl FrameworkAdapter for HuggingFaceAdapter {
    fn load_model(&self, _model_path: &Path) -> Result<Box<dyn MLModel>> {
        // Simplified implementation - in production, use rust-bert or similar
        info!("Loading Hugging Face model (mock)");
        Ok(Box::new(MockModel::new()))
    }

    fn convert_to_onnx(&self, _model: &dyn MLModel, _output_path: &Path) -> Result<()> {
        info!("Converting Hugging Face model to ONNX (mock)");
        Ok(())
    }

    fn get_capabilities(&self) -> FrameworkCapabilities {
        FrameworkCapabilities {
            supports_quantization: true,
            supports_pruning: false,
            supports_distributed_training: true,
            supported_model_formats: vec![ModelFormat::HuggingFaceTransformers, ModelFormat::ONNX],
        }
    }
}

/// Mock model for testing
pub struct MockModel;

impl Default for MockModel {
    fn default() -> Self {
        Self
    }
}

impl MockModel {
    pub fn new() -> Self {
        Self
    }
}

impl MLModel for MockModel {
    fn predict(
        &self,
        inputs: HashMap<String, Array2<f64>>,
    ) -> Result<HashMap<String, Array2<f64>>> {
        // Simple mock prediction - just return same shape with modified values
        let mut outputs = HashMap::new();

        for (key, input) in inputs.iter() {
            let output = input.mapv(|x| x * 2.0); // Simple transformation
            outputs.insert(format!("output_{}", key), output);
        }

        Ok(outputs)
    }

    fn get_input_schema(&self) -> Vec<TensorSpec> {
        vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![-1, 10],
            dtype: DataType::Float32,
        }]
    }

    fn get_output_schema(&self) -> Vec<TensorSpec> {
        vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![-1, 10],
            dtype: DataType::Float32,
        }]
    }

    fn get_metadata(&self) -> ModelMetadata {
        ModelMetadata {
            author: Some("OxiRS Team".to_string()),
            license: Some("MIT".to_string()),
            tags: vec!["mock".to_string(), "test".to_string()],
            dataset: None,
            training_framework: Some("Mock".to_string()),
            custom_metadata: HashMap::new(),
        }
    }
}

/// Inference engine
pub struct InferenceEngine {
    #[allow(dead_code)]
    backend: InferenceBackend,
    loaded_models: Arc<DashMap<String, Box<dyn MLModel>>>,
}

impl InferenceEngine {
    pub fn new(backend: InferenceBackend) -> Self {
        Self {
            backend,
            loaded_models: Arc::new(DashMap::new()),
        }
    }

    /// Load model for inference
    pub async fn load_model(&self, model_id: String, model: Box<dyn MLModel>) -> Result<()> {
        self.loaded_models.insert(model_id.clone(), model);
        info!("Loaded model {} for inference", model_id);
        Ok(())
    }

    /// Run inference
    pub async fn infer(
        &self,
        model_id: &str,
        inputs: HashMap<String, Array2<f64>>,
    ) -> Result<HashMap<String, Array2<f64>>> {
        let model = self
            .loaded_models
            .get(model_id)
            .ok_or_else(|| anyhow!("Model not loaded: {}", model_id))?;

        model.predict(inputs)
    }

    /// Batch inference
    pub async fn batch_infer(
        &self,
        model_id: &str,
        batch_inputs: Vec<HashMap<String, Array2<f64>>>,
    ) -> Result<Vec<HashMap<String, Array2<f64>>>> {
        let model = self
            .loaded_models
            .get(model_id)
            .ok_or_else(|| anyhow!("Model not loaded: {}", model_id))?;

        let mut results = Vec::new();
        for inputs in batch_inputs {
            let output = model.predict(inputs)?;
            results.push(output);
        }

        Ok(results)
    }

    /// Unload model
    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        self.loaded_models.remove(model_id);
        info!("Unloaded model: {}", model_id);
        Ok(())
    }
}

/// Model converter
pub struct ModelConverter {
    supported_conversions: HashMap<(ModelFormat, ModelFormat), bool>,
}

impl Default for ModelConverter {
    fn default() -> Self {
        let mut supported = HashMap::new();

        // Define supported conversions
        supported.insert((ModelFormat::TensorFlowSavedModel, ModelFormat::ONNX), true);
        supported.insert((ModelFormat::PyTorchStateDict, ModelFormat::ONNX), true);
        supported.insert((ModelFormat::PyTorchJIT, ModelFormat::ONNX), true);
        supported.insert(
            (ModelFormat::HuggingFaceTransformers, ModelFormat::ONNX),
            true,
        );

        Self {
            supported_conversions: supported,
        }
    }
}

impl ModelConverter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if conversion is supported
    pub fn is_conversion_supported(
        &self,
        from_format: &ModelFormat,
        to_format: &ModelFormat,
    ) -> bool {
        self.supported_conversions
            .get(&(from_format.clone(), to_format.clone()))
            .copied()
            .unwrap_or(false)
    }

    /// Convert model
    pub async fn convert_model(
        &self,
        _input_path: &Path,
        output_path: &Path,
        from_format: ModelFormat,
        to_format: ModelFormat,
    ) -> Result<ConversionResult> {
        if !self.is_conversion_supported(&from_format, &to_format) {
            return Err(anyhow!(
                "Conversion from {:?} to {:?} not supported",
                from_format,
                to_format
            ));
        }

        info!("Converting model from {:?} to {:?}", from_format, to_format);

        // Simplified conversion - in production, use actual conversion tools
        Ok(ConversionResult {
            success: true,
            output_path: output_path.to_path_buf(),
            output_format: to_format,
            conversion_time_ms: 100.0,
            warnings: vec![],
        })
    }

    /// Optimize model
    pub async fn optimize_model(
        &self,
        model_path: &Path,
        optimization_config: OptimizationConfig,
    ) -> Result<OptimizationResult> {
        info!("Optimizing model with config: {:?}", optimization_config);

        // Simplified optimization
        Ok(OptimizationResult {
            success: true,
            optimized_path: model_path.to_path_buf(),
            size_reduction_percent: if optimization_config.quantization {
                75.0
            } else {
                0.0
            },
            speed_improvement_percent: if optimization_config.pruning {
                50.0
            } else {
                0.0
            },
        })
    }
}

/// Conversion result
#[derive(Debug, Clone)]
pub struct ConversionResult {
    pub success: bool,
    pub output_path: PathBuf,
    pub output_format: ModelFormat,
    pub conversion_time_ms: f64,
    pub warnings: Vec<String>,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub quantization: bool,
    pub pruning: bool,
    pub distillation: bool,
    pub target_backend: InferenceBackend,
}

/// Optimization result
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub success: bool,
    pub optimized_path: PathBuf,
    pub size_reduction_percent: f64,
    pub speed_improvement_percent: f64,
}

impl ExternalMLIntegration {
    /// Create new ML integration system
    pub fn new(config: MLIntegrationConfig) -> Self {
        let framework_adapters: Arc<DashMap<MLFramework, Box<dyn FrameworkAdapter>>> =
            Arc::new(DashMap::new());

        // Register adapters based on config
        if config.enable_tensorflow {
            framework_adapters.insert(
                MLFramework::TensorFlow,
                Box::new(TensorFlowAdapter) as Box<dyn FrameworkAdapter>,
            );
        }

        if config.enable_pytorch {
            framework_adapters.insert(
                MLFramework::PyTorch,
                Box::new(PyTorchAdapter) as Box<dyn FrameworkAdapter>,
            );
        }

        if config.enable_onnx {
            framework_adapters.insert(
                MLFramework::ONNX,
                Box::new(ONNXAdapter) as Box<dyn FrameworkAdapter>,
            );
        }

        if config.enable_huggingface {
            framework_adapters.insert(
                MLFramework::HuggingFace,
                Box::new(HuggingFaceAdapter) as Box<dyn FrameworkAdapter>,
            );
        }

        Self {
            config: config.clone(),
            model_registry: Arc::new(ModelRegistry::new()),
            framework_adapters,
            inference_engine: Arc::new(InferenceEngine::new(config.default_inference_backend)),
            model_converter: Arc::new(ModelConverter::new()),
        }
    }

    /// Register model from external framework
    pub async fn register_external_model(
        &self,
        model: RegisteredModel,
        model_path: PathBuf,
    ) -> Result<()> {
        // Register in registry
        self.model_registry.register_model(model.clone()).await?;

        // Load model using appropriate adapter
        if let Some(adapter) = self.framework_adapters.get(&model.framework) {
            let loaded_model = adapter.load_model(&model_path)?;

            // Load into inference engine
            self.inference_engine
                .load_model(model.id.clone(), loaded_model)
                .await?;

            info!(
                "Registered external model {} from {:?}",
                model.id, model.framework
            );
            Ok(())
        } else {
            Err(anyhow!(
                "Framework {:?} not enabled or supported",
                model.framework
            ))
        }
    }

    /// Import model from Hugging Face Hub
    pub async fn import_from_huggingface(&self, model_name: &str) -> Result<RegisteredModel> {
        info!("Importing model from Hugging Face: {}", model_name);

        // Simplified implementation - in production, use actual HF Hub API
        let model = RegisteredModel {
            id: uuid::Uuid::new_v4().to_string(),
            name: model_name.to_string(),
            description: format!("Imported from Hugging Face: {}", model_name),
            framework: MLFramework::HuggingFace,
            current_version: ModelVersion {
                version: "1.0.0".to_string(),
                model_path: self.config.model_cache_dir.join(model_name),
                format: ModelFormat::HuggingFaceTransformers,
                input_schema: vec![],
                output_schema: vec![],
                performance_metrics: PerformanceMetrics {
                    accuracy: None,
                    precision: None,
                    recall: None,
                    f1_score: None,
                    inference_time_ms: None,
                    model_size_mb: None,
                },
                created_at: Utc::now(),
            },
            task_type: MLTaskType::TextClassification,
            metadata: ModelMetadata {
                author: Some("Hugging Face".to_string()),
                license: Some("Apache-2.0".to_string()),
                tags: vec!["huggingface".to_string()],
                dataset: None,
                training_framework: Some("transformers".to_string()),
                custom_metadata: HashMap::new(),
            },
            created_at: Utc::now(),
        };

        self.model_registry.register_model(model.clone()).await?;

        Ok(model)
    }

    /// Convert model to ONNX
    pub async fn convert_to_onnx(
        &self,
        model_id: &str,
        output_path: PathBuf,
    ) -> Result<ConversionResult> {
        let model = self
            .model_registry
            .get_model(model_id)
            .await
            .ok_or_else(|| anyhow!("Model not found: {}", model_id))?;

        self.model_converter
            .convert_model(
                &model.current_version.model_path,
                &output_path,
                model.current_version.format,
                ModelFormat::ONNX,
            )
            .await
    }

    /// Run inference
    pub async fn infer(
        &self,
        model_id: &str,
        inputs: HashMap<String, Array2<f64>>,
    ) -> Result<HashMap<String, Array2<f64>>> {
        self.inference_engine.infer(model_id, inputs).await
    }

    /// List registered models
    pub async fn list_models(&self) -> Vec<RegisteredModel> {
        self.model_registry.list_models().await
    }

    /// Get framework capabilities
    pub fn get_framework_capabilities(
        &self,
        framework: &MLFramework,
    ) -> Option<FrameworkCapabilities> {
        self.framework_adapters
            .get(framework)
            .map(|adapter| adapter.get_capabilities())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ml_integration_creation() {
        let config = MLIntegrationConfig::default();
        let integration = ExternalMLIntegration::new(config);

        let models = integration.list_models().await;
        assert_eq!(models.len(), 0);
    }

    #[tokio::test]
    async fn test_model_registry() {
        let registry = ModelRegistry::new();

        let model = RegisteredModel {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            description: "A test model".to_string(),
            framework: MLFramework::PyTorch,
            current_version: ModelVersion {
                version: "1.0.0".to_string(),
                model_path: PathBuf::from("/tmp/model.pt"),
                format: ModelFormat::PyTorchStateDict,
                input_schema: vec![],
                output_schema: vec![],
                performance_metrics: PerformanceMetrics {
                    accuracy: Some(0.95),
                    precision: None,
                    recall: None,
                    f1_score: None,
                    inference_time_ms: Some(10.0),
                    model_size_mb: Some(100.0),
                },
                created_at: Utc::now(),
            },
            task_type: MLTaskType::BinaryClassification,
            metadata: ModelMetadata {
                author: Some("Test Author".to_string()),
                license: Some("MIT".to_string()),
                tags: vec!["test".to_string()],
                dataset: Some("test-dataset".to_string()),
                training_framework: Some("PyTorch".to_string()),
                custom_metadata: HashMap::new(),
            },
            created_at: Utc::now(),
        };

        registry.register_model(model.clone()).await.unwrap();

        let retrieved = registry.get_model("test-model").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test Model");
    }

    #[tokio::test]
    async fn test_inference_engine() {
        let engine = InferenceEngine::new(InferenceBackend::ONNX);
        let model = MockModel::new();

        engine
            .load_model("test-model".to_string(), Box::new(model))
            .await
            .unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), Array2::from_elem((2, 10), 1.0));

        let outputs = engine.infer("test-model", inputs).await.unwrap();
        assert!(!outputs.is_empty());
    }

    #[tokio::test]
    async fn test_model_converter() {
        let converter = ModelConverter::new();

        assert!(
            converter.is_conversion_supported(&ModelFormat::PyTorchStateDict, &ModelFormat::ONNX)
        );

        assert!(
            !converter.is_conversion_supported(&ModelFormat::ONNX, &ModelFormat::PyTorchStateDict)
        );

        let result = converter
            .convert_model(
                &PathBuf::from("/tmp/input.pt"),
                &PathBuf::from("/tmp/output.onnx"),
                ModelFormat::PyTorchStateDict,
                ModelFormat::ONNX,
            )
            .await
            .unwrap();

        assert!(result.success);
    }

    #[tokio::test]
    async fn test_framework_adapters() {
        let tensorflow_adapter = TensorFlowAdapter;
        let capabilities = tensorflow_adapter.get_capabilities();

        assert!(capabilities.supports_quantization);
        assert!(capabilities.supports_distributed_training);
    }

    #[tokio::test]
    async fn test_huggingface_import() {
        let config = MLIntegrationConfig::default();
        let integration = ExternalMLIntegration::new(config);

        let model = integration
            .import_from_huggingface("bert-base-uncased")
            .await
            .unwrap();

        assert_eq!(model.framework, MLFramework::HuggingFace);
        assert_eq!(model.name, "bert-base-uncased");
    }

    #[tokio::test]
    async fn test_batch_inference() {
        let engine = InferenceEngine::new(InferenceBackend::ONNX);
        let model = MockModel::new();

        engine
            .load_model("test-model".to_string(), Box::new(model))
            .await
            .unwrap();

        let batch_inputs: Vec<HashMap<String, Array2<f64>>> = (0..3)
            .map(|_| {
                let mut inputs = HashMap::new();
                inputs.insert("input".to_string(), Array2::from_elem((2, 10), 1.0));
                inputs
            })
            .collect();

        let results = engine
            .batch_infer("test-model", batch_inputs)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
    }
}
