//! Model validation and integrity checks for embedding models
//!
//! This module provides comprehensive validation capabilities including:
//! - Checksum validation (SHA-256, BLAKE3) for model integrity
//! - Dimension consistency checks across model layers
//! - Model signature and format verification
//! - Model metadata validation
//!
//! All operations use proper error handling with no unwrap() calls.

use anyhow::{anyhow, Context, Result};
use blake3::Hasher as Blake3Hasher;
use scirs2_core::ndarray_ext::{ArrayView, Ix2};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use tokio::fs::File as AsyncFile;
use tokio::io::AsyncReadExt;
use tracing::{debug, error, info};

/// Supported checksum algorithms for model validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChecksumAlgorithm {
    /// SHA-256 cryptographic hash
    Sha256,
    /// BLAKE3 high-performance hash
    Blake3,
}

/// Model format types supported by validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// ONNX format (magic bytes: "ONNX")
    Onnx,
    /// SafeTensors format
    SafeTensors,
    /// Custom OxiRS embedding format
    OxirsEmbed,
    /// PyTorch format
    PyTorch,
    /// TensorFlow SavedModel
    TensorFlow,
    /// Unknown format
    Unknown,
}

/// Model validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Checksum algorithm to use
    pub checksum_algorithm: ChecksumAlgorithm,
    /// Whether to validate checksums
    pub validate_checksum: bool,
    /// Whether to validate dimensions
    pub validate_dimensions: bool,
    /// Whether to validate model signature
    pub validate_signature: bool,
    /// Whether to validate metadata
    pub validate_metadata: bool,
    /// Expected model format (None = auto-detect)
    pub expected_format: Option<ModelFormat>,
    /// Required metadata fields
    pub required_metadata_fields: Vec<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            checksum_algorithm: ChecksumAlgorithm::Blake3,
            validate_checksum: true,
            validate_dimensions: true,
            validate_signature: true,
            validate_metadata: true,
            expected_format: None,
            required_metadata_fields: vec![
                "model_name".to_string(),
                "embedding_dim".to_string(),
                "version".to_string(),
            ],
        }
    }
}

/// Model metadata for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub model_name: String,
    /// Model version
    pub version: String,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Input dimension
    pub input_dim: Option<usize>,
    /// Output dimension
    pub output_dim: Option<usize>,
    /// Model format
    pub format: ModelFormat,
    /// Expected checksum
    pub checksum: Option<String>,
    /// Checksum algorithm used
    pub checksum_algorithm: Option<ChecksumAlgorithm>,
    /// Additional metadata
    pub extra: HashMap<String, serde_json::Value>,
}

/// Validation result status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Validation passed
    Valid,
    /// Validation failed
    Invalid,
    /// Validation skipped
    Skipped,
    /// Validation warning (non-critical issues)
    Warning,
}

/// Validation result for a specific check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Type of validation performed
    pub validation_type: String,
    /// Result status
    pub status: ValidationStatus,
    /// Detailed message
    pub message: String,
    /// Additional details
    pub details: Option<serde_json::Value>,
}

/// Comprehensive validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Model path validated
    pub model_path: PathBuf,
    /// Overall validation status
    pub overall_status: ValidationStatus,
    /// Individual validation results
    pub results: Vec<ValidationResult>,
    /// Validation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ValidationReport {
    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.overall_status == ValidationStatus::Valid
            || self.overall_status == ValidationStatus::Warning
    }

    /// Get failed validations
    pub fn failed_validations(&self) -> Vec<&ValidationResult> {
        self.results
            .iter()
            .filter(|r| r.status == ValidationStatus::Invalid)
            .collect()
    }
}

/// Model validator for embedding models
pub struct ModelValidator {
    config: ValidationConfig,
}

impl ModelValidator {
    /// Create a new model validator with default configuration
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Create a model validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate a model file synchronously
    pub fn validate(
        &self,
        model_path: &Path,
        metadata: &ModelMetadata,
    ) -> Result<ValidationReport> {
        info!("Starting validation for model: {}", model_path.display());

        let mut results = Vec::new();

        // Validate checksum
        if self.config.validate_checksum {
            match self.validate_checksum_sync(model_path, metadata) {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!("Checksum validation failed: {}", e);
                    results.push(ValidationResult {
                        validation_type: "checksum".to_string(),
                        status: ValidationStatus::Invalid,
                        message: format!("Checksum validation error: {}", e),
                        details: None,
                    });
                }
            }
        }

        // Validate signature
        if self.config.validate_signature {
            match self.validate_signature(model_path, metadata) {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!("Signature validation failed: {}", e);
                    results.push(ValidationResult {
                        validation_type: "signature".to_string(),
                        status: ValidationStatus::Invalid,
                        message: format!("Signature validation error: {}", e),
                        details: None,
                    });
                }
            }
        }

        // Validate metadata
        if self.config.validate_metadata {
            match self.validate_metadata(metadata) {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!("Metadata validation failed: {}", e);
                    results.push(ValidationResult {
                        validation_type: "metadata".to_string(),
                        status: ValidationStatus::Invalid,
                        message: format!("Metadata validation error: {}", e),
                        details: None,
                    });
                }
            }
        }

        // Determine overall status
        let overall_status = if results
            .iter()
            .any(|r| r.status == ValidationStatus::Invalid)
        {
            ValidationStatus::Invalid
        } else if results
            .iter()
            .any(|r| r.status == ValidationStatus::Warning)
        {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Valid
        };

        Ok(ValidationReport {
            model_path: model_path.to_path_buf(),
            overall_status,
            results,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Validate a model file asynchronously
    pub async fn validate_async(
        &self,
        model_path: &Path,
        metadata: &ModelMetadata,
    ) -> Result<ValidationReport> {
        info!(
            "Starting async validation for model: {}",
            model_path.display()
        );

        let mut results = Vec::new();

        // Validate checksum
        if self.config.validate_checksum {
            match self.validate_checksum_async(model_path, metadata).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!("Checksum validation failed: {}", e);
                    results.push(ValidationResult {
                        validation_type: "checksum".to_string(),
                        status: ValidationStatus::Invalid,
                        message: format!("Checksum validation error: {}", e),
                        details: None,
                    });
                }
            }
        }

        // Validate signature (async)
        if self.config.validate_signature {
            match self.validate_signature_async(model_path, metadata).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!("Signature validation failed: {}", e);
                    results.push(ValidationResult {
                        validation_type: "signature".to_string(),
                        status: ValidationStatus::Invalid,
                        message: format!("Signature validation error: {}", e),
                        details: None,
                    });
                }
            }
        }

        // Validate metadata
        if self.config.validate_metadata {
            match self.validate_metadata(metadata) {
                Ok(result) => results.push(result),
                Err(e) => {
                    error!("Metadata validation failed: {}", e);
                    results.push(ValidationResult {
                        validation_type: "metadata".to_string(),
                        status: ValidationStatus::Invalid,
                        message: format!("Metadata validation error: {}", e),
                        details: None,
                    });
                }
            }
        }

        // Determine overall status
        let overall_status = if results
            .iter()
            .any(|r| r.status == ValidationStatus::Invalid)
        {
            ValidationStatus::Invalid
        } else if results
            .iter()
            .any(|r| r.status == ValidationStatus::Warning)
        {
            ValidationStatus::Warning
        } else {
            ValidationStatus::Valid
        };

        Ok(ValidationReport {
            model_path: model_path.to_path_buf(),
            overall_status,
            results,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Validate dimension consistency
    pub fn validate_dimensions(
        &self,
        embedding_dim: usize,
        input_tensors: &[ArrayView<f32, Ix2>],
        output_tensors: &[ArrayView<f32, Ix2>],
    ) -> Result<ValidationResult> {
        debug!(
            "Validating dimensions: embedding_dim={}, input_tensors={}, output_tensors={}",
            embedding_dim,
            input_tensors.len(),
            output_tensors.len()
        );

        // Check input tensor dimensions
        for (i, tensor) in input_tensors.iter().enumerate() {
            let shape = tensor.shape();
            if shape.len() != 2 {
                return Ok(ValidationResult {
                    validation_type: "dimension".to_string(),
                    status: ValidationStatus::Invalid,
                    message: format!(
                        "Input tensor {} has invalid rank: expected 2, got {}",
                        i,
                        shape.len()
                    ),
                    details: Some(serde_json::json!({ "tensor_index": i, "shape": shape })),
                });
            }

            // Check if embedding dimension matches
            if shape[1] != embedding_dim {
                return Ok(ValidationResult {
                    validation_type: "dimension".to_string(),
                    status: ValidationStatus::Invalid,
                    message: format!(
                        "Input tensor {} dimension mismatch: expected {}, got {}",
                        i, embedding_dim, shape[1]
                    ),
                    details: Some(
                        serde_json::json!({ "tensor_index": i, "expected": embedding_dim, "actual": shape[1] }),
                    ),
                });
            }
        }

        // Check output tensor dimensions
        for (i, tensor) in output_tensors.iter().enumerate() {
            let shape = tensor.shape();
            if shape.len() != 2 {
                return Ok(ValidationResult {
                    validation_type: "dimension".to_string(),
                    status: ValidationStatus::Invalid,
                    message: format!(
                        "Output tensor {} has invalid rank: expected 2, got {}",
                        i,
                        shape.len()
                    ),
                    details: Some(serde_json::json!({ "tensor_index": i, "shape": shape })),
                });
            }

            if shape[1] != embedding_dim {
                return Ok(ValidationResult {
                    validation_type: "dimension".to_string(),
                    status: ValidationStatus::Invalid,
                    message: format!(
                        "Output tensor {} dimension mismatch: expected {}, got {}",
                        i, embedding_dim, shape[1]
                    ),
                    details: Some(
                        serde_json::json!({ "tensor_index": i, "expected": embedding_dim, "actual": shape[1] }),
                    ),
                });
            }
        }

        Ok(ValidationResult {
            validation_type: "dimension".to_string(),
            status: ValidationStatus::Valid,
            message: "All dimension checks passed".to_string(),
            details: Some(serde_json::json!({
                "embedding_dim": embedding_dim,
                "input_tensors_validated": input_tensors.len(),
                "output_tensors_validated": output_tensors.len(),
            })),
        })
    }

    /// Validate checksum synchronously
    fn validate_checksum_sync(
        &self,
        model_path: &Path,
        metadata: &ModelMetadata,
    ) -> Result<ValidationResult> {
        let expected_checksum = metadata
            .checksum
            .as_ref()
            .ok_or_else(|| anyhow!("No checksum provided in metadata"))?;

        let checksum_algo = metadata
            .checksum_algorithm
            .unwrap_or(self.config.checksum_algorithm);

        let computed_checksum = self.compute_checksum_sync(model_path, checksum_algo)?;

        if computed_checksum == *expected_checksum {
            Ok(ValidationResult {
                validation_type: "checksum".to_string(),
                status: ValidationStatus::Valid,
                message: format!("Checksum validation passed ({:?})", checksum_algo),
                details: Some(serde_json::json!({
                    "algorithm": checksum_algo,
                    "checksum": computed_checksum,
                })),
            })
        } else {
            Ok(ValidationResult {
                validation_type: "checksum".to_string(),
                status: ValidationStatus::Invalid,
                message: format!("Checksum mismatch ({:?})", checksum_algo),
                details: Some(serde_json::json!({
                    "algorithm": checksum_algo,
                    "expected": expected_checksum,
                    "actual": computed_checksum,
                })),
            })
        }
    }

    /// Validate checksum asynchronously
    async fn validate_checksum_async(
        &self,
        model_path: &Path,
        metadata: &ModelMetadata,
    ) -> Result<ValidationResult> {
        let expected_checksum = metadata
            .checksum
            .as_ref()
            .ok_or_else(|| anyhow!("No checksum provided in metadata"))?;

        let checksum_algo = metadata
            .checksum_algorithm
            .unwrap_or(self.config.checksum_algorithm);

        let computed_checksum = self
            .compute_checksum_async(model_path, checksum_algo)
            .await?;

        if computed_checksum == *expected_checksum {
            Ok(ValidationResult {
                validation_type: "checksum".to_string(),
                status: ValidationStatus::Valid,
                message: format!("Checksum validation passed ({:?})", checksum_algo),
                details: Some(serde_json::json!({
                    "algorithm": checksum_algo,
                    "checksum": computed_checksum,
                })),
            })
        } else {
            Ok(ValidationResult {
                validation_type: "checksum".to_string(),
                status: ValidationStatus::Invalid,
                message: format!("Checksum mismatch ({:?})", checksum_algo),
                details: Some(serde_json::json!({
                    "algorithm": checksum_algo,
                    "expected": expected_checksum,
                    "actual": computed_checksum,
                })),
            })
        }
    }

    /// Compute checksum synchronously
    fn compute_checksum_sync(&self, path: &Path, algorithm: ChecksumAlgorithm) -> Result<String> {
        let file = File::open(path).context("Failed to open model file")?;
        let mut reader = BufReader::new(file);

        match algorithm {
            ChecksumAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                let mut buffer = [0u8; 8192];
                loop {
                    let count = reader.read(&mut buffer).context("Failed to read file")?;
                    if count == 0 {
                        break;
                    }
                    hasher.update(&buffer[..count]);
                }
                Ok(format!("{:x}", hasher.finalize()))
            }
            ChecksumAlgorithm::Blake3 => {
                let mut hasher = Blake3Hasher::new();
                let mut buffer = [0u8; 8192];
                loop {
                    let count = reader.read(&mut buffer).context("Failed to read file")?;
                    if count == 0 {
                        break;
                    }
                    hasher.update(&buffer[..count]);
                }
                Ok(hasher.finalize().to_hex().to_string())
            }
        }
    }

    /// Compute checksum asynchronously
    async fn compute_checksum_async(
        &self,
        path: &Path,
        algorithm: ChecksumAlgorithm,
    ) -> Result<String> {
        let mut file = AsyncFile::open(path)
            .await
            .context("Failed to open model file")?;

        match algorithm {
            ChecksumAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                let mut buffer = vec![0u8; 8192];
                loop {
                    let count = file
                        .read(&mut buffer)
                        .await
                        .context("Failed to read file")?;
                    if count == 0 {
                        break;
                    }
                    hasher.update(&buffer[..count]);
                }
                Ok(format!("{:x}", hasher.finalize()))
            }
            ChecksumAlgorithm::Blake3 => {
                let mut hasher = Blake3Hasher::new();
                let mut buffer = vec![0u8; 8192];
                loop {
                    let count = file
                        .read(&mut buffer)
                        .await
                        .context("Failed to read file")?;
                    if count == 0 {
                        break;
                    }
                    hasher.update(&buffer[..count]);
                }
                Ok(hasher.finalize().to_hex().to_string())
            }
        }
    }

    /// Validate model signature and format
    fn validate_signature(
        &self,
        model_path: &Path,
        metadata: &ModelMetadata,
    ) -> Result<ValidationResult> {
        let file = File::open(model_path).context("Failed to open model file")?;
        let mut reader = BufReader::new(file);
        let mut magic_bytes = [0u8; 8];

        reader
            .read_exact(&mut magic_bytes)
            .context("Failed to read magic bytes")?;

        let detected_format = Self::detect_format(&magic_bytes);

        // Check if format matches expected
        if let Some(expected) = self.config.expected_format {
            if detected_format != expected && detected_format != ModelFormat::Unknown {
                return Ok(ValidationResult {
                    validation_type: "signature".to_string(),
                    status: ValidationStatus::Invalid,
                    message: format!(
                        "Format mismatch: expected {:?}, got {:?}",
                        expected, detected_format
                    ),
                    details: Some(serde_json::json!({
                        "expected_format": expected,
                        "detected_format": detected_format,
                        "magic_bytes": magic_bytes,
                    })),
                });
            }
        }

        // Check if format matches metadata
        if detected_format != metadata.format && detected_format != ModelFormat::Unknown {
            return Ok(ValidationResult {
                validation_type: "signature".to_string(),
                status: ValidationStatus::Warning,
                message: format!(
                    "Format mismatch with metadata: metadata says {:?}, detected {:?}",
                    metadata.format, detected_format
                ),
                details: Some(serde_json::json!({
                    "metadata_format": metadata.format,
                    "detected_format": detected_format,
                })),
            });
        }

        Ok(ValidationResult {
            validation_type: "signature".to_string(),
            status: ValidationStatus::Valid,
            message: format!("Model signature valid: {:?}", detected_format),
            details: Some(serde_json::json!({
                "format": detected_format,
            })),
        })
    }

    /// Validate model signature asynchronously
    async fn validate_signature_async(
        &self,
        model_path: &Path,
        metadata: &ModelMetadata,
    ) -> Result<ValidationResult> {
        let mut file = AsyncFile::open(model_path)
            .await
            .context("Failed to open model file")?;
        let mut magic_bytes = [0u8; 8];

        file.read_exact(&mut magic_bytes)
            .await
            .context("Failed to read magic bytes")?;

        let detected_format = Self::detect_format(&magic_bytes);

        // Check if format matches expected
        if let Some(expected) = self.config.expected_format {
            if detected_format != expected && detected_format != ModelFormat::Unknown {
                return Ok(ValidationResult {
                    validation_type: "signature".to_string(),
                    status: ValidationStatus::Invalid,
                    message: format!(
                        "Format mismatch: expected {:?}, got {:?}",
                        expected, detected_format
                    ),
                    details: Some(serde_json::json!({
                        "expected_format": expected,
                        "detected_format": detected_format,
                        "magic_bytes": magic_bytes,
                    })),
                });
            }
        }

        // Check if format matches metadata
        if detected_format != metadata.format && detected_format != ModelFormat::Unknown {
            return Ok(ValidationResult {
                validation_type: "signature".to_string(),
                status: ValidationStatus::Warning,
                message: format!(
                    "Format mismatch with metadata: metadata says {:?}, detected {:?}",
                    metadata.format, detected_format
                ),
                details: Some(serde_json::json!({
                    "metadata_format": metadata.format,
                    "detected_format": detected_format,
                })),
            });
        }

        Ok(ValidationResult {
            validation_type: "signature".to_string(),
            status: ValidationStatus::Valid,
            message: format!("Model signature valid: {:?}", detected_format),
            details: Some(serde_json::json!({
                "format": detected_format,
            })),
        })
    }

    /// Detect model format from magic bytes
    fn detect_format(magic_bytes: &[u8]) -> ModelFormat {
        // ONNX: starts with "08 03" or has ONNX protobuf header
        if magic_bytes.starts_with(&[0x08, 0x03]) {
            return ModelFormat::Onnx;
        }

        // SafeTensors: JSON header
        if magic_bytes.starts_with(b"{") {
            return ModelFormat::SafeTensors;
        }

        // PyTorch: ZIP archive (PK header)
        if magic_bytes.starts_with(&[0x50, 0x4B, 0x03, 0x04]) {
            return ModelFormat::PyTorch;
        }

        // TensorFlow SavedModel: protobuf
        if magic_bytes.starts_with(&[0x0A]) {
            return ModelFormat::TensorFlow;
        }

        // OxiRS custom format: "OXIRS\0\0\0"
        if magic_bytes.starts_with(b"OXIRS") {
            return ModelFormat::OxirsEmbed;
        }

        ModelFormat::Unknown
    }

    /// Validate model metadata
    fn validate_metadata(&self, metadata: &ModelMetadata) -> Result<ValidationResult> {
        let mut missing_fields = Vec::new();

        // Check required fields
        for field in &self.config.required_metadata_fields {
            match field.as_str() {
                "model_name" if metadata.model_name.is_empty() => {
                    missing_fields.push("model_name".to_string());
                }
                "version" if metadata.version.is_empty() => {
                    missing_fields.push("version".to_string());
                }
                "embedding_dim" if metadata.embedding_dim == 0 => {
                    missing_fields.push("embedding_dim".to_string());
                }
                _ => {}
            }
        }

        if !missing_fields.is_empty() {
            return Ok(ValidationResult {
                validation_type: "metadata".to_string(),
                status: ValidationStatus::Invalid,
                message: format!("Missing required metadata fields: {:?}", missing_fields),
                details: Some(serde_json::json!({
                    "missing_fields": missing_fields,
                })),
            });
        }

        // Validate dimension values
        if metadata.embedding_dim == 0 {
            return Ok(ValidationResult {
                validation_type: "metadata".to_string(),
                status: ValidationStatus::Invalid,
                message: "Invalid embedding dimension: must be > 0".to_string(),
                details: None,
            });
        }

        Ok(ValidationResult {
            validation_type: "metadata".to_string(),
            status: ValidationStatus::Valid,
            message: "Metadata validation passed".to_string(),
            details: Some(serde_json::json!({
                "model_name": metadata.model_name,
                "version": metadata.version,
                "embedding_dim": metadata.embedding_dim,
            })),
        })
    }
}

impl Default for ModelValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::Array;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_checksum_sha256() {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(b"test data")
            .expect("Failed to write data");

        let validator = ModelValidator::new();
        let checksum = validator
            .compute_checksum_sync(temp_file.path(), ChecksumAlgorithm::Sha256)
            .expect("Failed to compute checksum");

        // Expected SHA-256 of "test data"
        assert_eq!(
            checksum,
            "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9"
        );
    }

    #[test]
    fn test_checksum_blake3() {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(b"test data")
            .expect("Failed to write data");

        let validator = ModelValidator::new();
        let checksum = validator
            .compute_checksum_sync(temp_file.path(), ChecksumAlgorithm::Blake3)
            .expect("Failed to compute checksum");

        // BLAKE3 should produce consistent output
        assert!(!checksum.is_empty());
        assert_eq!(checksum.len(), 64); // BLAKE3 produces 32-byte (64 hex chars) hash
    }

    #[test]
    fn test_dimension_validation_valid() {
        let validator = ModelValidator::new();
        let embedding_dim = 128;

        let input1 = Array::zeros((10, 128));
        let input2 = Array::zeros((20, 128));
        let output1 = Array::zeros((10, 128));

        let input_views = vec![input1.view(), input2.view()];
        let output_views = vec![output1.view()];

        let result = validator
            .validate_dimensions(embedding_dim, &input_views, &output_views)
            .expect("Validation failed");

        assert_eq!(result.status, ValidationStatus::Valid);
    }

    #[test]
    fn test_dimension_validation_invalid() {
        let validator = ModelValidator::new();
        let embedding_dim = 128;

        let input1 = Array::zeros((10, 128));
        let input2 = Array::zeros((20, 64)); // Wrong dimension
        let output1 = Array::zeros((10, 128));

        let input_views = vec![input1.view(), input2.view()];
        let output_views = vec![output1.view()];

        let result = validator
            .validate_dimensions(embedding_dim, &input_views, &output_views)
            .expect("Validation failed");

        assert_eq!(result.status, ValidationStatus::Invalid);
        assert!(result.message.contains("dimension mismatch"));
    }

    #[test]
    fn test_format_detection_onnx() {
        let magic = [0x08, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let format = ModelValidator::detect_format(&magic);
        assert_eq!(format, ModelFormat::Onnx);
    }

    #[test]
    fn test_format_detection_pytorch() {
        let magic = [0x50, 0x4B, 0x03, 0x04, 0x00, 0x00, 0x00, 0x00];
        let format = ModelValidator::detect_format(&magic);
        assert_eq!(format, ModelFormat::PyTorch);
    }

    #[test]
    fn test_format_detection_oxirs() {
        let magic = b"OXIRS\0\0\0";
        let format = ModelValidator::detect_format(magic);
        assert_eq!(format, ModelFormat::OxirsEmbed);
    }

    #[test]
    fn test_metadata_validation_valid() {
        let validator = ModelValidator::new();
        let metadata = ModelMetadata {
            model_name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            embedding_dim: 128,
            input_dim: Some(128),
            output_dim: Some(128),
            format: ModelFormat::OxirsEmbed,
            checksum: None,
            checksum_algorithm: None,
            extra: HashMap::new(),
        };

        let result = validator
            .validate_metadata(&metadata)
            .expect("Validation failed");
        assert_eq!(result.status, ValidationStatus::Valid);
    }

    #[test]
    fn test_metadata_validation_missing_fields() {
        let validator = ModelValidator::new();
        let metadata = ModelMetadata {
            model_name: "".to_string(), // Missing
            version: "1.0.0".to_string(),
            embedding_dim: 128,
            input_dim: Some(128),
            output_dim: Some(128),
            format: ModelFormat::OxirsEmbed,
            checksum: None,
            checksum_algorithm: None,
            extra: HashMap::new(),
        };

        let result = validator
            .validate_metadata(&metadata)
            .expect("Validation failed");
        assert_eq!(result.status, ValidationStatus::Invalid);
        assert!(result.message.contains("Missing required"));
    }

    #[test]
    fn test_metadata_validation_invalid_dimension() {
        let validator = ModelValidator::new();
        let metadata = ModelMetadata {
            model_name: "test".to_string(),
            version: "1.0.0".to_string(),
            embedding_dim: 0, // Invalid
            input_dim: Some(128),
            output_dim: Some(128),
            format: ModelFormat::OxirsEmbed,
            checksum: None,
            checksum_algorithm: None,
            extra: HashMap::new(),
        };

        let result = validator
            .validate_metadata(&metadata)
            .expect("Validation failed");
        assert_eq!(result.status, ValidationStatus::Invalid);
    }

    #[tokio::test]
    async fn test_async_checksum() {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        temp_file
            .write_all(b"async test data")
            .expect("Failed to write data");

        let validator = ModelValidator::new();
        let checksum = validator
            .compute_checksum_async(temp_file.path(), ChecksumAlgorithm::Blake3)
            .await
            .expect("Failed to compute checksum");

        assert!(!checksum.is_empty());
    }

    #[test]
    fn test_validation_report_is_valid() {
        let report = ValidationReport {
            model_path: PathBuf::from("/test/model"),
            overall_status: ValidationStatus::Valid,
            results: vec![],
            timestamp: chrono::Utc::now(),
        };

        assert!(report.is_valid());
    }

    #[test]
    fn test_validation_report_failed_validations() {
        let report = ValidationReport {
            model_path: PathBuf::from("/test/model"),
            overall_status: ValidationStatus::Invalid,
            results: vec![
                ValidationResult {
                    validation_type: "checksum".to_string(),
                    status: ValidationStatus::Invalid,
                    message: "Checksum mismatch".to_string(),
                    details: None,
                },
                ValidationResult {
                    validation_type: "dimension".to_string(),
                    status: ValidationStatus::Valid,
                    message: "OK".to_string(),
                    details: None,
                },
            ],
            timestamp: chrono::Utc::now(),
        };

        let failed = report.failed_validations();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].validation_type, "checksum");
    }

    #[test]
    fn test_comprehensive_validation() {
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        // Write ONNX magic bytes
        temp_file
            .write_all(&[0x08, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
            .expect("Failed to write");
        temp_file
            .write_all(b"model data here")
            .expect("Failed to write");

        let validator = ModelValidator::new();

        // Compute actual checksum
        let checksum = validator
            .compute_checksum_sync(temp_file.path(), ChecksumAlgorithm::Blake3)
            .expect("Failed to compute checksum");

        let metadata = ModelMetadata {
            model_name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            embedding_dim: 128,
            input_dim: Some(128),
            output_dim: Some(128),
            format: ModelFormat::Onnx,
            checksum: Some(checksum),
            checksum_algorithm: Some(ChecksumAlgorithm::Blake3),
            extra: HashMap::new(),
        };

        let report = validator
            .validate(temp_file.path(), &metadata)
            .expect("Validation failed");

        assert!(report.is_valid());
        assert!(report.results.len() >= 2); // At least checksum and signature
    }
}
