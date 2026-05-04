//! Model persistence and serialization utilities

use crate::models::{ComplEx, DistMult, GNNConfig, GNNEmbedding, HoLE, HoLEConfig, RotatE, TransE};
use crate::{EmbeddingModel, ModelConfig, ModelStats};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info};

/// Errors specific to model persistence operations
#[derive(Debug, Error)]
pub enum PersistenceError {
    /// The requested export format requires an optional feature flag that is not enabled
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),
    /// The feature is gated behind a Cargo feature flag and not yet fully implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    /// IO error during persistence
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Serialisation / deserialisation error
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Model serialization format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    pub model_type: String,
    pub config: ModelConfig,
    pub stats: ModelStats,
    pub entity_mappings: std::collections::HashMap<String, usize>,
    pub relation_mappings: std::collections::HashMap<String, usize>,
    pub metadata: ModelMetadata,
}

/// Additional model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub version: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub trained_at: Option<chrono::DateTime<chrono::Utc>>,
    pub training_duration_seconds: Option<f64>,
    pub checksum: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            created_at: chrono::Utc::now(),
            trained_at: None,
            training_duration_seconds: None,
            checksum: None,
            description: None,
            tags: Vec::new(),
        }
    }
}

/// Model repository for managing multiple models
pub struct ModelRepository {
    base_path: String,
    models: std::collections::HashMap<String, ModelInfo>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub version: String,
    pub path: String,
    pub metadata: ModelMetadata,
}

impl ModelRepository {
    /// Create a new model repository
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref().to_string_lossy().to_string();

        // Create directory if it doesn't exist
        fs::create_dir_all(&base_path)?;

        let mut repo = Self {
            base_path,
            models: std::collections::HashMap::new(),
        };

        // Scan existing models
        repo.scan_models()?;

        Ok(repo)
    }

    /// Scan for existing models in the repository
    fn scan_models(&mut self) -> Result<()> {
        let entries = fs::read_dir(&self.base_path)?;

        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let model_path = entry.path();
                if let Some(model_name) = model_path.file_name() {
                    if let Some(name_str) = model_name.to_str() {
                        if let Ok(info) = self.load_model_info(name_str) {
                            self.models.insert(name_str.to_string(), info);
                        }
                    }
                }
            }
        }

        info!("Scanned {} models in repository", self.models.len());
        Ok(())
    }

    /// Load model information from directory
    fn load_model_info(&self, model_name: &str) -> Result<ModelInfo> {
        let base_path = &self.base_path;
        let model_path = format!("{base_path}/{model_name}");
        let metadata_path = format!("{model_path}/metadata.json");

        if !Path::new(&metadata_path).exists() {
            return Err(anyhow!("Model metadata not found: {metadata_path}"));
        }

        let metadata_content = fs::read_to_string(metadata_path)?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

        // Read the persisted model type if present
        let model_type_path = format!("{model_path}/model_type.json");
        let model_type = if Path::new(&model_type_path).exists() {
            let raw = fs::read_to_string(&model_type_path)?;
            // The file stores a JSON-encoded string (e.g. `"TransE"`); deserialise it.
            // If the file is somehow invalid JSON fall back to trimming quotes directly.
            match serde_json::from_str::<String>(&raw) {
                Ok(s) => s,
                Err(_) => raw.trim_matches('"').to_string(),
            }
        } else {
            "unknown".to_string()
        };

        Ok(ModelInfo {
            id: model_name.to_string(),
            name: model_name.to_string(),
            model_type,
            version: metadata.version.clone(),
            path: model_path,
            metadata,
        })
    }

    /// Save a model to the repository
    pub fn save_model(
        &mut self,
        model: &dyn EmbeddingModel,
        name: &str,
        description: Option<String>,
    ) -> Result<()> {
        let base_path = &self.base_path;
        let model_path = format!("{base_path}/{name}");
        fs::create_dir_all(&model_path)?;

        // Save model data
        let model_file = format!("{model_path}/model.bin");
        model.save(&model_file)?;

        // Save model type for later reconstruction
        let model_type_file = format!("{model_path}/model_type.json");
        fs::write(&model_type_file, serde_json::to_string(model.model_type())?)?;

        // Save metadata
        let metadata = ModelMetadata {
            description,
            trained_at: Some(chrono::Utc::now()),
            ..Default::default()
        };

        let metadata_file = format!("{model_path}/metadata.json");
        let metadata_content = serde_json::to_string_pretty(&metadata)?;
        fs::write(metadata_file, metadata_content)?;

        // Update repository index
        let info = ModelInfo {
            id: name.to_string(),
            name: name.to_string(),
            model_type: model.model_type().to_string(),
            version: metadata.version.clone(),
            path: model_path,
            metadata,
        };

        self.models.insert(name.to_string(), info);

        info!("Saved model '{}' to repository", name);
        Ok(())
    }

    /// Load a model from the repository
    pub fn load_model(&self, name: &str) -> Result<Box<dyn EmbeddingModel>> {
        let model_info = self
            .models
            .get(name)
            .ok_or_else(|| anyhow!("Model not found: {}", name))?;

        let model_path = &model_info.path;
        let model_file = format!("{model_path}/model.bin");

        // Dispatch based on the persisted model type
        let mut model: Box<dyn EmbeddingModel> = match model_info.model_type.as_str() {
            "TransE" => Box::new(TransE::new(ModelConfig::default())),
            "DistMult" => Box::new(DistMult::new(ModelConfig::default())),
            "ComplEx" => Box::new(ComplEx::new(ModelConfig::default())),
            "RotatE" => Box::new(RotatE::new(ModelConfig::default())),
            "HoLE" => Box::new(HoLE::new(HoLEConfig::default())),
            "GNN" | "GNNEmbedding" => Box::new(GNNEmbedding::new(GNNConfig::default())),
            other => {
                return Err(anyhow!(
                    "Cannot load model: unsupported model type '{}'",
                    other
                ))
            }
        };

        model.load(&model_file)?;

        info!(
            "Loaded model '{}' (type={}) from repository",
            name, model_info.model_type
        );
        Ok(model)
    }

    /// List all models in the repository
    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// Delete a model from the repository
    pub fn delete_model(&mut self, name: &str) -> Result<()> {
        if let Some(model_info) = self.models.remove(name) {
            fs::remove_dir_all(model_info.path)?;
            info!("Deleted model '{}' from repository", name);
            Ok(())
        } else {
            Err(anyhow!("Model not found: {}", name))
        }
    }

    /// Get model information
    pub fn get_model_info(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }
}

/// Checkpoint manager for training
pub struct CheckpointManager {
    checkpoint_dir: String,
    max_checkpoints: usize,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P, max_checkpoints: usize) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_string_lossy().to_string();
        fs::create_dir_all(&checkpoint_dir)?;

        Ok(Self {
            checkpoint_dir,
            max_checkpoints,
        })
    }

    /// Save a checkpoint
    pub fn save_checkpoint(
        &self,
        model: &dyn EmbeddingModel,
        epoch: usize,
        loss: f64,
    ) -> Result<String> {
        let checkpoint_name = format!("checkpoint_epoch_{epoch}_loss_{loss:.6}.bin");
        let checkpoint_dir = &self.checkpoint_dir;
        let checkpoint_path = format!("{checkpoint_dir}/{checkpoint_name}");

        model.save(&checkpoint_path)?;

        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;

        debug!("Saved checkpoint: {}", checkpoint_path);
        Ok(checkpoint_path)
    }

    /// Clean up old checkpoints, keeping only the most recent ones
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let entries = fs::read_dir(&self.checkpoint_dir)?;
        let mut checkpoints: Vec<_> = entries
            .filter_map(|entry| {
                entry.ok().and_then(|e| {
                    let path = e.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                        e.metadata()
                            .ok()
                            .map(|m| (path, m.modified().unwrap_or(std::time::UNIX_EPOCH)))
                    } else {
                        None
                    }
                })
            })
            .collect();

        checkpoints.sort_by_key(|(_, modified)| *modified);

        // Remove old checkpoints if we have too many
        if checkpoints.len() > self.max_checkpoints {
            let to_remove = checkpoints.len() - self.max_checkpoints;
            for (path, _) in checkpoints.iter().take(to_remove) {
                fs::remove_file(path)?;
                debug!("Removed old checkpoint: {:?}", path);
            }
        }

        Ok(())
    }

    /// List all checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<String>> {
        let entries = fs::read_dir(&self.checkpoint_dir)?;
        let mut checkpoints = Vec::new();

        for entry in entries {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".bin") {
                    checkpoints.push(name.to_string());
                }
            }
        }

        checkpoints.sort();
        Ok(checkpoints)
    }
}

/// Export models to different formats
pub struct ModelExporter;

impl ModelExporter {
    /// Export embeddings to CSV format
    pub fn export_to_csv(model: &dyn EmbeddingModel, output_path: &str) -> Result<()> {
        use std::io::Write;

        let mut file = fs::File::create(output_path)?;

        // Write header
        writeln!(file, "type,name,dimensions,embeddings")?;

        // Export entity embeddings
        for entity in model.get_entities() {
            if let Ok(embedding) = model.get_entity_embedding(&entity) {
                let values: Vec<String> = embedding.values.iter().map(|x| x.to_string()).collect();
                writeln!(
                    file,
                    "entity,{},{},\"{}\"",
                    entity,
                    embedding.dimensions,
                    values.join(",")
                )?;
            }
        }

        // Export relation embeddings
        for relation in model.get_relations() {
            if let Ok(embedding) = model.get_relation_embedding(&relation) {
                let values: Vec<String> = embedding.values.iter().map(|x| x.to_string()).collect();
                writeln!(
                    file,
                    "relation,{},{},\"{}\"",
                    relation,
                    embedding.dimensions,
                    values.join(",")
                )?;
            }
        }

        info!("Exported model embeddings to CSV: {}", output_path);
        Ok(())
    }

    /// Export to ONNX format.
    ///
    /// Requires the `onnx-export` Cargo feature.  Without it the call returns a
    /// [`PersistenceError::UnsupportedFormat`] error so callers get a clear,
    /// actionable message rather than a silent no-op.
    ///
    /// # Feature gate
    ///
    /// Enable the `onnx-export` feature in your `Cargo.toml`:
    /// ```toml
    /// oxirs-embed = { version = "*", features = ["onnx-export"] }
    /// ```
    pub fn export_to_onnx(
        _model: &dyn EmbeddingModel,
        _output_path: &str,
    ) -> Result<(), PersistenceError> {
        #[cfg(feature = "onnx-export")]
        {
            // Feature gate exists for future use; a pure-Rust ONNX writer
            // is not yet available in the COOLJAPAN ecosystem.
            Err(PersistenceError::NotImplemented(
                "ONNX writer not yet available — the 'onnx-export' feature is reserved \
                for a future pure-Rust ONNX serialiser"
                    .to_string(),
            ))
        }
        #[cfg(not(feature = "onnx-export"))]
        Err(PersistenceError::UnsupportedFormat(
            "ONNX export requires the 'onnx-export' feature flag. \
            Enable it in your Cargo.toml: oxirs-embed = { features = [\"onnx-export\"] }"
                .to_string(),
        ))
    }

    /// Export to TensorFlow SavedModel format.
    ///
    /// Requires the `tf-export` Cargo feature.  Without it the call returns a
    /// [`PersistenceError::UnsupportedFormat`] error.
    ///
    /// # Feature gate
    ///
    /// Enable the `tf-export` feature in your `Cargo.toml`:
    /// ```toml
    /// oxirs-embed = { version = "*", features = ["tf-export"] }
    /// ```
    pub fn export_to_tensorflow(
        _model: &dyn EmbeddingModel,
        _output_path: &str,
    ) -> Result<(), PersistenceError> {
        #[cfg(feature = "tf-export")]
        {
            // Feature gate exists for future use; TensorFlow SavedModel export
            // depends on a pure-Rust protobuf writer for the SavedModel format.
            Err(PersistenceError::NotImplemented(
                "TensorFlow SavedModel writer not yet available — the 'tf-export' feature is \
                reserved for a future pure-Rust TensorFlow serialiser"
                    .to_string(),
            ))
        }
        #[cfg(not(feature = "tf-export"))]
        Err(PersistenceError::UnsupportedFormat(
            "TensorFlow export requires the 'tf-export' feature flag. \
            Enable it in your Cargo.toml: oxirs-embed = { features = [\"tf-export\"] }"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::TransE;
    use tempfile::TempDir;

    #[test]
    fn test_model_repository() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut repo = ModelRepository::new(temp_dir.path())?;

        assert_eq!(repo.list_models().len(), 0);

        // Create a dummy metadata file
        let model_dir = temp_dir.path().join("test_model");
        fs::create_dir_all(&model_dir)?;

        let metadata = ModelMetadata::default();
        let metadata_content = serde_json::to_string_pretty(&metadata)?;
        fs::write(model_dir.join("metadata.json"), metadata_content)?;

        // Rescan
        repo.scan_models()?;
        assert_eq!(repo.list_models().len(), 1);

        Ok(())
    }

    #[test]
    fn test_checkpoint_manager() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let checkpoint_manager = CheckpointManager::new(temp_dir.path(), 3)?;

        let checkpoints = checkpoint_manager.list_checkpoints()?;
        assert_eq!(checkpoints.len(), 0);

        Ok(())
    }

    /// Verify that save_model persists the model type and load_model reads it back,
    /// dispatching to the correct concrete type.
    #[test]
    fn test_save_and_load_model_type_persistence() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut repo = ModelRepository::new(temp_dir.path())?;

        // Build a minimal TransE model (untrained is fine for this test)
        let model = TransE::new(ModelConfig::default());

        // Save it — this writes model.bin (stub), model_type.json, and metadata.json
        repo.save_model(&model, "transe_test", Some("unit test".to_string()))?;

        // Verify model_type.json was created with the correct value
        let model_dir = temp_dir.path().join("transe_test");
        let type_file = model_dir.join("model_type.json");
        assert!(
            type_file.exists(),
            "model_type.json should have been created"
        );

        let raw = fs::read_to_string(&type_file)?;
        let stored_type: String = serde_json::from_str(&raw)?;
        assert_eq!(stored_type, "TransE");

        // Load the model back — should succeed and return a TransE instance
        let loaded = repo.load_model("transe_test")?;
        assert_eq!(loaded.model_type(), "TransE");

        Ok(())
    }

    /// Verify that load_model returns an error for an unknown/missing model
    #[test]
    fn test_load_model_not_found() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let repo = ModelRepository::new(temp_dir.path())?;

        let result = repo.load_model("nonexistent");
        assert!(result.is_err());
        let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("nonexistent") || msg.contains("not found"));

        Ok(())
    }

    /// Verify that load_model_info picks up model_type from model_type.json
    #[test]
    fn test_model_info_type_from_file() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut repo = ModelRepository::new(temp_dir.path())?;

        // Manually write a model directory with metadata and model_type
        let model_dir = temp_dir.path().join("manual_model");
        fs::create_dir_all(&model_dir)?;

        let metadata = ModelMetadata::default();
        fs::write(
            model_dir.join("metadata.json"),
            serde_json::to_string_pretty(&metadata)?,
        )?;
        fs::write(
            model_dir.join("model_type.json"),
            serde_json::to_string("DistMult")?,
        )?;

        // Rescan to pick up the manually placed model
        repo.scan_models()?;

        let info = repo
            .get_model_info("manual_model")
            .ok_or_else(|| anyhow!("model info should be present"))?;
        assert_eq!(info.model_type, "DistMult");

        Ok(())
    }

    /// Verify that load_model returns an error for an unsupported model type
    #[test]
    fn test_load_model_unsupported_type() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut repo = ModelRepository::new(temp_dir.path())?;

        // Manually create a model directory with an unsupported type
        let model_dir = temp_dir.path().join("exotic_model");
        fs::create_dir_all(&model_dir)?;

        let metadata = ModelMetadata::default();
        fs::write(
            model_dir.join("metadata.json"),
            serde_json::to_string_pretty(&metadata)?,
        )?;
        fs::write(
            model_dir.join("model_type.json"),
            serde_json::to_string("SomeFutureModel")?,
        )?;

        repo.scan_models()?;

        let result = repo.load_model("exotic_model");
        assert!(result.is_err());
        let msg = result.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(
            msg.contains("unsupported") || msg.contains("SomeFutureModel"),
            "error message should mention the unsupported type, got: {msg}"
        );

        Ok(())
    }
}
