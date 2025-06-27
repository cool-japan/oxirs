//! Model persistence and serialization utilities

use crate::{EmbeddingModel, ModelConfig, ModelStats};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::{debug, info};

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
        let model_path = format!("{}/{}", self.base_path, model_name);
        let metadata_path = format!("{}/metadata.json", model_path);

        if !Path::new(&metadata_path).exists() {
            return Err(anyhow!("Model metadata not found: {}", metadata_path));
        }

        let metadata_content = fs::read_to_string(metadata_path)?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_content)?;

        Ok(ModelInfo {
            id: model_name.to_string(),
            name: model_name.to_string(),
            model_type: "unknown".to_string(), // Would be loaded from actual model
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
        let model_path = format!("{}/{}", self.base_path, name);
        fs::create_dir_all(&model_path)?;

        // Save model data
        let model_file = format!("{}/model.bin", model_path);
        model.save(&model_file)?;

        // Save metadata
        let metadata = ModelMetadata {
            description,
            trained_at: Some(chrono::Utc::now()),
            ..Default::default()
        };

        let metadata_file = format!("{}/metadata.json", model_path);
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

        let model_file = format!("{}/model.bin", model_info.path);

        // This is a placeholder - in a real implementation, we'd need to:
        // 1. Determine the model type from metadata
        // 2. Create the appropriate model instance
        // 3. Load the model data

        // For now, return an error as this requires model-specific deserialization
        Err(anyhow!("Model loading not yet implemented"))
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
        let checkpoint_name = format!("checkpoint_epoch_{}_loss_{:.6}.bin", epoch, loss);
        let checkpoint_path = format!("{}/{}", self.checkpoint_dir, checkpoint_name);

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

    /// Export to ONNX format (placeholder)
    pub fn export_to_onnx(_model: &dyn EmbeddingModel, _output_path: &str) -> Result<()> {
        // This would require implementing ONNX export
        Err(anyhow!("ONNX export not yet implemented"))
    }

    /// Export to TensorFlow SavedModel format (placeholder)
    pub fn export_to_tensorflow(_model: &dyn EmbeddingModel, _output_path: &str) -> Result<()> {
        // This would require implementing TensorFlow export
        Err(anyhow!("TensorFlow export not yet implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
