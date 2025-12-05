//! Cloud Storage Integration for SAMM Models
//!
//! This module provides a flexible trait-based cloud storage abstraction for SAMM models.
//! Users can implement their own cloud storage backends or use pre-built integrations.
//!
//! # Features
//!
//! - **Trait-Based Design**: Implement `CloudStorageBackend` for any storage provider
//! - **Model Caching**: Optional local caching of frequently accessed models
//! - **Batch Operations**: Upload/download multiple models efficiently
//! - **Async Support**: Full async/await support for I/O operations
//!
//! # Examples
//!
//! ```rust,no_run
//! use oxirs_samm::cloud_storage::{CloudModelStorage, MemoryBackend};
//! use oxirs_samm::metamodel::Aspect;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create an in-memory storage backend for testing
//! let backend = MemoryBackend::new();
//! let mut storage = CloudModelStorage::new(Box::new(backend));
//!
//! // Upload a model
//! let aspect = Aspect::new("urn:samm:org.example:1.0.0#Vehicle".to_string());
//! storage.upload_model("models/vehicle.ttl", &aspect).await?;
//!
//! // Download a model
//! let downloaded = storage.download_model("models/vehicle.ttl").await?;
//!
//! // List all models
//! let models = storage.list_models("models/").await?;
//! println!("Found {} models", models.len());
//! # Ok(())
//! # }
//! ```
//!
//! # Implementing Custom Backends
//!
//! ```rust
//! use oxirs_samm::cloud_storage::CloudStorageBackend;
//! use async_trait::async_trait;
//!
//! struct MyS3Backend {
//!     // Your AWS S3 client
//! }
//!
//! #[async_trait]
//! impl CloudStorageBackend for MyS3Backend {
//!     async fn upload(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
//!         // Upload to S3
//!         Ok(())
//!     }
//!
//!     async fn download(&self, key: &str) -> std::result::Result<Vec<u8>, String> {
//!         // Download from S3
//!         Ok(vec![])
//!     }
//!
//!     async fn exists(&self, key: &str) -> std::result::Result<bool, String> {
//!         // Check if object exists in S3
//!         Ok(false)
//!     }
//!
//!     async fn delete(&self, key: &str) -> std::result::Result<(), String> {
//!         // Delete from S3
//!         Ok(())
//!     }
//!
//!     async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String> {
//!         // List objects in S3
//!         Ok(vec![])
//!     }
//! }
//! ```

use crate::error::{Result, SammError};
use crate::metamodel::Aspect;
use crate::parser::parse_aspect_from_string;
use crate::serializer::serialize_aspect_to_string;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tracing::{debug, error, info};

/// Trait for cloud storage backends
///
/// Implement this trait to add support for different cloud storage providers
/// (AWS S3, Google Cloud Storage, Azure Blob Storage, etc.)
#[async_trait]
pub trait CloudStorageBackend: Send + Sync {
    /// Upload data to cloud storage
    async fn upload(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String>;

    /// Download data from cloud storage
    async fn download(&self, key: &str) -> std::result::Result<Vec<u8>, String>;

    /// Check if an object exists
    async fn exists(&self, key: &str) -> std::result::Result<bool, String>;

    /// Delete an object
    async fn delete(&self, key: &str) -> std::result::Result<(), String>;

    /// List objects with a given prefix
    async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String>;

    /// Get object metadata (optional, returns empty metadata by default)
    async fn get_metadata(&self, key: &str) -> std::result::Result<ObjectMetadata, String> {
        Ok(ObjectMetadata {
            key: key.to_string(),
            size: 0,
            last_modified: None,
        })
    }
}

/// Cloud storage client for SAMM models
pub struct CloudModelStorage {
    backend: Box<dyn CloudStorageBackend>,
    cache: Option<Arc<Mutex<ModelCache>>>,
}

/// Local cache for cloud models
#[derive(Debug)]
struct ModelCache {
    models: HashMap<String, (Aspect, SystemTime)>,
    ttl: Duration,
}

impl ModelCache {
    fn new(ttl: Duration) -> Self {
        Self {
            models: HashMap::new(),
            ttl,
        }
    }

    fn get(&mut self, key: &str) -> Option<Aspect> {
        if let Some((model, timestamp)) = self.models.get(key) {
            if timestamp.elapsed().unwrap_or(Duration::MAX) < self.ttl {
                debug!("Cache hit for model: {}", key);
                return Some(model.clone());
            } else {
                debug!("Cache expired for model: {}", key);
                self.models.remove(key);
            }
        }
        None
    }

    fn put(&mut self, key: String, model: Aspect) {
        self.models.insert(key, (model, SystemTime::now()));
    }

    fn clear(&mut self) {
        self.models.clear();
    }
}

impl CloudModelStorage {
    /// Create a new cloud model storage client
    ///
    /// # Arguments
    ///
    /// * `backend` - Cloud storage backend implementation
    ///
    /// # Example
    ///
    /// ```rust
    /// # use oxirs_samm::cloud_storage::{CloudModelStorage, MemoryBackend};
    /// let backend = MemoryBackend::new();
    /// let storage = CloudModelStorage::new(Box::new(backend));
    /// ```
    pub fn new(backend: Box<dyn CloudStorageBackend>) -> Self {
        info!("Initialized cloud model storage");
        Self {
            backend,
            cache: Some(Arc::new(Mutex::new(ModelCache::new(Duration::from_secs(
                3600,
            ))))),
        }
    }

    /// Create storage without caching
    pub fn new_without_cache(backend: Box<dyn CloudStorageBackend>) -> Self {
        info!("Initialized cloud model storage (no cache)");
        Self {
            backend,
            cache: None,
        }
    }

    /// Upload a SAMM model to cloud storage
    pub async fn upload_model(&mut self, key: &str, aspect: &Aspect) -> Result<()> {
        info!("Uploading model to cloud: {}", key);

        // Serialize aspect to Turtle format
        let ttl_content = serialize_aspect_to_string(aspect)?;

        // Upload to cloud
        self.backend
            .upload(key, ttl_content.into_bytes())
            .await
            .map_err(|e| SammError::cloud_error(format!("Upload failed: {}", e)))?;

        // Update cache
        if let Some(cache) = &self.cache {
            if let Ok(mut cache_guard) = cache.lock() {
                cache_guard.put(key.to_string(), aspect.clone());
            }
        }

        info!("Successfully uploaded model: {}", key);
        Ok(())
    }

    /// Download a SAMM model from cloud storage
    pub async fn download_model(&mut self, key: &str) -> Result<Aspect> {
        // Check cache first
        if let Some(cache) = &self.cache {
            if let Ok(mut cache_guard) = cache.lock() {
                if let Some(model) = cache_guard.get(key) {
                    return Ok(model);
                }
            }
        }

        info!("Downloading model from cloud: {}", key);

        // Download from cloud
        let data = self
            .backend
            .download(key)
            .await
            .map_err(|e| SammError::cloud_error(format!("Download failed: {}", e)))?;

        // Parse the Turtle content
        let ttl_content = String::from_utf8(data)
            .map_err(|e| SammError::ParseError(format!("Invalid UTF-8: {}", e)))?;

        // Use a dummy base URI for parsing
        let aspect = parse_aspect_from_string(&ttl_content, "urn:samm:org.eclipse.esmf").await?;

        // Update cache
        if let Some(cache) = &self.cache {
            if let Ok(mut cache_guard) = cache.lock() {
                cache_guard.put(key.to_string(), aspect.clone());
            }
        }

        info!("Successfully downloaded model: {}", key);
        Ok(aspect)
    }

    /// Check if a model exists in cloud storage
    pub async fn model_exists(&self, key: &str) -> Result<bool> {
        self.backend
            .exists(key)
            .await
            .map_err(|e| SammError::cloud_error(format!("Existence check failed: {}", e)))
    }

    /// Delete a model from cloud storage
    pub async fn delete_model(&mut self, key: &str) -> Result<()> {
        info!("Deleting model from cloud: {}", key);

        self.backend
            .delete(key)
            .await
            .map_err(|e| SammError::cloud_error(format!("Delete failed: {}", e)))?;

        // Remove from cache
        if let Some(cache) = &self.cache {
            if let Ok(mut cache_guard) = cache.lock() {
                cache_guard.models.remove(key);
            }
        }

        info!("Successfully deleted model: {}", key);
        Ok(())
    }

    /// List all models in a directory/prefix
    pub async fn list_models(&self, prefix: &str) -> Result<Vec<ModelInfo>> {
        info!("Listing models with prefix: {}", prefix);

        let keys = self
            .backend
            .list(prefix)
            .await
            .map_err(|e| SammError::cloud_error(format!("List failed: {}", e)))?;

        let mut models = Vec::new();
        for key in keys {
            if key.ends_with(".ttl") {
                if let Ok(metadata) = self.backend.get_metadata(&key).await {
                    models.push(ModelInfo {
                        key: metadata.key,
                        size: metadata.size,
                        last_modified: metadata.last_modified,
                    });
                }
            }
        }

        Ok(models)
    }

    /// Upload multiple models in batch
    pub async fn upload_models_batch(
        &mut self,
        models: Vec<(String, Aspect)>,
    ) -> Result<BatchResult> {
        info!("Uploading {} models in batch", models.len());

        let mut successful = 0;
        let mut failed = Vec::new();

        for (key, aspect) in models {
            match self.upload_model(&key, &aspect).await {
                Ok(_) => successful += 1,
                Err(e) => {
                    error!("Failed to upload {}: {}", key, e);
                    failed.push((key, e.to_string()));
                }
            }
        }

        let failed_count = failed.len();

        info!(
            "Batch upload complete: {} successful, {} failed",
            successful, failed_count
        );

        Ok(BatchResult {
            successful,
            failed,
            total: successful + failed_count,
        })
    }

    /// Clear the local cache
    pub fn clear_cache(&mut self) {
        if let Some(cache) = &self.cache {
            if let Ok(mut cache_guard) = cache.lock() {
                cache_guard.clear();
                info!("Cache cleared");
            }
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().and_then(|cache| {
            cache.lock().ok().map(|guard| CacheStats {
                entries: guard.models.len(),
                ttl_seconds: guard.ttl.as_secs(),
            })
        })
    }
}

/// In-memory storage backend for testing
pub struct MemoryBackend {
    storage: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl MemoryBackend {
    /// Create a new in-memory backend
    pub fn new() -> Self {
        Self {
            storage: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for MemoryBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CloudStorageBackend for MemoryBackend {
    async fn upload(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
        let mut storage = self.storage.lock().unwrap();
        storage.insert(key.to_string(), data);
        Ok(())
    }

    async fn download(&self, key: &str) -> std::result::Result<Vec<u8>, String> {
        let storage = self.storage.lock().unwrap();
        storage
            .get(key)
            .cloned()
            .ok_or_else(|| format!("Key not found: {}", key))
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, String> {
        let storage = self.storage.lock().unwrap();
        Ok(storage.contains_key(key))
    }

    async fn delete(&self, key: &str) -> std::result::Result<(), String> {
        let mut storage = self.storage.lock().unwrap();
        storage.remove(key);
        Ok(())
    }

    async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String> {
        let storage = self.storage.lock().unwrap();
        Ok(storage
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }

    async fn get_metadata(&self, key: &str) -> std::result::Result<ObjectMetadata, String> {
        let storage = self.storage.lock().unwrap();
        storage
            .get(key)
            .map(|data| ObjectMetadata {
                key: key.to_string(),
                size: data.len(),
                last_modified: Some(SystemTime::now()),
            })
            .ok_or_else(|| format!("Key not found: {}", key))
    }
}

/// Object metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    /// Object key
    pub key: String,
    /// Size in bytes
    pub size: usize,
    /// Last modification time
    pub last_modified: Option<SystemTime>,
}

/// Information about a cloud-stored model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Cloud storage key
    pub key: String,
    /// File size in bytes
    pub size: usize,
    /// Last modification timestamp
    pub last_modified: Option<SystemTime>,
}

/// Batch operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Number of successful operations
    pub successful: usize,
    /// Failed operations with error messages
    pub failed: Vec<(String, String)>,
    /// Total operations attempted
    pub total: usize,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Number of cached entries
    pub entries: usize,
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metamodel::ModelElement;

    #[test]
    fn test_model_info_creation() {
        let info = ModelInfo {
            key: "models/test.ttl".to_string(),
            size: 1024,
            last_modified: Some(SystemTime::now()),
        };

        assert_eq!(info.key, "models/test.ttl");
        assert_eq!(info.size, 1024);
        assert!(info.last_modified.is_some());
    }

    #[test]
    fn test_batch_result() {
        let result = BatchResult {
            successful: 5,
            failed: vec![("model1.ttl".to_string(), "Error".to_string())],
            total: 6,
        };

        assert_eq!(result.successful, 5);
        assert_eq!(result.failed.len(), 1);
        assert_eq!(result.total, 6);
    }

    #[test]
    fn test_cache_stats() {
        let stats = CacheStats {
            entries: 10,
            ttl_seconds: 3600,
        };

        assert_eq!(stats.entries, 10);
        assert_eq!(stats.ttl_seconds, 3600);
    }

    #[tokio::test]
    async fn test_memory_backend() {
        let backend = MemoryBackend::new();

        // Test upload
        let data = b"test data".to_vec();
        backend.upload("test.txt", data.clone()).await.unwrap();

        // Test exists
        assert!(backend.exists("test.txt").await.unwrap());
        assert!(!backend.exists("nonexistent.txt").await.unwrap());

        // Test download
        let downloaded = backend.download("test.txt").await.unwrap();
        assert_eq!(downloaded, data);

        // Test list
        backend.upload("dir/file1.txt", vec![]).await.unwrap();
        backend.upload("dir/file2.txt", vec![]).await.unwrap();
        let files = backend.list("dir/").await.unwrap();
        assert_eq!(files.len(), 2);

        // Test delete
        backend.delete("test.txt").await.unwrap();
        assert!(!backend.exists("test.txt").await.unwrap());
    }

    #[tokio::test]
    async fn test_cloud_model_storage() {
        let backend = MemoryBackend::new();
        let mut storage = CloudModelStorage::new(Box::new(backend));

        // Create a test aspect
        let aspect = Aspect::new("urn:samm:org.test:1.0.0#TestAspect".to_string());

        // Test upload
        storage
            .upload_model("models/test.ttl", &aspect)
            .await
            .unwrap();

        // Test exists
        assert!(storage.model_exists("models/test.ttl").await.unwrap());

        // Test download
        let downloaded = storage.download_model("models/test.ttl").await.unwrap();
        assert_eq!(downloaded.name(), aspect.name());

        // Test list
        let models = storage.list_models("models/").await.unwrap();
        assert_eq!(models.len(), 1);

        // Test delete
        storage.delete_model("models/test.ttl").await.unwrap();
        assert!(!storage.model_exists("models/test.ttl").await.unwrap());
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let backend = MemoryBackend::new();
        let mut storage = CloudModelStorage::new(Box::new(backend));

        let aspect = Aspect::new("urn:samm:org.test:1.0.0#CachedAspect".to_string());

        // Upload model
        storage
            .upload_model("cached/model.ttl", &aspect)
            .await
            .unwrap();

        // First download (from backend)
        let _first = storage.download_model("cached/model.ttl").await.unwrap();

        // Check cache stats
        let stats = storage.cache_stats().unwrap();
        assert_eq!(stats.entries, 1);

        // Second download (from cache)
        let _second = storage.download_model("cached/model.ttl").await.unwrap();

        // Clear cache
        storage.clear_cache();
        let stats_after_clear = storage.cache_stats().unwrap();
        assert_eq!(stats_after_clear.entries, 0);
    }
}
