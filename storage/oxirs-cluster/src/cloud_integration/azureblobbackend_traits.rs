//! # AzureBlobBackend - Trait Implementations
//!
//! This module contains trait implementations for `AzureBlobBackend`.
//!
//! ## Implemented Traits
//!
//! - `CloudStorageProvider`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::functions::{md5_hash, CloudStorageProvider};
use super::types::{
    AzureBlobBackend, CloudError, CloudProvider, HealthStatus, ObjectMetadata,
    StorageOperationResult, StorageTier,
};

#[async_trait::async_trait]
impl CloudStorageProvider for AzureBlobBackend {
    fn provider(&self) -> CloudProvider {
        CloudProvider::Azure
    }
    async fn upload(
        &self,
        key: &str,
        data: &[u8],
        tier: StorageTier,
    ) -> Result<StorageOperationResult, CloudError> {
        let start = Instant::now();
        let mut client = self.client.write().await;
        client.objects.insert(key.to_string(), data.to_vec());
        let metadata = ObjectMetadata {
            key: key.to_string(),
            size: data.len() as u64,
            last_modified: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX_EPOCH")
                .as_secs(),
            content_type: "application/octet-stream".to_string(),
            storage_tier: tier,
            etag: format!("0x{:X}", md5_hash(data)),
            custom_metadata: HashMap::new(),
        };
        client.metadata.insert(key.to_string(), metadata.clone());
        let duration = start.elapsed().as_millis() as u64;
        self.metrics.uploads.inc();
        Ok(StorageOperationResult {
            success: true,
            duration_ms: duration,
            bytes_transferred: data.len() as u64,
            error: None,
            etag: Some(metadata.etag),
        })
    }
    async fn download(&self, key: &str) -> Result<(Vec<u8>, StorageOperationResult), CloudError> {
        let start = Instant::now();
        let client = self.client.read().await;
        match client.objects.get(key) {
            Some(data) => {
                let duration = start.elapsed().as_millis() as u64;
                self.metrics.downloads.inc();
                Ok((
                    data.clone(),
                    StorageOperationResult {
                        success: true,
                        duration_ms: duration,
                        bytes_transferred: data.len() as u64,
                        error: None,
                        etag: client.metadata.get(key).map(|m| m.etag.clone()),
                    },
                ))
            }
            None => {
                self.metrics.errors.inc();
                Err(CloudError::ObjectNotFound(key.to_string()))
            }
        }
    }
    async fn delete(&self, key: &str) -> Result<StorageOperationResult, CloudError> {
        let start = Instant::now();
        let mut client = self.client.write().await;
        if client.objects.remove(key).is_some() {
            client.metadata.remove(key);
            let duration = start.elapsed().as_millis() as u64;
            Ok(StorageOperationResult {
                success: true,
                duration_ms: duration,
                bytes_transferred: 0,
                error: None,
                etag: None,
            })
        } else {
            Err(CloudError::ObjectNotFound(key.to_string()))
        }
    }
    async fn list(&self, prefix: &str) -> Result<Vec<String>, CloudError> {
        let client = self.client.read().await;
        let keys: Vec<String> = client
            .objects
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        Ok(keys)
    }
    async fn exists(&self, key: &str) -> Result<bool, CloudError> {
        let client = self.client.read().await;
        Ok(client.objects.contains_key(key))
    }
    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata, CloudError> {
        let client = self.client.read().await;
        client
            .metadata
            .get(key)
            .cloned()
            .ok_or_else(|| CloudError::ObjectNotFound(key.to_string()))
    }
    async fn initiate_multipart(&self, key: &str) -> Result<String, CloudError> {
        Ok(format!("azure-upload-{}-{}", key, uuid::Uuid::new_v4()))
    }
    async fn upload_part(
        &self,
        _key: &str,
        _upload_id: &str,
        part_number: u32,
        data: &[u8],
    ) -> Result<String, CloudError> {
        Ok(format!("azure-{:x}-{}", md5_hash(data), part_number))
    }
    async fn complete_multipart(
        &self,
        key: &str,
        _upload_id: &str,
        parts: &[(u32, String)],
    ) -> Result<StorageOperationResult, CloudError> {
        let start = Instant::now();
        let duration = start.elapsed().as_millis() as u64;
        Ok(StorageOperationResult {
            success: true,
            duration_ms: duration,
            bytes_transferred: 0,
            error: None,
            etag: Some(format!(
                "azure-{:x}-{}",
                md5_hash(key.as_bytes()),
                parts.len()
            )),
        })
    }
    async fn health_check(&self) -> Result<HealthStatus, CloudError> {
        let start = Instant::now();
        let latency = start.elapsed().as_millis() as u64;
        Ok(HealthStatus {
            healthy: true,
            latency_ms: latency,
            error_rate: 0.0,
            last_check: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX_EPOCH")
                .as_secs(),
            message: "Azure Blob backend healthy".to_string(),
        })
    }
}
