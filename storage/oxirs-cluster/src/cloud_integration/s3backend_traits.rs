//! # S3Backend - Trait Implementations
//!
//! This module contains trait implementations for `S3Backend`.
//!
//! ## Implemented Traits
//!
//! - `CloudStorageProvider`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::functions::CloudStorageProvider;
use super::types::{
    CloudError, CloudProvider, HealthStatus, ObjectMetadata, S3Backend, StorageOperationResult,
    StorageTier,
};

#[async_trait::async_trait]
impl CloudStorageProvider for S3Backend {
    fn provider(&self) -> CloudProvider {
        CloudProvider::AWS
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
            etag: format!("{:x}", md5_hash(data)),
            custom_metadata: HashMap::new(),
        };
        client.metadata.insert(key.to_string(), metadata.clone());
        let duration = start.elapsed().as_millis() as u64;
        self.metrics.uploads.inc();
        for _ in 0..data.len() {
            self.metrics.upload_bytes.inc();
        }
        self.metrics.latency_sum.set(duration as f64);
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
                for _ in 0..data.len() {
                    self.metrics.download_bytes.inc();
                }
                self.metrics.latency_sum.set(duration as f64);
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
        Ok(format!("upload-{}-{}", key, uuid::Uuid::new_v4()))
    }
    async fn upload_part(
        &self,
        _key: &str,
        _upload_id: &str,
        part_number: u32,
        data: &[u8],
    ) -> Result<String, CloudError> {
        Ok(format!("{:x}-{}", md5_hash(data), part_number))
    }
    async fn complete_multipart(
        &self,
        key: &str,
        _upload_id: &str,
        parts: &[(u32, String)],
    ) -> Result<StorageOperationResult, CloudError> {
        let start = Instant::now();
        let total_parts = parts.len();
        let duration = start.elapsed().as_millis() as u64;
        Ok(StorageOperationResult {
            success: true,
            duration_ms: duration,
            bytes_transferred: 0,
            error: None,
            etag: Some(format!("{:x}-{}", md5_hash(key.as_bytes()), total_parts)),
        })
    }
    async fn health_check(&self) -> Result<HealthStatus, CloudError> {
        let start = Instant::now();
        let test_key = "__health_check__";
        let test_data = b"health";
        let mut client = self.client.write().await;
        client
            .objects
            .insert(test_key.to_string(), test_data.to_vec());
        client.objects.remove(test_key);
        let latency = start.elapsed().as_millis() as u64;
        Ok(HealthStatus {
            healthy: true,
            latency_ms: latency,
            error_rate: 0.0,
            last_check: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after UNIX_EPOCH")
                .as_secs(),
            message: "S3 backend healthy".to_string(),
        })
    }
}
