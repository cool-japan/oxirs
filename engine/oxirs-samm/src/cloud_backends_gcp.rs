//! Google Cloud Storage backend.

use crate::cloud_backends_common::url_encode;
use crate::cloud_storage::CloudStorageBackend;
use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use tracing::{debug, warn};

/// Configuration for the Google Cloud Storage backend.
#[derive(Debug, Clone)]
pub struct GcsConfig {
    /// GCS bucket name.
    pub bucket: String,
    /// Service account JSON key (as a raw JSON string). Mutually exclusive with `access_token`.
    pub service_account_key: Option<String>,
    /// Pre-obtained OAuth2 Bearer token. Mutually exclusive with `service_account_key`.
    pub access_token: Option<String>,
    /// Optional object-key prefix.
    pub path_prefix: String,
}

impl GcsConfig {
    /// Create a new `GcsConfig` with a Bearer token (simplest auth method).
    pub fn with_access_token(bucket: impl Into<String>, access_token: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            service_account_key: None,
            access_token: Some(access_token.into()),
            path_prefix: String::new(),
        }
    }

    /// Create a new `GcsConfig` with a service-account JSON key.
    pub fn with_service_account(bucket: impl Into<String>, key_json: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            service_account_key: Some(key_json.into()),
            access_token: None,
            path_prefix: String::new(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.bucket.is_empty() {
            return Err("GcsConfig.bucket must not be empty".to_string());
        }
        if self.access_token.is_none() && self.service_account_key.is_none() {
            return Err(
                "GcsConfig requires either an access_token or service_account_key".to_string(),
            );
        }
        // `service_account_key` alone cannot yet authenticate: the JWT/OAuth2
        // service-account token-exchange flow (RS256-signing a claim set with
        // the private key from the JSON key file, then POSTing to Google's
        // token endpoint) is not implemented in this crate — there is no
        // RSA-signing dependency wired in. Rather than constructing a backend
        // that will only discover this at first-request time (a much later,
        // harder-to-diagnose failure), reject the configuration up front.
        if self.access_token.is_none() && self.service_account_key.is_some() {
            return Err(
                "GcsConfig::with_service_account() cannot authenticate on its own: automatic \
                 service-account JWT/OAuth2 token exchange is not yet implemented in oxirs-samm. \
                 Exchange the service-account key for a Bearer token yourself (e.g. via \
                 `gcloud auth print-access-token` or the Google Cloud SDK) and construct the \
                 config with `GcsConfig::with_access_token` instead."
                    .to_string(),
            );
        }
        Ok(())
    }

    fn bearer_token(&self) -> Option<String> {
        if let Some(ref tok) = self.access_token {
            return Some(tok.clone());
        }
        // Unreachable in practice: `validate()` (always called by
        // `GcsBackend::new`) rejects service-account-only configs before a
        // backend can be constructed. Kept as a defensive fallback in case a
        // `GcsConfig` is ever used without going through `GcsBackend::new`.
        warn!("GcsConfig: service_account_key supplied but automatic token exchange is not yet implemented; supply access_token instead");
        None
    }

    fn full_key(&self, key: &str) -> String {
        if self.path_prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}{}", self.path_prefix, key)
        }
    }
}

const GCS_BASE_URL: &str = "https://storage.googleapis.com/storage/v1";
const GCS_UPLOAD_URL: &str = "https://storage.googleapis.com/upload/storage/v1";

/// Google Cloud Storage backend.
///
/// Authenticates via a Bearer token (OAuth2 access token).
#[derive(Debug)]
pub struct GcsBackend {
    config: GcsConfig,
    client: Client,
}

impl GcsBackend {
    /// Create a new `GcsBackend`.
    pub fn new(config: GcsConfig) -> Result<Self, String> {
        config.validate()?;
        let client = crate::cloud_backends_common::build_tls_client()?;
        Ok(Self { config, client })
    }

    fn auth_header(&self) -> Result<String, String> {
        self.config
            .bearer_token()
            .map(|t| format!("Bearer {t}"))
            .ok_or_else(|| "No GCS Bearer token available".to_string())
    }

    async fn gcs_upload(&self, key: &str, data: Vec<u8>) -> Result<(), String> {
        let auth = self.auth_header()?;
        let upload_url = format!(
            "{}/b/{}/o?uploadType=media&name={}",
            GCS_UPLOAD_URL,
            self.config.bucket,
            url_encode(key)
        );
        debug!("GCS upload to {}", upload_url);

        let response = self
            .client
            .post(&upload_url)
            .header("Authorization", auth)
            .header("Content-Type", "application/octet-stream")
            .body(data)
            .send()
            .await
            .map_err(|e| format!("GCS upload failed: {e}"))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("GCS upload returned {status}: {body}"))
        }
    }

    async fn gcs_download(&self, key: &str) -> Result<Vec<u8>, String> {
        let auth = self.auth_header()?;
        let url = format!(
            "{}/b/{}/o/{}?alt=media",
            GCS_BASE_URL,
            self.config.bucket,
            url_encode(key)
        );
        debug!("GCS download from {}", url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", auth)
            .send()
            .await
            .map_err(|e| format!("GCS download failed: {e}"))?;

        if response.status().is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| format!("Failed to read GCS download body: {e}"))?;
            Ok(bytes.to_vec())
        } else if response.status() == StatusCode::NOT_FOUND {
            Err(format!("GCS object not found: {key}"))
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("GCS download returned {status}: {body}"))
        }
    }

    async fn gcs_exists(&self, key: &str) -> Result<bool, String> {
        let auth = self.auth_header()?;
        let url = format!(
            "{}/b/{}/o/{}",
            GCS_BASE_URL,
            self.config.bucket,
            url_encode(key)
        );
        debug!("GCS exists check {}", url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", auth)
            .send()
            .await
            .map_err(|e| format!("GCS exists check failed: {e}"))?;

        match response.status() {
            StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            other => Err(format!("GCS exists returned unexpected status: {other}")),
        }
    }

    async fn gcs_delete(&self, key: &str) -> Result<(), String> {
        let auth = self.auth_header()?;
        let url = format!(
            "{}/b/{}/o/{}",
            GCS_BASE_URL,
            self.config.bucket,
            url_encode(key)
        );
        debug!("GCS delete {}", url);

        let response = self
            .client
            .delete(&url)
            .header("Authorization", auth)
            .send()
            .await
            .map_err(|e| format!("GCS delete failed: {e}"))?;

        if response.status().is_success() || response.status() == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("GCS delete returned {status}: {body}"))
        }
    }

    async fn gcs_list(&self, prefix: &str) -> Result<Vec<String>, String> {
        let auth = self.auth_header()?;
        let url = if prefix.is_empty() {
            format!("{}/b/{}/o", GCS_BASE_URL, self.config.bucket)
        } else {
            format!(
                "{}/b/{}/o?prefix={}",
                GCS_BASE_URL,
                self.config.bucket,
                url_encode(prefix)
            )
        };
        debug!("GCS list {} prefix={}", url, prefix);

        let response = self
            .client
            .get(&url)
            .header("Authorization", auth)
            .send()
            .await
            .map_err(|e| format!("GCS list failed: {e}"))?;

        if response.status().is_success() {
            let body: serde_json::Value = response
                .json()
                .await
                .map_err(|e| format!("Failed to parse GCS list response: {e}"))?;
            let keys = body
                .get("items")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| item.get("name")?.as_str().map(String::from))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Ok(keys)
        } else if response.status() == StatusCode::NOT_FOUND {
            Ok(vec![])
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("GCS list returned {status}: {body}"))
        }
    }
}

#[async_trait]
impl CloudStorageBackend for GcsBackend {
    async fn upload(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
        let full_key = self.config.full_key(key);
        self.gcs_upload(&full_key, data).await
    }

    async fn download(&self, key: &str) -> std::result::Result<Vec<u8>, String> {
        let full_key = self.config.full_key(key);
        self.gcs_download(&full_key).await
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, String> {
        let full_key = self.config.full_key(key);
        self.gcs_exists(&full_key).await
    }

    async fn delete(&self, key: &str) -> std::result::Result<(), String> {
        let full_key = self.config.full_key(key);
        self.gcs_delete(&full_key).await
    }

    async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String> {
        let full_prefix = self.config.full_key(prefix);
        self.gcs_list(&full_prefix).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_account_only_config_fails_construction() {
        // A `service_account_key`-only config can never authenticate (no JWT
        // exchange is implemented), so `GcsBackend::new` must fail loudly at
        // construction time rather than allow a backend to be built that
        // will only fail later, at first-request time.
        let config = GcsConfig::with_service_account("my-bucket", "{\"type\":\"service_account\"}");
        let result = GcsBackend::new(config);
        assert!(result.is_err());
        let message = result.unwrap_err();
        assert!(message.contains("token exchange"), "message was: {message}");
    }

    #[test]
    fn test_access_token_config_constructs_successfully() {
        let config = GcsConfig::with_access_token("my-bucket", "test-token");
        assert!(GcsBackend::new(config).is_ok());
    }

    #[test]
    fn test_empty_bucket_fails_validation() {
        let config = GcsConfig::with_access_token("", "test-token");
        assert!(GcsBackend::new(config).is_err());
    }
}
