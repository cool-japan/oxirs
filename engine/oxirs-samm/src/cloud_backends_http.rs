//! Generic HTTP REST storage backend.
//!
//! Useful for integration tests, custom REST APIs, and CI mocks.

use crate::cloud_backends_common::url_encode;
use crate::cloud_storage::CloudStorageBackend;
use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use tracing::debug;

/// Configuration for a generic HTTP REST storage backend.
///
/// This backend is useful for:
/// - Integration tests with a local HTTP server
/// - Custom storage servers that expose a REST API
/// - Mocking cloud storage in CI environments
///
/// Expected server conventions:
/// - `PUT  <base_url>/<key>`            – upload
/// - `GET  <base_url>/<key>`            – download
/// - `HEAD <base_url>/<key>`            – exists check (200 = yes, 404 = no)
/// - `DELETE <base_url>/<key>`          – delete
/// - `GET  <base_url>?prefix=<prefix>`  – list (returns JSON array of strings)
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Base URL of the storage server (e.g. `"http://localhost:8080/storage"`).
    pub base_url: String,
    /// Optional `Authorization` header value.
    pub auth_header: Option<String>,
    /// Optional path prefix prepended to every object key.
    pub path_prefix: String,
}

impl HttpConfig {
    /// Create a basic `HttpConfig` with no authentication.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            auth_header: None,
            path_prefix: String::new(),
        }
    }

    /// Create an `HttpConfig` with Bearer token authentication.
    pub fn with_bearer(base_url: impl Into<String>, token: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            auth_header: Some(format!("Bearer {}", token.into())),
            path_prefix: String::new(),
        }
    }

    pub(crate) fn full_key(&self, key: &str) -> String {
        if self.path_prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}{}", self.path_prefix, key)
        }
    }

    fn object_url(&self, key: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), key)
    }
}

/// Generic HTTP REST storage backend.
pub struct HttpBackend {
    config: HttpConfig,
    client: Client,
}

impl HttpBackend {
    /// Create a new `HttpBackend`.
    pub fn new(config: HttpConfig) -> Result<Self, String> {
        if config.base_url.is_empty() {
            return Err("HttpConfig.base_url must not be empty".to_string());
        }
        let client = crate::cloud_backends_common::build_tls_client()?;
        Ok(Self { config, client })
    }

    fn add_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref auth) = self.config.auth_header {
            builder.header("Authorization", auth)
        } else {
            builder
        }
    }

    async fn http_put(&self, key: &str, data: Vec<u8>) -> Result<(), String> {
        let url = self.config.object_url(key);
        debug!("HTTP PUT {}", url);

        let request = self
            .client
            .put(&url)
            .header("Content-Type", "application/octet-stream")
            .body(data);
        let request = self.add_auth(request);

        let response = request
            .send()
            .await
            .map_err(|e| format!("HTTP PUT failed: {e}"))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("HTTP PUT returned {status}: {body}"))
        }
    }

    async fn http_get(&self, key: &str) -> Result<Vec<u8>, String> {
        let url = self.config.object_url(key);
        debug!("HTTP GET {}", url);

        let request = self.client.get(&url);
        let request = self.add_auth(request);

        let response = request
            .send()
            .await
            .map_err(|e| format!("HTTP GET failed: {e}"))?;

        if response.status().is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| format!("Failed to read HTTP GET body: {e}"))?;
            Ok(bytes.to_vec())
        } else if response.status() == StatusCode::NOT_FOUND {
            Err(format!("HTTP object not found: {key}"))
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("HTTP GET returned {status}: {body}"))
        }
    }

    async fn http_head(&self, key: &str) -> Result<bool, String> {
        let url = self.config.object_url(key);
        debug!("HTTP HEAD {}", url);

        let request = self.client.head(&url);
        let request = self.add_auth(request);

        let response = request
            .send()
            .await
            .map_err(|e| format!("HTTP HEAD failed: {e}"))?;

        match response.status() {
            StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            other => Err(format!("HTTP HEAD returned unexpected status: {other}")),
        }
    }

    async fn http_delete(&self, key: &str) -> Result<(), String> {
        let url = self.config.object_url(key);
        debug!("HTTP DELETE {}", url);

        let request = self.client.delete(&url);
        let request = self.add_auth(request);

        let response = request
            .send()
            .await
            .map_err(|e| format!("HTTP DELETE failed: {e}"))?;

        if response.status().is_success() || response.status() == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("HTTP DELETE returned {status}: {body}"))
        }
    }

    async fn http_list(&self, prefix: &str) -> Result<Vec<String>, String> {
        let url = if prefix.is_empty() {
            self.config.base_url.clone()
        } else {
            format!("{}?prefix={}", self.config.base_url, url_encode(prefix))
        };
        debug!("HTTP LIST {}", url);

        let request = self.client.get(&url);
        let request = self.add_auth(request);

        let response = request
            .send()
            .await
            .map_err(|e| format!("HTTP LIST failed: {e}"))?;

        if response.status().is_success() {
            let keys: Vec<String> = response
                .json()
                .await
                .map_err(|e| format!("Failed to parse HTTP LIST response as JSON: {e}"))?;
            Ok(keys)
        } else if response.status() == StatusCode::NOT_FOUND {
            Ok(vec![])
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("HTTP LIST returned {status}: {body}"))
        }
    }
}

#[async_trait]
impl CloudStorageBackend for HttpBackend {
    async fn upload(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
        let full_key = self.config.full_key(key);
        self.http_put(&full_key, data).await
    }

    async fn download(&self, key: &str) -> std::result::Result<Vec<u8>, String> {
        let full_key = self.config.full_key(key);
        self.http_get(&full_key).await
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, String> {
        let full_key = self.config.full_key(key);
        self.http_head(&full_key).await
    }

    async fn delete(&self, key: &str) -> std::result::Result<(), String> {
        let full_key = self.config.full_key(key);
        self.http_delete(&full_key).await
    }

    async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String> {
        let full_prefix = self.config.full_key(prefix);
        self.http_list(&full_prefix).await
    }
}
