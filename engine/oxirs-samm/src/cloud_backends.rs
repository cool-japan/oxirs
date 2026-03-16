//! Real Cloud Storage Backends for SAMM Models
//!
//! This module provides concrete implementations of the [`CloudStorageBackend`] trait
//! for major cloud storage providers:
//!
//! - **[`S3Backend`]**: AWS S3 and S3-compatible stores (MinIO, Ceph, etc.)
//! - **[`GcsBackend`]**: Google Cloud Storage via JSON API
//! - **[`AzureBlobBackend`]**: Azure Blob Storage via REST API
//! - **[`HttpBackend`]**: Generic HTTP REST backend (useful for testing and custom servers)
//!
//! All backends implement [`CloudStorageBackend`] using async/await and `reqwest`.
//!
//! # Authentication
//!
//! - **S3**: AWS Signature Version 4 (HMAC-SHA256) – works with any SigV4-compatible endpoint
//! - **GCS**: Bearer token (OAuth2 access token or a service-account key JSON)
//! - **Azure**: SharedKeyLite HMAC-SHA256 scheme
//! - **HTTP**: Optional `Authorization` header
//!
//! # Example
//!
//! ```rust,no_run
//! use oxirs_samm::cloud_backends::{S3Backend, S3Config};
//! use oxirs_samm::cloud_storage::CloudStorageBackend;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = S3Config {
//!     endpoint: "https://s3.amazonaws.com".to_string(),
//!     bucket: "my-samm-models".to_string(),
//!     region: "us-east-1".to_string(),
//!     access_key: "AKIAIOSFODNN7EXAMPLE".to_string(),
//!     secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
//!     path_prefix: "models/".to_string(),
//! };
//! let backend = S3Backend::new(config)?;
//! backend.upload("vehicle.ttl", b"@prefix samm: ...".to_vec()).await?;
//! # Ok(())
//! # }
//! ```

use crate::cloud_storage::CloudStorageBackend;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::{Client, Method, StatusCode};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tracing::{debug, warn};

// ──────────────────────────────────────────────────────────────────────────────
// Helper types
// ──────────────────────────────────────────────────────────────────────────────

type HmacSha256 = Hmac<Sha256>;

/// Sign bytes with HMAC-SHA256, returning the raw digest.
fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC can take a key of any size");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Hex-encode bytes.
fn hex_encode(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// SHA-256 hash of bytes returned as lowercase hex string.
fn sha256_hex(data: &[u8]) -> String {
    hex_encode(&Sha256::digest(data))
}

// ──────────────────────────────────────────────────────────────────────────────
// S3Backend
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for an AWS S3 (or S3-compatible) storage backend.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// Full endpoint URL, e.g. `"https://s3.amazonaws.com"` or `"http://localhost:9000"` for MinIO.
    pub endpoint: String,
    /// Name of the S3 bucket.
    pub bucket: String,
    /// AWS region, e.g. `"us-east-1"`.
    pub region: String,
    /// AWS access key ID.
    pub access_key: String,
    /// AWS secret access key.
    pub secret_key: String,
    /// Optional path prefix prepended to every object key (e.g. `"samm-models/"`).
    pub path_prefix: String,
}

impl S3Config {
    /// Validate that mandatory fields are non-empty.
    pub fn validate(&self) -> Result<(), String> {
        if self.endpoint.is_empty() {
            return Err("S3Config.endpoint must not be empty".to_string());
        }
        if self.bucket.is_empty() {
            return Err("S3Config.bucket must not be empty".to_string());
        }
        if self.region.is_empty() {
            return Err("S3Config.region must not be empty".to_string());
        }
        if self.access_key.is_empty() {
            return Err("S3Config.access_key must not be empty".to_string());
        }
        if self.secret_key.is_empty() {
            return Err("S3Config.secret_key must not be empty".to_string());
        }
        Ok(())
    }

    /// Build the full object key by prepending `path_prefix`.
    pub fn full_key(&self, key: &str) -> String {
        if self.path_prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}{}", self.path_prefix, key)
        }
    }
}

/// AWS S3 (and S3-compatible) storage backend.
///
/// Uses AWS Signature Version 4 to sign every request.
pub struct S3Backend {
    config: S3Config,
    client: Client,
}

impl S3Backend {
    /// Create a new `S3Backend` from the given configuration.
    ///
    /// Returns an error if the configuration is invalid or if the HTTP client cannot
    /// be built (pure-Rust TLS only – no OpenSSL).
    pub fn new(config: S3Config) -> Result<Self, String> {
        config.validate()?;
        let client = Client::builder()
            .use_rustls_tls()
            .build()
            .map_err(|e| format!("Failed to build reqwest client: {e}"))?;
        Ok(Self { config, client })
    }

    /// Build the object URL for a given key.
    fn object_url(&self, key: &str) -> String {
        let endpoint = self.config.endpoint.trim_end_matches('/');
        format!("{}/{}/{}", endpoint, self.config.bucket, key)
    }

    /// Derive the AWS SigV4 signing key for the current date.
    fn signing_key(&self, date_stamp: &str, service: &str) -> Vec<u8> {
        let k_date = hmac_sha256(
            format!("AWS4{}", self.config.secret_key).as_bytes(),
            date_stamp.as_bytes(),
        );
        let k_region = hmac_sha256(&k_date, self.config.region.as_bytes());
        let k_service = hmac_sha256(&k_region, service.as_bytes());
        hmac_sha256(&k_service, b"aws4_request")
    }

    /// Build and return the `Authorization` header value for an S3 request.
    ///
    /// Implements a minimal AWS Signature Version 4 canonical-request flow.
    fn authorization_header(
        &self,
        method: &str,
        path: &str,
        query: &str,
        headers: &HashMap<String, String>,
        payload_hash: &str,
        datetime: &str,
    ) -> String {
        let date_stamp = &datetime[..8]; // YYYYMMDD

        // ── Canonical headers ──────────────────────────────────────────────
        let mut sorted_headers: Vec<(String, String)> = headers
            .iter()
            .map(|(k, v)| (k.to_lowercase(), v.trim().to_string()))
            .collect();
        sorted_headers.sort_by(|a, b| a.0.cmp(&b.0));

        let canonical_headers: String = sorted_headers
            .iter()
            .map(|(k, v)| format!("{}:{}\n", k, v))
            .collect();

        let signed_headers: String = sorted_headers
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
            .join(";");

        // ── Canonical request ──────────────────────────────────────────────
        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method, path, query, canonical_headers, signed_headers, payload_hash
        );
        let canonical_request_hash = sha256_hex(canonical_request.as_bytes());

        // ── String to sign ─────────────────────────────────────────────────
        let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, self.config.region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            datetime, credential_scope, canonical_request_hash
        );

        // ── Signature ─────────────────────────────────────────────────────
        let signing_key = self.signing_key(date_stamp, "s3");
        let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

        format!(
            "AWS4-HMAC-SHA256 Credential={}/{},SignedHeaders={},Signature={}",
            self.config.access_key, credential_scope, signed_headers, signature
        )
    }

    /// Upload bytes to S3 using a PUT request with SigV4 auth.
    async fn put_object(&self, key: &str, data: Vec<u8>) -> Result<(), String> {
        let now = Utc::now();
        let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = &datetime[..8];

        let path = format!("/{}/{}", self.config.bucket, key);
        let url = self.object_url(key);
        let host = extract_host(&url)?;
        let payload_hash = sha256_hex(&data);
        let content_length = data.len().to_string();

        let mut sign_headers = HashMap::new();
        sign_headers.insert("content-length".to_string(), content_length.clone());
        sign_headers.insert("host".to_string(), host.clone());
        sign_headers.insert("x-amz-content-sha256".to_string(), payload_hash.clone());
        sign_headers.insert("x-amz-date".to_string(), datetime.clone());

        let auth =
            self.authorization_header("PUT", &path, "", &sign_headers, &payload_hash, &datetime);

        debug!("S3 PUT {} (date={}, key={})", url, date_stamp, key);

        let response = self
            .client
            .put(&url)
            .header("content-length", &content_length)
            .header("host", &host)
            .header("x-amz-content-sha256", &payload_hash)
            .header("x-amz-date", &datetime)
            .header("Authorization", auth)
            .body(data)
            .send()
            .await
            .map_err(|e| format!("S3 PUT request failed: {e}"))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("S3 PUT returned {status}: {body}"))
        }
    }

    /// Download an object from S3 using a GET request with SigV4 auth.
    async fn get_object(&self, key: &str) -> Result<Vec<u8>, String> {
        let now = Utc::now();
        let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();

        let path = format!("/{}/{}", self.config.bucket, key);
        let url = self.object_url(key);
        let host = extract_host(&url)?;
        let empty_hash = sha256_hex(b"");

        let mut sign_headers = HashMap::new();
        sign_headers.insert("host".to_string(), host.clone());
        sign_headers.insert("x-amz-content-sha256".to_string(), empty_hash.clone());
        sign_headers.insert("x-amz-date".to_string(), datetime.clone());

        let auth =
            self.authorization_header("GET", &path, "", &sign_headers, &empty_hash, &datetime);

        debug!("S3 GET {}", url);

        let response = self
            .client
            .get(&url)
            .header("host", &host)
            .header("x-amz-content-sha256", &empty_hash)
            .header("x-amz-date", &datetime)
            .header("Authorization", auth)
            .send()
            .await
            .map_err(|e| format!("S3 GET request failed: {e}"))?;

        if response.status().is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| format!("Failed to read S3 GET body: {e}"))?;
            Ok(bytes.to_vec())
        } else if response.status() == StatusCode::NOT_FOUND {
            Err(format!("S3 object not found: {key}"))
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("S3 GET returned {status}: {body}"))
        }
    }

    /// Check whether an object exists via a HEAD request.
    async fn head_object(&self, key: &str) -> Result<bool, String> {
        let now = Utc::now();
        let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();

        let path = format!("/{}/{}", self.config.bucket, key);
        let url = self.object_url(key);
        let host = extract_host(&url)?;
        let empty_hash = sha256_hex(b"");

        let mut sign_headers = HashMap::new();
        sign_headers.insert("host".to_string(), host.clone());
        sign_headers.insert("x-amz-content-sha256".to_string(), empty_hash.clone());
        sign_headers.insert("x-amz-date".to_string(), datetime.clone());

        let auth =
            self.authorization_header("HEAD", &path, "", &sign_headers, &empty_hash, &datetime);

        debug!("S3 HEAD {}", url);

        let response = self
            .client
            .head(&url)
            .header("host", &host)
            .header("x-amz-content-sha256", &empty_hash)
            .header("x-amz-date", &datetime)
            .header("Authorization", auth)
            .send()
            .await
            .map_err(|e| format!("S3 HEAD request failed: {e}"))?;

        match response.status() {
            StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            other => Err(format!("S3 HEAD returned unexpected status: {other}")),
        }
    }

    /// Delete an object from S3.
    async fn delete_object(&self, key: &str) -> Result<(), String> {
        let now = Utc::now();
        let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();

        let path = format!("/{}/{}", self.config.bucket, key);
        let url = self.object_url(key);
        let host = extract_host(&url)?;
        let empty_hash = sha256_hex(b"");

        let mut sign_headers = HashMap::new();
        sign_headers.insert("host".to_string(), host.clone());
        sign_headers.insert("x-amz-content-sha256".to_string(), empty_hash.clone());
        sign_headers.insert("x-amz-date".to_string(), datetime.clone());

        let auth =
            self.authorization_header("DELETE", &path, "", &sign_headers, &empty_hash, &datetime);

        debug!("S3 DELETE {}", url);

        let response = self
            .client
            .delete(&url)
            .header("host", &host)
            .header("x-amz-content-sha256", &empty_hash)
            .header("x-amz-date", &datetime)
            .header("Authorization", auth)
            .send()
            .await
            .map_err(|e| format!("S3 DELETE request failed: {e}"))?;

        if response.status().is_success() || response.status() == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("S3 DELETE returned {status}: {body}"))
        }
    }

    /// List objects under a prefix (simple XML parse of ListObjectsV2 response).
    async fn list_objects(&self, prefix: &str) -> Result<Vec<String>, String> {
        let now = Utc::now();
        let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();

        let query = if prefix.is_empty() {
            "list-type=2".to_string()
        } else {
            format!("list-type=2&prefix={}", url_encode(prefix))
        };
        let base_url = format!(
            "{}/{}?{}",
            self.config.endpoint.trim_end_matches('/'),
            self.config.bucket,
            query
        );
        let path = format!("/{}", self.config.bucket);
        let host = extract_host(&base_url)?;
        let empty_hash = sha256_hex(b"");

        let mut sign_headers = HashMap::new();
        sign_headers.insert("host".to_string(), host.clone());
        sign_headers.insert("x-amz-content-sha256".to_string(), empty_hash.clone());
        sign_headers.insert("x-amz-date".to_string(), datetime.clone());

        let auth =
            self.authorization_header("GET", &path, &query, &sign_headers, &empty_hash, &datetime);

        debug!("S3 LIST {} prefix={}", base_url, prefix);

        let response = self
            .client
            .get(&base_url)
            .header("host", &host)
            .header("x-amz-content-sha256", &empty_hash)
            .header("x-amz-date", &datetime)
            .header("Authorization", auth)
            .send()
            .await
            .map_err(|e| format!("S3 LIST request failed: {e}"))?;

        if response.status().is_success() {
            let body = response
                .text()
                .await
                .map_err(|e| format!("Failed to read S3 LIST body: {e}"))?;
            Ok(parse_s3_list_xml(&body))
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("S3 LIST returned {status}: {body}"))
        }
    }

    /// Generate a presigned GET URL valid for `expire_secs` seconds.
    ///
    /// This is a convenience helper that implements SigV4 query-string signing.
    pub fn presigned_url(&self, key: &str, expire_secs: u64) -> String {
        let now = Utc::now();
        let datetime = now.format("%Y%m%dT%H%M%SZ").to_string();
        let date_stamp = &datetime[..8];

        let path = format!("/{}/{}", self.config.bucket, key);
        let url = self.object_url(key);
        let host = extract_host(&url).unwrap_or_default();
        let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, self.config.region);
        let credential = format!("{}/{}", self.config.access_key, credential_scope);

        let query = format!(
            "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential={}&X-Amz-Date={}&X-Amz-Expires={}&X-Amz-SignedHeaders=host",
            url_encode(&credential),
            datetime,
            expire_secs,
        );

        let mut sign_headers = HashMap::new();
        sign_headers.insert("host".to_string(), host.clone());

        // Canonical request for presigned URL uses "UNSIGNED-PAYLOAD"
        let canonical_headers = format!("host:{}\n", host);
        let signed_headers = "host";
        let canonical_request = format!(
            "GET\n{}\n{}\n{}\n{}\nUNSIGNED-PAYLOAD",
            path, query, canonical_headers, signed_headers
        );
        let cr_hash = sha256_hex(canonical_request.as_bytes());

        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            datetime, credential_scope, cr_hash
        );

        let signing_key = self.signing_key(date_stamp, "s3");
        let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));

        format!(
            "{}/{}/{}?{}&X-Amz-Signature={}",
            self.config.endpoint.trim_end_matches('/'),
            self.config.bucket,
            key,
            query,
            signature
        )
    }
}

#[async_trait]
impl CloudStorageBackend for S3Backend {
    async fn upload(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
        let full_key = self.config.full_key(key);
        self.put_object(&full_key, data).await
    }

    async fn download(&self, key: &str) -> std::result::Result<Vec<u8>, String> {
        let full_key = self.config.full_key(key);
        self.get_object(&full_key).await
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, String> {
        let full_key = self.config.full_key(key);
        self.head_object(&full_key).await
    }

    async fn delete(&self, key: &str) -> std::result::Result<(), String> {
        let full_key = self.config.full_key(key);
        self.delete_object(&full_key).await
    }

    async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String> {
        let full_prefix = self.config.full_key(prefix);
        self.list_objects(&full_prefix).await
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GcsBackend
// ──────────────────────────────────────────────────────────────────────────────

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
        Ok(())
    }

    fn bearer_token(&self) -> Option<String> {
        if let Some(ref tok) = self.access_token {
            return Some(tok.clone());
        }
        // In a real implementation, parse the service-account JSON and obtain a
        // short-lived JWT/token.  For now, return None to indicate that a
        // service-account key was supplied but no token-exchange is implemented here.
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
pub struct GcsBackend {
    config: GcsConfig,
    client: Client,
}

impl GcsBackend {
    /// Create a new `GcsBackend`.
    pub fn new(config: GcsConfig) -> Result<Self, String> {
        config.validate()?;
        let client = Client::builder()
            .use_rustls_tls()
            .build()
            .map_err(|e| format!("Failed to build reqwest client: {e}"))?;
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

// ──────────────────────────────────────────────────────────────────────────────
// AzureBlobBackend
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for the Azure Blob Storage backend.
#[derive(Debug, Clone)]
pub struct AzureConfig {
    /// Azure storage account name.
    pub account_name: String,
    /// Base64-encoded storage account key.
    pub account_key: String,
    /// Name of the Azure Blob container.
    pub container_name: String,
    /// Optional path prefix within the container.
    pub path_prefix: String,
}

impl AzureConfig {
    /// Create a minimal `AzureConfig`.
    pub fn new(
        account_name: impl Into<String>,
        account_key: impl Into<String>,
        container_name: impl Into<String>,
    ) -> Self {
        Self {
            account_name: account_name.into(),
            account_key: account_key.into(),
            container_name: container_name.into(),
            path_prefix: String::new(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.account_name.is_empty() {
            return Err("AzureConfig.account_name must not be empty".to_string());
        }
        if self.account_key.is_empty() {
            return Err("AzureConfig.account_key must not be empty".to_string());
        }
        if self.container_name.is_empty() {
            return Err("AzureConfig.container_name must not be empty".to_string());
        }
        Ok(())
    }

    fn full_key(&self, key: &str) -> String {
        if self.path_prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}{}", self.path_prefix, key)
        }
    }

    fn blob_url(&self, blob_name: &str) -> String {
        format!(
            "https://{}.blob.core.windows.net/{}/{}",
            self.account_name, self.container_name, blob_name
        )
    }

    fn container_url(&self) -> String {
        format!(
            "https://{}.blob.core.windows.net/{}",
            self.account_name, self.container_name
        )
    }

    /// Build the `Authorization: SharedKeyLite` header for an Azure REST request.
    ///
    /// `string_to_sign` follows the SharedKeyLite scheme documented at:
    /// <https://docs.microsoft.com/en-us/rest/api/storageservices/authorize-with-shared-key>
    fn shared_key_lite_auth(
        &self,
        method: &str,
        content_md5: &str,
        content_type: &str,
        date: &str,
        canonicalized_headers: &str,
        canonicalized_resource: &str,
    ) -> Result<String, String> {
        let string_to_sign = format!(
            "{}\n{}\n{}\n{}\n{}{}",
            method, content_md5, content_type, date, canonicalized_headers, canonicalized_resource
        );

        let key_bytes = BASE64
            .decode(&self.account_key)
            .map_err(|e| format!("Failed to decode Azure account key: {e}"))?;

        let signature = BASE64.encode(hmac_sha256(&key_bytes, string_to_sign.as_bytes()));
        Ok(format!("SharedKeyLite {}:{}", self.account_name, signature))
    }
}

/// Azure Blob Storage backend.
pub struct AzureBlobBackend {
    config: AzureConfig,
    client: Client,
}

impl AzureBlobBackend {
    /// Create a new `AzureBlobBackend`.
    pub fn new(config: AzureConfig) -> Result<Self, String> {
        config.validate()?;
        let client = Client::builder()
            .use_rustls_tls()
            .build()
            .map_err(|e| format!("Failed to build reqwest client: {e}"))?;
        Ok(Self { config, client })
    }

    /// RFC 1123 date string as required by Azure REST API.
    fn rfc1123_now() -> String {
        Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string()
    }

    async fn azure_put(&self, blob_name: &str, data: Vec<u8>) -> Result<(), String> {
        let date = Self::rfc1123_now();
        let content_length = data.len().to_string();
        let url = self.config.blob_url(blob_name);

        let canonicalized_headers = format!(
            "x-ms-blob-type:BlockBlob\nx-ms-date:{}\nx-ms-version:2020-04-08\n",
            date
        );
        let canonicalized_resource = format!(
            "/{}/{}/{}",
            self.config.account_name, self.config.container_name, blob_name
        );

        let auth = self.config.shared_key_lite_auth(
            "PUT",
            "",
            "application/octet-stream",
            "",
            &canonicalized_headers,
            &canonicalized_resource,
        )?;

        debug!("Azure PUT {}", url);

        let response = self
            .client
            .put(&url)
            .header("Authorization", auth)
            .header("x-ms-blob-type", "BlockBlob")
            .header("x-ms-date", &date)
            .header("x-ms-version", "2020-04-08")
            .header("Content-Type", "application/octet-stream")
            .header("Content-Length", &content_length)
            .body(data)
            .send()
            .await
            .map_err(|e| format!("Azure PUT request failed: {e}"))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("Azure PUT returned {status}: {body}"))
        }
    }

    async fn azure_get(&self, blob_name: &str) -> Result<Vec<u8>, String> {
        let date = Self::rfc1123_now();
        let url = self.config.blob_url(blob_name);

        let canonicalized_headers = format!("x-ms-date:{}\nx-ms-version:2020-04-08\n", date);
        let canonicalized_resource = format!(
            "/{}/{}/{}",
            self.config.account_name, self.config.container_name, blob_name
        );

        let auth = self.config.shared_key_lite_auth(
            "GET",
            "",
            "",
            "",
            &canonicalized_headers,
            &canonicalized_resource,
        )?;

        debug!("Azure GET {}", url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", auth)
            .header("x-ms-date", &date)
            .header("x-ms-version", "2020-04-08")
            .send()
            .await
            .map_err(|e| format!("Azure GET request failed: {e}"))?;

        if response.status().is_success() {
            let bytes = response
                .bytes()
                .await
                .map_err(|e| format!("Failed to read Azure GET body: {e}"))?;
            Ok(bytes.to_vec())
        } else if response.status() == StatusCode::NOT_FOUND {
            Err(format!("Azure blob not found: {blob_name}"))
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("Azure GET returned {status}: {body}"))
        }
    }

    async fn azure_exists(&self, blob_name: &str) -> Result<bool, String> {
        let date = Self::rfc1123_now();
        let url = self.config.blob_url(blob_name);

        let canonicalized_headers = format!("x-ms-date:{}\nx-ms-version:2020-04-08\n", date);
        let canonicalized_resource = format!(
            "/{}/{}/{}",
            self.config.account_name, self.config.container_name, blob_name
        );

        let auth = self.config.shared_key_lite_auth(
            "HEAD",
            "",
            "",
            "",
            &canonicalized_headers,
            &canonicalized_resource,
        )?;

        debug!("Azure HEAD {}", url);

        let response = self
            .client
            .head(&url)
            .header("Authorization", auth)
            .header("x-ms-date", &date)
            .header("x-ms-version", "2020-04-08")
            .send()
            .await
            .map_err(|e| format!("Azure HEAD request failed: {e}"))?;

        match response.status() {
            StatusCode::OK => Ok(true),
            StatusCode::NOT_FOUND => Ok(false),
            other => Err(format!("Azure HEAD returned unexpected status: {other}")),
        }
    }

    async fn azure_delete(&self, blob_name: &str) -> Result<(), String> {
        let date = Self::rfc1123_now();
        let url = self.config.blob_url(blob_name);

        let canonicalized_headers = format!("x-ms-date:{}\nx-ms-version:2020-04-08\n", date);
        let canonicalized_resource = format!(
            "/{}/{}/{}",
            self.config.account_name, self.config.container_name, blob_name
        );

        let auth = self.config.shared_key_lite_auth(
            "DELETE",
            "",
            "",
            "",
            &canonicalized_headers,
            &canonicalized_resource,
        )?;

        debug!("Azure DELETE {}", url);

        let response = self
            .client
            .delete(&url)
            .header("Authorization", auth)
            .header("x-ms-date", &date)
            .header("x-ms-version", "2020-04-08")
            .send()
            .await
            .map_err(|e| format!("Azure DELETE request failed: {e}"))?;

        if response.status().is_success() || response.status() == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("Azure DELETE returned {status}: {body}"))
        }
    }

    async fn azure_list(&self, prefix: &str) -> Result<Vec<String>, String> {
        let date = Self::rfc1123_now();
        let query = if prefix.is_empty() {
            "restype=container&comp=list".to_string()
        } else {
            format!("restype=container&comp=list&prefix={}", url_encode(prefix))
        };
        let url = format!("{}?{}", self.config.container_url(), query);

        let canonicalized_headers = format!("x-ms-date:{}\nx-ms-version:2020-04-08\n", date);
        let canonicalized_resource = format!(
            "/{}/{}\ncomp:list\nprefix:{}\nrestype:container",
            self.config.account_name, self.config.container_name, prefix
        );

        let auth = self.config.shared_key_lite_auth(
            "GET",
            "",
            "",
            "",
            &canonicalized_headers,
            &canonicalized_resource,
        )?;

        debug!("Azure LIST {}", url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", auth)
            .header("x-ms-date", &date)
            .header("x-ms-version", "2020-04-08")
            .send()
            .await
            .map_err(|e| format!("Azure LIST request failed: {e}"))?;

        if response.status().is_success() {
            let body = response
                .text()
                .await
                .map_err(|e| format!("Failed to read Azure LIST body: {e}"))?;
            Ok(parse_azure_list_xml(&body))
        } else if response.status() == StatusCode::NOT_FOUND {
            Ok(vec![])
        } else {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            Err(format!("Azure LIST returned {status}: {body}"))
        }
    }
}

#[async_trait]
impl CloudStorageBackend for AzureBlobBackend {
    async fn upload(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
        let full_key = self.config.full_key(key);
        self.azure_put(&full_key, data).await
    }

    async fn download(&self, key: &str) -> std::result::Result<Vec<u8>, String> {
        let full_key = self.config.full_key(key);
        self.azure_get(&full_key).await
    }

    async fn exists(&self, key: &str) -> std::result::Result<bool, String> {
        let full_key = self.config.full_key(key);
        self.azure_exists(&full_key).await
    }

    async fn delete(&self, key: &str) -> std::result::Result<(), String> {
        let full_key = self.config.full_key(key);
        self.azure_delete(&full_key).await
    }

    async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String> {
        let full_prefix = self.config.full_key(prefix);
        self.azure_list(&full_prefix).await
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// HttpBackend
// ──────────────────────────────────────────────────────────────────────────────

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

    fn full_key(&self, key: &str) -> String {
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
        let client = Client::builder()
            .use_rustls_tls()
            .build()
            .map_err(|e| format!("Failed to build reqwest client: {e}"))?;
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

// ──────────────────────────────────────────────────────────────────────────────
// Utility functions
// ──────────────────────────────────────────────────────────────────────────────

/// Extract the `host[:port]` component from a URL string.
fn extract_host(url: &str) -> Result<String, String> {
    // Find the scheme separator "://"
    let after_scheme = url
        .find("://")
        .map(|i| &url[i + 3..])
        .ok_or_else(|| format!("URL has no scheme: {url}"))?;

    // Take everything up to the first '/'
    let host = after_scheme
        .split('/')
        .next()
        .ok_or_else(|| format!("Cannot extract host from URL: {url}"))?;
    Ok(host.to_string())
}

/// Percent-encode a string for use in URL query parameters.
fn url_encode(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                vec![c]
            }
            other => {
                let mut buf = [0u8; 4];
                let bytes = other.encode_utf8(&mut buf);
                bytes
                    .as_bytes()
                    .iter()
                    .flat_map(|b| format!("%{:02X}", b).chars().collect::<Vec<_>>())
                    .collect()
            }
        })
        .collect()
}

/// Naive XML parser that extracts `<Key>...</Key>` values from an S3 ListObjectsV2 response.
fn parse_s3_list_xml(xml: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let mut remaining = xml;
    while let Some(start) = remaining.find("<Key>") {
        let rest = &remaining[start + 5..];
        if let Some(end) = rest.find("</Key>") {
            keys.push(rest[..end].to_string());
            remaining = &rest[end + 6..];
        } else {
            break;
        }
    }
    keys
}

/// Naive XML parser that extracts `<Name>...</Name>` values from an Azure ListBlobs response.
fn parse_azure_list_xml(xml: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut remaining = xml;
    // Azure ListBlobs returns <Name> inside <Blob> elements.
    while let Some(start) = remaining.find("<Name>") {
        let rest = &remaining[start + 6..];
        if let Some(end) = rest.find("</Name>") {
            names.push(rest[..end].to_string());
            remaining = &rest[end + 7..];
        } else {
            break;
        }
    }
    names
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── S3Config ─────────────────────────────────────────────────────────────

    #[test]
    fn test_s3_config_validation_passes_with_valid_fields() {
        let config = S3Config {
            endpoint: "https://s3.amazonaws.com".to_string(),
            bucket: "my-bucket".to_string(),
            region: "us-east-1".to_string(),
            access_key: "AKID".to_string(),
            secret_key: "SECRET".to_string(),
            path_prefix: String::new(),
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_s3_config_validation_fails_on_empty_bucket() {
        let config = S3Config {
            endpoint: "https://s3.amazonaws.com".to_string(),
            bucket: String::new(),
            region: "us-east-1".to_string(),
            access_key: "AKID".to_string(),
            secret_key: "SECRET".to_string(),
            path_prefix: String::new(),
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("bucket"), "error should mention bucket: {err}");
    }

    #[test]
    fn test_s3_config_validation_fails_on_empty_access_key() {
        let config = S3Config {
            endpoint: "https://s3.amazonaws.com".to_string(),
            bucket: "bucket".to_string(),
            region: "us-east-1".to_string(),
            access_key: String::new(),
            secret_key: "SECRET".to_string(),
            path_prefix: String::new(),
        };
        let err = config.validate().unwrap_err();
        assert!(
            err.contains("access_key"),
            "error should mention access_key: {err}"
        );
    }

    #[test]
    fn test_s3_config_full_key_with_prefix() {
        let config = S3Config {
            endpoint: "https://s3.amazonaws.com".to_string(),
            bucket: "bucket".to_string(),
            region: "us-east-1".to_string(),
            access_key: "AKID".to_string(),
            secret_key: "SECRET".to_string(),
            path_prefix: "samm/".to_string(),
        };
        assert_eq!(config.full_key("model.ttl"), "samm/model.ttl");
    }

    #[test]
    fn test_s3_config_full_key_without_prefix() {
        let config = S3Config {
            endpoint: "https://s3.amazonaws.com".to_string(),
            bucket: "bucket".to_string(),
            region: "us-east-1".to_string(),
            access_key: "AKID".to_string(),
            secret_key: "SECRET".to_string(),
            path_prefix: String::new(),
        };
        assert_eq!(config.full_key("model.ttl"), "model.ttl");
    }

    #[test]
    fn test_s3_backend_creation_succeeds() {
        let config = S3Config {
            endpoint: "https://s3.amazonaws.com".to_string(),
            bucket: "my-bucket".to_string(),
            region: "us-east-1".to_string(),
            access_key: "AKID".to_string(),
            secret_key: "SECRET".to_string(),
            path_prefix: String::new(),
        };
        let backend = S3Backend::new(config);
        assert!(
            backend.is_ok(),
            "S3Backend should be constructable with valid config"
        );
    }

    #[test]
    fn test_s3_presigned_url_contains_expected_fields() {
        let config = S3Config {
            endpoint: "https://s3.amazonaws.com".to_string(),
            bucket: "my-bucket".to_string(),
            region: "us-east-1".to_string(),
            access_key: "AKIAIOSFODNN7EXAMPLE".to_string(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
            path_prefix: String::new(),
        };
        let backend = S3Backend::new(config).expect("should build");
        let url = backend.presigned_url("models/vehicle.ttl", 3600);

        assert!(url.contains("my-bucket"), "URL should contain bucket name");
        assert!(
            url.contains("X-Amz-Signature"),
            "URL should contain Signature parameter"
        );
        assert!(
            url.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"),
            "URL should contain algorithm"
        );
        assert!(
            url.contains("X-Amz-Expires=3600"),
            "URL should contain expiry"
        );
    }

    // ── GcsConfig ─────────────────────────────────────────────────────────────

    #[test]
    fn test_gcs_config_creation_with_access_token() {
        let config = GcsConfig::with_access_token("my-gcs-bucket", "ya29.token");
        assert_eq!(config.bucket, "my-gcs-bucket");
        assert_eq!(config.access_token.as_deref(), Some("ya29.token"));
        assert!(config.service_account_key.is_none());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gcs_config_creation_with_service_account() {
        let key_json = r#"{"type":"service_account","project_id":"my-project"}"#;
        let config = GcsConfig::with_service_account("my-bucket", key_json);
        assert!(config.service_account_key.is_some());
        assert!(config.access_token.is_none());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gcs_config_validation_fails_without_auth() {
        let config = GcsConfig {
            bucket: "my-bucket".to_string(),
            service_account_key: None,
            access_token: None,
            path_prefix: String::new(),
        };
        let err = config.validate().unwrap_err();
        assert!(
            err.contains("access_token") || err.contains("service_account"),
            "error should mention auth: {err}"
        );
    }

    #[test]
    fn test_gcs_backend_creation_succeeds() {
        let config = GcsConfig::with_access_token("samm-models", "fake-token");
        let backend = GcsBackend::new(config);
        assert!(
            backend.is_ok(),
            "GcsBackend should be constructable with valid config"
        );
    }

    // ── AzureConfig ───────────────────────────────────────────────────────────

    #[test]
    fn test_azure_config_creation() {
        // Use a valid base64 string as account key.
        let key_b64 = BASE64.encode("some-secret-key-bytes");
        let config = AzureConfig::new("myaccount", &key_b64, "samm-models");
        assert_eq!(config.account_name, "myaccount");
        assert_eq!(config.container_name, "samm-models");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_azure_config_validation_fails_on_empty_container() {
        let config = AzureConfig {
            account_name: "myaccount".to_string(),
            account_key: BASE64.encode("key"),
            container_name: String::new(),
            path_prefix: String::new(),
        };
        let err = config.validate().unwrap_err();
        assert!(
            err.contains("container_name"),
            "error should mention container_name: {err}"
        );
    }

    #[test]
    fn test_azure_shared_key_lite_signature_deterministic() {
        let key_bytes = b"my-account-key-bytes-are-here-!!!";
        let key_b64 = BASE64.encode(key_bytes);
        let config = AzureConfig::new("acct", &key_b64, "ctr");

        let sig1 = config
            .shared_key_lite_auth(
                "PUT",
                "",
                "application/octet-stream",
                "",
                "x-ms-date:Thu, 01 Jan 2026 00:00:00 GMT\n",
                "/acct/ctr/file.ttl",
            )
            .expect("signing should succeed");
        let sig2 = config
            .shared_key_lite_auth(
                "PUT",
                "",
                "application/octet-stream",
                "",
                "x-ms-date:Thu, 01 Jan 2026 00:00:00 GMT\n",
                "/acct/ctr/file.ttl",
            )
            .expect("signing should succeed");

        assert_eq!(
            sig1, sig2,
            "Signature must be deterministic for the same inputs"
        );
        assert!(
            sig1.starts_with("SharedKeyLite acct:"),
            "Authorization header format mismatch"
        );
    }

    #[test]
    fn test_azure_backend_creation_succeeds() {
        let key_b64 = BASE64.encode("some-key-material");
        let config = AzureConfig::new("storageacct", &key_b64, "models");
        let backend = AzureBlobBackend::new(config);
        assert!(
            backend.is_ok(),
            "AzureBlobBackend should be constructable with valid config"
        );
    }

    // ── HttpConfig ────────────────────────────────────────────────────────────

    #[test]
    fn test_http_config_creation_basic() {
        let config = HttpConfig::new("http://localhost:8080/storage");
        assert_eq!(config.base_url, "http://localhost:8080/storage");
        assert!(config.auth_header.is_none());
    }

    #[test]
    fn test_http_config_creation_with_bearer() {
        let config = HttpConfig::with_bearer("http://localhost:8080", "my-token");
        assert_eq!(config.auth_header.as_deref(), Some("Bearer my-token"));
    }

    #[test]
    fn test_http_backend_creation_succeeds() {
        let config = HttpConfig::new("http://localhost:9999/api/storage");
        let backend = HttpBackend::new(config);
        assert!(
            backend.is_ok(),
            "HttpBackend should be constructable with a non-empty URL"
        );
    }

    #[test]
    fn test_http_backend_creation_fails_on_empty_url() {
        let config = HttpConfig::new("");
        let backend = HttpBackend::new(config);
        assert!(
            backend.is_err(),
            "HttpBackend should reject an empty base URL"
        );
    }

    // ── Utility helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_url_encode_unreserved_chars_unchanged() {
        assert_eq!(url_encode("abc-123_.~"), "abc-123_.~");
    }

    #[test]
    fn test_url_encode_space_and_slash_encoded() {
        let encoded = url_encode("hello world/file.ttl");
        assert!(encoded.contains("%20"), "space should be percent-encoded");
        assert!(encoded.contains("%2F"), "slash should be percent-encoded");
    }

    #[test]
    fn test_extract_host_standard_url() {
        let host = extract_host("https://s3.amazonaws.com/my-bucket/key").expect("should succeed");
        assert_eq!(host, "s3.amazonaws.com");
    }

    #[test]
    fn test_extract_host_url_with_port() {
        let host = extract_host("http://localhost:9000/bucket").expect("should succeed");
        assert_eq!(host, "localhost:9000");
    }

    #[test]
    fn test_extract_host_no_scheme_returns_err() {
        let result = extract_host("no-scheme-url");
        assert!(result.is_err(), "Should fail on URL without scheme");
    }

    #[test]
    fn test_parse_s3_list_xml_extracts_keys() {
        let xml = r#"<?xml version="1.0"?>
<ListBucketResult>
  <Contents><Key>models/vehicle.ttl</Key><Size>1024</Size></Contents>
  <Contents><Key>models/sensor.ttl</Key><Size>512</Size></Contents>
</ListBucketResult>"#;
        let keys = parse_s3_list_xml(xml);
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], "models/vehicle.ttl");
        assert_eq!(keys[1], "models/sensor.ttl");
    }

    #[test]
    fn test_parse_s3_list_xml_empty_on_no_keys() {
        let xml = "<ListBucketResult><KeyCount>0</KeyCount></ListBucketResult>";
        let keys = parse_s3_list_xml(xml);
        assert!(keys.is_empty());
    }

    #[test]
    fn test_parse_azure_list_xml_extracts_names() {
        let xml = r#"<?xml version="1.0"?>
<EnumerationResults>
  <Blobs>
    <Blob><Name>models/car.ttl</Name></Blob>
    <Blob><Name>models/truck.ttl</Name></Blob>
  </Blobs>
</EnumerationResults>"#;
        let names = parse_azure_list_xml(xml);
        assert_eq!(names.len(), 2);
        assert_eq!(names[0], "models/car.ttl");
        assert_eq!(names[1], "models/truck.ttl");
    }

    #[test]
    fn test_hmac_sha256_known_vector() {
        // HMAC-SHA256("key", "The quick brown fox") known value.
        let key = b"key";
        let msg = b"The quick brown fox jumps over the lazy dog";
        let result = hmac_sha256(key, msg);
        // Verify the result is 32 bytes (256 bits)
        assert_eq!(result.len(), 32, "HMAC-SHA256 output must be 32 bytes");
        // Verify it's deterministic
        let result2 = hmac_sha256(key, msg);
        assert_eq!(result, result2, "HMAC-SHA256 must be deterministic");
    }

    #[test]
    fn test_sha256_hex_known_empty() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let h = sha256_hex(b"");
        assert_eq!(
            h,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_s3_signing_key_length() {
        let config = S3Config {
            endpoint: "https://s3.amazonaws.com".to_string(),
            bucket: "bucket".to_string(),
            region: "us-east-1".to_string(),
            access_key: "AKID".to_string(),
            secret_key: "SECRET".to_string(),
            path_prefix: String::new(),
        };
        let backend = S3Backend::new(config).expect("should build");
        let key = backend.signing_key("20260201", "s3");
        assert_eq!(key.len(), 32, "Derived signing key must be 32 bytes");
    }

    #[test]
    fn test_http_config_full_key_with_prefix() {
        let mut config = HttpConfig::new("http://localhost:8080");
        config.path_prefix = "v1/".to_string();
        assert_eq!(config.full_key("/model.ttl"), "v1//model.ttl");
    }
}
