//! AWS S3 (and S3-compatible) storage backend.

use crate::cloud_backends_common::{
    extract_host, hex_encode, hmac_sha256, parse_s3_list_xml, sha256_hex, url_encode,
};
use crate::cloud_storage::CloudStorageBackend;
use async_trait::async_trait;
use chrono::Utc;
use reqwest::{Client, StatusCode};
use std::collections::HashMap;
use tracing::debug;

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
        let client = crate::cloud_backends_common::build_tls_client()?;
        Ok(Self { config, client })
    }

    /// Build the object URL for a given key.
    fn object_url(&self, key: &str) -> String {
        let endpoint = self.config.endpoint.trim_end_matches('/');
        format!("{}/{}/{}", endpoint, self.config.bucket, key)
    }

    /// Derive the AWS SigV4 signing key for the current date.
    pub(crate) fn signing_key(&self, date_stamp: &str, service: &str) -> Vec<u8> {
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
        let date_stamp = &datetime[..8];

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

        let canonical_request = format!(
            "{}\n{}\n{}\n{}\n{}\n{}",
            method, path, query, canonical_headers, signed_headers, payload_hash
        );
        let canonical_request_hash = sha256_hex(canonical_request.as_bytes());

        let credential_scope = format!("{}/{}/s3/aws4_request", date_stamp, self.config.region);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{}\n{}\n{}",
            datetime, credential_scope, canonical_request_hash
        );

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
