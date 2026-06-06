//! Azure Blob Storage backend.

use crate::cloud_backends_common::{hmac_sha256, parse_azure_list_xml, url_encode};
use crate::cloud_storage::CloudStorageBackend;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::Utc;
use reqwest::{Client, StatusCode};
use tracing::debug;

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

    pub(crate) fn validate(&self) -> Result<(), String> {
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
    pub(crate) fn shared_key_lite_auth(
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
        let client = crate::cloud_backends_common::build_tls_client()?;
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
