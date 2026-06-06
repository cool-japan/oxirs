//! Real Cloud Storage Backends for SAMM Models
//!
//! This module provides concrete implementations of the [`CloudStorageBackend`]
//! trait for major cloud storage providers:
//!
//! - **[`S3Backend`]**: AWS S3 and S3-compatible stores (MinIO, Ceph, etc.)
//! - **[`GcsBackend`]**: Google Cloud Storage via JSON API
//! - **[`AzureBlobBackend`]**: Azure Blob Storage via REST API
//! - **[`HttpBackend`]**: Generic HTTP REST backend (useful for testing and custom servers)
//! - **[`LocalFsBackend`]**: Local filesystem adapter (testing and air-gapped deployments)
//!
//! All backends implement [`CloudStorageBackend`] using async/await and `reqwest`.
//!
//! # Module layout
//!
//! This file is a thin facade. The implementation is split across sibling
//! modules and re-exported here so existing imports of
//! `oxirs_samm::cloud_backends::*` continue to work unchanged:
//!
//! - [`crate::cloud_backends_common`]: HMAC/SHA-256, hex, URL and XML helpers
//! - [`crate::cloud_backends_types`]: config envelopes and capability enums
//! - [`crate::cloud_backends_aws`]: [`S3Config`] / [`S3Backend`]
//! - [`crate::cloud_backends_gcp`]: [`GcsConfig`] / [`GcsBackend`]
//! - [`crate::cloud_backends_azure`]: [`AzureConfig`] / [`AzureBlobBackend`]
//! - [`crate::cloud_backends_http`]: [`HttpConfig`] / [`HttpBackend`]
//! - [`crate::cloud_backends_impl`]: backend aggregation and [`LocalFsBackend`]
//! - [`crate::cloud_backends_sync`]: multi-backend replication
//!
//! [`CloudStorageBackend`]: crate::cloud_storage::CloudStorageBackend
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

// Re-export the full public surface from the sibling modules so that callers
// using `oxirs_samm::cloud_backends::*` keep working exactly as before.
pub use crate::cloud_backends_aws::*;
pub use crate::cloud_backends_azure::*;
pub use crate::cloud_backends_common::*;
pub use crate::cloud_backends_gcp::*;
pub use crate::cloud_backends_http::*;
pub use crate::cloud_backends_impl::*;
pub use crate::cloud_backends_sync::*;
pub use crate::cloud_backends_types::*;
