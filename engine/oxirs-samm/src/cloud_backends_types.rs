//! Type definitions for the SAMM cloud backend subsystem.
//!
//! Contains the `CloudBackend` enum, per-provider configuration structs
//! (`S3Config`, `GcsConfig`, `AzureConfig`, `HttpConfig`), credential types,
//! and capability / access policy enumerations shared by all backend
//! implementations.

// Re-export all per-provider config types from the specialist modules so that
// consumers can `use oxirs_samm::cloud_backends_types::*`.
pub use crate::cloud_backends_aws::S3Config;
pub use crate::cloud_backends_azure::AzureConfig;
pub use crate::cloud_backends_gcp::GcsConfig;
pub use crate::cloud_backends_http::HttpConfig;

// ──────────────────────────────────────────────────────────────────────────────
// CloudBackend enum — top-level discriminant for the backend in use
// ──────────────────────────────────────────────────────────────────────────────

/// Discriminant for the active storage backend.
///
/// Allows code that manages multiple backends to carry a single typed value
/// without erasing the concrete backend type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudBackend {
    /// AWS S3 or any S3-compatible endpoint (MinIO, Ceph, etc.)
    S3,
    /// Google Cloud Storage
    Gcs,
    /// Azure Blob Storage
    Azure,
    /// Generic HTTP REST backend
    Http,
    /// Local filesystem — useful for testing and air-gapped environments
    LocalFilesystem,
}

impl std::fmt::Display for CloudBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CloudBackend::S3 => write!(f, "S3"),
            CloudBackend::Gcs => write!(f, "GCS"),
            CloudBackend::Azure => write!(f, "AzureBlob"),
            CloudBackend::Http => write!(f, "HTTP"),
            CloudBackend::LocalFilesystem => write!(f, "LocalFilesystem"),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// StorageCapability — feature flags advertised by a backend
// ──────────────────────────────────────────────────────────────────────────────

/// Capabilities that a cloud storage backend may advertise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageCapability {
    /// Supports versioned objects / object history.
    Versioning,
    /// Supports server-side encryption at rest.
    ServerSideEncryption,
    /// Supports presigned (time-limited, unauthenticated) download URLs.
    PresignedUrls,
    /// Supports listing objects by prefix.
    PrefixListing,
    /// Supports atomic multipart uploads for large objects.
    MultipartUpload,
    /// Supports tag / metadata annotations on objects.
    ObjectTagging,
    /// Supports cross-region / cross-account replication.
    Replication,
}

// ──────────────────────────────────────────────────────────────────────────────
// AccessPolicy
// ──────────────────────────────────────────────────────────────────────────────

/// Access policy applied to objects uploaded via a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccessPolicy {
    /// Objects are accessible only to the owning account (default).
    #[default]
    Private,
    /// Objects are publicly readable.
    PublicRead,
    /// Objects are publicly readable and writable (rarely desired).
    PublicReadWrite,
    /// Objects inherit the bucket/container ACL.
    BucketOwnerFull,
}

// ──────────────────────────────────────────────────────────────────────────────
// BackendConfig — unified configuration envelope
// ──────────────────────────────────────────────────────────────────────────────

/// Unified configuration envelope that wraps a provider-specific config and
/// attaches cross-cutting concerns like the access policy and capability flags.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Which backend this config targets.
    pub backend: CloudBackend,
    /// Access policy applied to newly uploaded objects.
    pub access_policy: AccessPolicy,
    /// Maximum object size in bytes that this backend accepts per single PUT.
    /// `None` means the backend can handle arbitrarily large single-part
    /// uploads (or the limit is unknown).
    pub max_single_put_bytes: Option<u64>,
    /// Human-readable label for logging and metrics.
    pub label: String,
}

impl BackendConfig {
    /// Create a new `BackendConfig` with sensible defaults.
    pub fn new(backend: CloudBackend, label: impl Into<String>) -> Self {
        Self {
            backend,
            access_policy: AccessPolicy::Private,
            max_single_put_bytes: None,
            label: label.into(),
        }
    }
}
