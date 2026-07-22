//! Server-domain configuration types: network, HTTP/TLS, datasets, services.
use crate::config::config_runtime::RateLimitConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use validator::{Validate, ValidationError};

/// Server-level settings with validation
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServerSettings {
    #[validate(range(min = 1, max = 65535))]
    pub port: u16,

    #[validate(length(min = 1))]
    pub host: String,

    pub admin_ui: bool,
    pub cors: bool,

    #[validate(range(min = 1))]
    pub max_connections: usize,

    #[validate(range(min = 1))]
    pub request_timeout_secs: u64,

    #[validate(range(min = 1))]
    pub graceful_shutdown_timeout_secs: u64,

    pub tls: Option<TlsConfig>,

    /// Directory for storing backups
    #[serde(default)]
    pub backup_directory: Option<PathBuf>,

    /// Root directory for static assets served via `/$/cdn/assets/*path`
    /// (see [`crate::handlers::production::serve_static_asset`]). When
    /// `None`, static asset serving is disabled and the route returns 404
    /// for every request instead of silently pretending to serve files.
    #[serde(default)]
    pub static_asset_dir: Option<PathBuf>,

    /// Path to the configuration file (set at runtime)
    #[serde(skip)]
    pub config_file: Option<PathBuf>,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TlsConfig {
    #[validate(custom(function = "validate_path"))]
    pub cert_path: PathBuf,

    #[validate(custom(function = "validate_path"))]
    pub key_path: PathBuf,

    pub require_client_cert: bool,

    pub ca_cert_path: Option<PathBuf>,
}

/// HTTP protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct HttpProtocolSettings {
    /// Enable HTTP/2
    #[serde(default = "default_true")]
    pub http2_enabled: bool,

    /// Enable HTTP/3 (QUIC) - experimental
    #[serde(default)]
    pub http3_enabled: bool,

    /// HTTP/2 initial connection window size (bytes)
    #[serde(default = "default_http2_connection_window")]
    #[validate(range(min = 65535, max = 16777216))]
    pub http2_initial_connection_window_size: u32,

    /// HTTP/2 initial stream window size (bytes)
    #[serde(default = "default_http2_stream_window")]
    #[validate(range(min = 65535, max = 16777216))]
    pub http2_initial_stream_window_size: u32,

    /// HTTP/2 max concurrent streams
    #[serde(default = "default_http2_max_streams")]
    #[validate(range(min = 1, max = 1000))]
    pub http2_max_concurrent_streams: u32,

    /// HTTP/2 max frame size (bytes)
    #[serde(default = "default_http2_frame_size")]
    #[validate(range(min = 16384, max = 16777215))]
    pub http2_max_frame_size: u32,

    /// HTTP/2 keep alive interval (seconds)
    #[serde(default = "default_http2_keepalive")]
    #[validate(range(min = 1))]
    pub http2_keep_alive_interval_secs: u64,

    /// HTTP/2 keep alive timeout (seconds)
    #[serde(default = "default_http2_keepalive_timeout")]
    #[validate(range(min = 1))]
    pub http2_keep_alive_timeout_secs: u64,

    /// Enable server push for SPARQL results
    #[serde(default)]
    pub enable_server_push: bool,

    /// Enable header compression (HPACK for HTTP/2, QPACK for HTTP/3)
    #[serde(default = "default_true")]
    pub enable_header_compression: bool,

    /// Optimize for SPARQL workloads
    #[serde(default = "default_true")]
    pub sparql_optimized: bool,
}

fn default_true() -> bool {
    true
}

fn default_http2_connection_window() -> u32 {
    1024 * 1024 // 1MB
}

fn default_http2_stream_window() -> u32 {
    256 * 1024 // 256KB
}

fn default_http2_max_streams() -> u32 {
    100
}

fn default_http2_frame_size() -> u32 {
    16384 // 16KB
}

fn default_http2_keepalive() -> u64 {
    60 // seconds
}

fn default_http2_keepalive_timeout() -> u64 {
    20 // seconds
}

impl Default for HttpProtocolSettings {
    fn default() -> Self {
        Self {
            http2_enabled: true,
            http3_enabled: false,
            http2_initial_connection_window_size: default_http2_connection_window(),
            http2_initial_stream_window_size: default_http2_stream_window(),
            http2_max_concurrent_streams: default_http2_max_streams(),
            http2_max_frame_size: default_http2_frame_size(),
            http2_keep_alive_interval_secs: default_http2_keepalive(),
            http2_keep_alive_timeout_secs: default_http2_keepalive_timeout(),
            enable_server_push: false,
            enable_header_compression: true,
            sparql_optimized: true,
        }
    }
}

/// Dataset configuration with validation
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DatasetConfig {
    #[validate(length(min = 1))]
    pub name: String,

    #[validate(length(min = 1))]
    pub location: String,

    pub read_only: bool,

    #[validate(nested)]
    pub text_index: Option<TextIndexConfig>,

    pub shacl_shapes: Vec<PathBuf>,

    #[validate(nested)]
    pub services: Vec<ServiceConfig>,

    #[validate(nested)]
    pub access_control: Option<AccessControlConfig>,

    pub backup: Option<BackupConfig>,
}

/// Service configuration for datasets
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ServiceConfig {
    #[validate(length(min = 1))]
    pub name: String,

    pub service_type: ServiceType,

    #[validate(length(min = 1))]
    pub endpoint: String,

    pub auth_required: bool,

    pub rate_limit: Option<RateLimitConfig>,
}

/// Types of services supported
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceType {
    SparqlQuery,
    SparqlUpdate,
    GraphStore,
    GraphQL,
    Rest,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AccessControlConfig {
    pub read_roles: Vec<String>,
    pub write_roles: Vec<String>,
    pub admin_roles: Vec<String>,
    pub public_read: bool,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct BackupConfig {
    pub enabled: bool,

    pub directory: PathBuf,

    #[validate(range(min = 1))]
    pub interval_hours: u64,

    #[validate(range(min = 1))]
    pub retain_count: usize,
}

/// Text indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TextIndexConfig {
    pub enabled: bool,

    #[validate(length(min = 1))]
    pub analyzer: String,

    #[validate(range(min = 1))]
    pub max_results: usize,

    pub stemming: bool,

    pub stop_words: Vec<String>,
}

/// Type alias for the per-dataset config map.
pub type DatasetMap = HashMap<String, DatasetConfig>;

/// Custom validation function for PathBuf
fn validate_path(path: &std::path::Path) -> Result<(), ValidationError> {
    if path.as_os_str().is_empty() {
        return Err(ValidationError::new("path_empty"));
    }
    Ok(())
}
