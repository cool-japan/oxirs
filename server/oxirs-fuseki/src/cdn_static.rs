//! CDN support for static assets in OxiRS Fuseki
//!
//! This module provides static file serving with CDN integration for the Admin UI
//! and other static assets. It integrates with the edge_caching module for cache
//! header generation and purging.

use crate::edge_caching::{CacheControl, EdgeCacheConfig, EdgeCacheManager, EdgeCacheProvider};
use crate::error::{FusekiError, FusekiResult};
use axum::body::Body;
use axum::http::{header, HeaderMap, HeaderValue, Request, Response, StatusCode};
use axum::response::IntoResponse;
use dashmap::DashMap;
use flate2::read::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, info, warn};

/// CDN static asset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnStaticConfig {
    /// Enable static file serving
    pub enabled: bool,
    /// Root directory for static files
    pub root_directory: PathBuf,
    /// URL prefix for static assets (e.g., "/static")
    pub url_prefix: String,
    /// Enable asset fingerprinting (hash-based versioning)
    pub enable_fingerprinting: bool,
    /// Enable on-the-fly compression
    pub enable_compression: bool,
    /// Compression threshold (files larger than this will be compressed)
    pub compression_threshold_bytes: usize,
    /// Maximum file size to serve (prevent DoS)
    pub max_file_size_bytes: usize,
    /// Index file for directory requests
    pub index_file: String,
    /// Enable directory listing
    pub enable_directory_listing: bool,
    /// Cache control settings for different file types
    pub cache_policies: HashMap<String, CachePolicy>,
    /// CDN origin URL (for generating absolute URLs)
    pub cdn_origin_url: Option<String>,
    /// Allowed file extensions (empty means all allowed)
    pub allowed_extensions: Vec<String>,
    /// Denied file extensions (takes precedence over allowed)
    pub denied_extensions: Vec<String>,
}

impl Default for CdnStaticConfig {
    fn default() -> Self {
        let mut cache_policies = HashMap::new();

        // Immutable assets (versioned/fingerprinted files)
        cache_policies.insert(
            "immutable".to_string(),
            CachePolicy {
                max_age: 31536000, // 1 year
                stale_while_revalidate: None,
                public: true,
                immutable: true,
            },
        );

        // JS and CSS files
        cache_policies.insert(
            "js".to_string(),
            CachePolicy {
                max_age: 86400, // 1 day
                stale_while_revalidate: Some(3600),
                public: true,
                immutable: false,
            },
        );
        cache_policies.insert(
            "css".to_string(),
            CachePolicy {
                max_age: 86400,
                stale_while_revalidate: Some(3600),
                public: true,
                immutable: false,
            },
        );

        // Images
        cache_policies.insert(
            "images".to_string(),
            CachePolicy {
                max_age: 604800, // 1 week
                stale_while_revalidate: Some(86400),
                public: true,
                immutable: false,
            },
        );

        // Fonts
        cache_policies.insert(
            "fonts".to_string(),
            CachePolicy {
                max_age: 31536000, // 1 year
                stale_while_revalidate: None,
                public: true,
                immutable: true,
            },
        );

        // HTML files
        cache_policies.insert(
            "html".to_string(),
            CachePolicy {
                max_age: 3600, // 1 hour
                stale_while_revalidate: Some(300),
                public: true,
                immutable: false,
            },
        );

        Self {
            enabled: true,
            root_directory: PathBuf::from("./static"),
            url_prefix: "/static".to_string(),
            enable_fingerprinting: true,
            enable_compression: true,
            compression_threshold_bytes: 1024,     // 1KB
            max_file_size_bytes: 10 * 1024 * 1024, // 10MB
            index_file: "index.html".to_string(),
            enable_directory_listing: false,
            cache_policies,
            cdn_origin_url: None,
            allowed_extensions: vec![],
            denied_extensions: vec![
                "exe".to_string(),
                "sh".to_string(),
                "bat".to_string(),
                "cmd".to_string(),
                "dll".to_string(),
                "so".to_string(),
            ],
        }
    }
}

/// Cache policy for a file type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePolicy {
    pub max_age: u64,
    pub stale_while_revalidate: Option<u64>,
    pub public: bool,
    pub immutable: bool,
}

impl CachePolicy {
    /// Convert to Cache-Control header value
    pub fn to_header_value(&self) -> String {
        let mut parts = Vec::new();

        if self.public {
            parts.push("public".to_string());
        } else {
            parts.push("private".to_string());
        }

        parts.push(format!("max-age={}", self.max_age));

        if let Some(swr) = self.stale_while_revalidate {
            parts.push(format!("stale-while-revalidate={}", swr));
        }

        if self.immutable {
            parts.push("immutable".to_string());
        }

        parts.join(", ")
    }
}

/// Static asset entry with metadata
#[derive(Debug, Clone)]
pub struct StaticAsset {
    /// File path relative to root
    pub path: PathBuf,
    /// Content type (MIME type)
    pub content_type: String,
    /// File size in bytes
    pub size: usize,
    /// File content hash (for fingerprinting/ETags)
    pub content_hash: String,
    /// Last modified time
    pub last_modified: SystemTime,
    /// Compressed content (if applicable)
    pub compressed_content: Option<Vec<u8>>,
    /// Original content
    pub content: Vec<u8>,
    /// Cache policy to use
    pub cache_policy: String,
}

/// CDN static asset manager
pub struct CdnStaticManager {
    config: CdnStaticConfig,
    /// Asset cache (path -> asset)
    assets: Arc<DashMap<String, StaticAsset>>,
    /// Fingerprint mapping (original path -> fingerprinted path)
    fingerprint_map: Arc<DashMap<String, String>>,
    /// Edge cache manager for CDN integration
    edge_cache: Option<Arc<EdgeCacheManager>>,
    /// Statistics
    stats: Arc<DashMap<String, StaticAssetStats>>,
}

/// Statistics for a static asset
#[derive(Debug, Clone, Default)]
pub struct StaticAssetStats {
    pub hits: u64,
    pub last_accessed: Option<Instant>,
    pub bytes_served: u64,
    pub compressed_bytes_served: u64,
}

/// CDN static manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnStaticStatistics {
    pub total_assets: usize,
    pub total_size_bytes: u64,
    pub compressed_size_bytes: u64,
    pub total_hits: u64,
    pub total_bytes_served: u64,
    pub compression_ratio: f64,
}

impl CdnStaticManager {
    /// Create a new CDN static manager
    pub fn new(config: CdnStaticConfig, edge_cache: Option<Arc<EdgeCacheManager>>) -> Self {
        Self {
            config,
            assets: Arc::new(DashMap::new()),
            fingerprint_map: Arc::new(DashMap::new()),
            edge_cache,
            stats: Arc::new(DashMap::new()),
        }
    }

    /// Initialize the manager and scan for static files
    pub fn initialize(&self) -> FusekiResult<()> {
        if !self.config.enabled {
            info!("CDN static serving is disabled");
            return Ok(());
        }

        if !self.config.root_directory.exists() {
            warn!(
                "Static root directory does not exist: {:?}",
                self.config.root_directory
            );
            return Ok(());
        }

        info!(
            "Scanning static files from: {:?}",
            self.config.root_directory
        );
        self.scan_directory(&self.config.root_directory, &self.config.root_directory)?;

        info!("Loaded {} static assets", self.assets.len());

        Ok(())
    }

    /// Recursively scan directory for static files
    fn scan_directory(&self, path: &Path, root: &Path) -> FusekiResult<()> {
        if !path.is_dir() {
            return Ok(());
        }

        let entries = std::fs::read_dir(path).map_err(|e| {
            FusekiError::configuration(format!("Failed to read directory {:?}: {}", path, e))
        })?;

        for entry in entries.flatten() {
            let entry_path = entry.path();

            if entry_path.is_dir() {
                self.scan_directory(&entry_path, root)?;
            } else if entry_path.is_file() {
                if let Err(e) = self.load_asset(&entry_path, root) {
                    warn!("Failed to load asset {:?}: {}", entry_path, e);
                }
            }
        }

        Ok(())
    }

    /// Load a single asset file
    fn load_asset(&self, path: &Path, root: &Path) -> FusekiResult<()> {
        // Get file extension
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        // Check denied extensions
        if self.config.denied_extensions.contains(&extension) {
            debug!("Skipping denied extension: {:?}", path);
            return Ok(());
        }

        // Check allowed extensions (if not empty)
        if !self.config.allowed_extensions.is_empty()
            && !self.config.allowed_extensions.contains(&extension)
        {
            debug!("Skipping non-allowed extension: {:?}", path);
            return Ok(());
        }

        // Read file content
        let content = std::fs::read(path).map_err(|e| {
            FusekiError::configuration(format!("Failed to read file {:?}: {}", path, e))
        })?;

        // Check file size
        if content.len() > self.config.max_file_size_bytes {
            warn!(
                "Skipping oversized file: {:?} ({} bytes)",
                path,
                content.len()
            );
            return Ok(());
        }

        // Compute content hash
        let content_hash = Self::compute_hash(&content);

        // Get relative path
        let relative_path = path
            .strip_prefix(root)
            .map_err(|e| FusekiError::configuration(format!("Invalid path: {}", e)))?;

        // Determine content type
        let content_type = Self::get_content_type(&extension);

        // Determine cache policy
        let cache_policy = self.determine_cache_policy(&extension);

        // Get last modified time
        let metadata = std::fs::metadata(path).map_err(|e| {
            FusekiError::configuration(format!("Failed to get metadata for {:?}: {}", path, e))
        })?;
        let last_modified = metadata.modified().unwrap_or(SystemTime::now());

        // Compress if applicable
        let compressed_content = if self.config.enable_compression
            && content.len() > self.config.compression_threshold_bytes
            && Self::is_compressible(&extension)
        {
            Some(Self::compress_content(&content)?)
        } else {
            None
        };

        let asset = StaticAsset {
            path: relative_path.to_path_buf(),
            content_type,
            size: content.len(),
            content_hash: content_hash.clone(),
            last_modified,
            compressed_content,
            content,
            cache_policy,
        };

        // Store with original path
        let url_path = format!(
            "{}/{}",
            self.config.url_prefix.trim_end_matches('/'),
            relative_path.to_string_lossy().replace('\\', "/")
        );
        self.assets.insert(url_path.clone(), asset);

        // Generate fingerprinted path if enabled
        if self.config.enable_fingerprinting {
            let fingerprinted_path = self.generate_fingerprinted_path(&url_path, &content_hash);
            self.fingerprint_map
                .insert(url_path.clone(), fingerprinted_path.clone());

            debug!("Loaded asset: {} -> {}", url_path, fingerprinted_path);
        } else {
            debug!("Loaded asset: {}", url_path);
        }

        Ok(())
    }

    /// Compute MD5 hash of content
    fn compute_hash(content: &[u8]) -> String {
        let digest = md5::compute(content);
        format!("{:x}", digest)
    }

    /// Get content type from extension
    fn get_content_type(extension: &str) -> String {
        match extension {
            // Text
            "html" | "htm" => "text/html; charset=utf-8",
            "css" => "text/css; charset=utf-8",
            "js" | "mjs" => "application/javascript; charset=utf-8",
            "json" => "application/json; charset=utf-8",
            "xml" => "application/xml; charset=utf-8",
            "txt" => "text/plain; charset=utf-8",
            "md" => "text/markdown; charset=utf-8",
            "csv" => "text/csv; charset=utf-8",

            // Images
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "svg" => "image/svg+xml",
            "ico" => "image/x-icon",
            "webp" => "image/webp",
            "avif" => "image/avif",

            // Fonts
            "woff" => "font/woff",
            "woff2" => "font/woff2",
            "ttf" => "font/ttf",
            "otf" => "font/otf",
            "eot" => "application/vnd.ms-fontobject",

            // Data formats
            "ttl" => "text/turtle; charset=utf-8",
            "nt" => "application/n-triples; charset=utf-8",
            "nq" => "application/n-quads; charset=utf-8",
            "rdf" => "application/rdf+xml; charset=utf-8",
            "sparql" => "application/sparql-query; charset=utf-8",

            // Other
            "pdf" => "application/pdf",
            "zip" => "application/zip",
            "wasm" => "application/wasm",
            "map" => "application/json", // Source maps

            _ => "application/octet-stream",
        }
        .to_string()
    }

    /// Check if content type is compressible
    fn is_compressible(extension: &str) -> bool {
        matches!(
            extension,
            "html"
                | "htm"
                | "css"
                | "js"
                | "mjs"
                | "json"
                | "xml"
                | "txt"
                | "md"
                | "csv"
                | "svg"
                | "ttl"
                | "nt"
                | "nq"
                | "rdf"
                | "sparql"
                | "map"
        )
    }

    /// Compress content using gzip
    fn compress_content(content: &[u8]) -> FusekiResult<Vec<u8>> {
        let mut encoder = GzEncoder::new(content, Compression::best());
        let mut compressed = Vec::new();
        encoder.read_to_end(&mut compressed).map_err(|e| {
            FusekiError::configuration(format!("Failed to compress content: {}", e))
        })?;
        Ok(compressed)
    }

    /// Determine cache policy for extension
    fn determine_cache_policy(&self, extension: &str) -> String {
        match extension {
            "js" | "mjs" => "js".to_string(),
            "css" => "css".to_string(),
            "html" | "htm" => "html".to_string(),
            "png" | "jpg" | "jpeg" | "gif" | "svg" | "webp" | "avif" | "ico" => {
                "images".to_string()
            }
            "woff" | "woff2" | "ttf" | "otf" | "eot" => "fonts".to_string(),
            _ => "immutable".to_string(),
        }
    }

    /// Generate fingerprinted path with content hash
    fn generate_fingerprinted_path(&self, path: &str, hash: &str) -> String {
        if let Some((base, ext)) = path.rsplit_once('.') {
            format!("{}.{}.{}", base, &hash[..8], ext)
        } else {
            format!("{}.{}", path, &hash[..8])
        }
    }

    /// Serve a static asset
    pub fn serve_asset(
        &self,
        path: &str,
        headers: &HeaderMap,
    ) -> FusekiResult<StaticAssetResponse> {
        if !self.config.enabled {
            return Err(FusekiError::not_found("Static serving is disabled"));
        }

        // Normalize path
        let normalized_path = self.normalize_path(path);

        // Try to find asset
        let asset = self
            .assets
            .get(&normalized_path)
            .ok_or_else(|| {
                // Try fingerprinted path lookup
                if let Some(original) = self.reverse_fingerprint_lookup(&normalized_path) {
                    if let Some(asset) = self.assets.get(&original) {
                        return Ok(asset);
                    }
                }
                Err(FusekiError::not_found(format!("Asset not found: {}", path)))
            })
            .map_err(|e: Result<_, _>| e.unwrap_err())?;

        // Check If-None-Match (ETag)
        if let Some(if_none_match) = headers.get(header::IF_NONE_MATCH) {
            if let Ok(etag) = if_none_match.to_str() {
                if etag.trim_matches('"') == asset.content_hash {
                    return Ok(StaticAssetResponse::NotModified);
                }
            }
        }

        // Check If-Modified-Since
        if let Some(if_modified_since) = headers.get(header::IF_MODIFIED_SINCE) {
            if let Ok(since_str) = if_modified_since.to_str() {
                if let Ok(since) = httpdate::parse_http_date(since_str) {
                    if let Ok(modified) = asset.last_modified.duration_since(SystemTime::UNIX_EPOCH)
                    {
                        if let Ok(since_duration) = since.duration_since(SystemTime::UNIX_EPOCH) {
                            if modified <= since_duration {
                                return Ok(StaticAssetResponse::NotModified);
                            }
                        }
                    }
                }
            }
        }

        // Check if client accepts gzip
        let accepts_gzip = headers
            .get(header::ACCEPT_ENCODING)
            .and_then(|v| v.to_str().ok())
            .map(|v| v.contains("gzip"))
            .unwrap_or(false);

        // Prepare response
        let (content, is_compressed) = if accepts_gzip {
            if let Some(compressed) = &asset.compressed_content {
                (compressed.clone(), true)
            } else {
                (asset.content.clone(), false)
            }
        } else {
            (asset.content.clone(), false)
        };

        // Get cache policy
        let cache_policy = self
            .config
            .cache_policies
            .get(&asset.cache_policy)
            .cloned()
            .unwrap_or(CachePolicy {
                max_age: 3600,
                stale_while_revalidate: Some(300),
                public: true,
                immutable: false,
            });

        // Update statistics
        let mut stats = self.stats.entry(normalized_path.clone()).or_default();
        stats.hits += 1;
        stats.last_accessed = Some(Instant::now());
        if is_compressed {
            stats.compressed_bytes_served += content.len() as u64;
        } else {
            stats.bytes_served += content.len() as u64;
        }

        Ok(StaticAssetResponse::Content {
            content,
            content_type: asset.content_type.clone(),
            etag: asset.content_hash.clone(),
            last_modified: asset.last_modified,
            cache_control: cache_policy.to_header_value(),
            is_compressed,
        })
    }

    /// Normalize URL path
    fn normalize_path(&self, path: &str) -> String {
        // Decode URL encoding
        let decoded = percent_encoding::percent_decode_str(path)
            .decode_utf8_lossy()
            .to_string();

        // Remove double slashes and normalize
        let normalized = decoded
            .replace("//", "/")
            .trim_start_matches('/')
            .trim_end_matches('/')
            .to_string();

        // Ensure prefix
        if normalized.starts_with(self.config.url_prefix.trim_start_matches('/')) {
            format!("/{}", normalized)
        } else {
            format!(
                "{}/{}",
                self.config.url_prefix.trim_end_matches('/'),
                normalized
            )
        }
    }

    /// Reverse lookup fingerprinted path to original
    fn reverse_fingerprint_lookup(&self, fingerprinted: &str) -> Option<String> {
        for entry in self.fingerprint_map.iter() {
            if entry.value() == fingerprinted {
                return Some(entry.key().clone());
            }
        }
        None
    }

    /// Get fingerprinted URL for an asset
    pub fn get_fingerprinted_url(&self, path: &str) -> Option<String> {
        let normalized = self.normalize_path(path);
        self.fingerprint_map.get(&normalized).map(|v| {
            let fingerprinted = v.value().clone();
            if let Some(ref origin) = self.config.cdn_origin_url {
                format!("{}{}", origin.trim_end_matches('/'), fingerprinted)
            } else {
                fingerprinted
            }
        })
    }

    /// Invalidate cache for an asset
    pub async fn invalidate_asset(&self, path: &str) -> FusekiResult<()> {
        let normalized = self.normalize_path(path);

        // Remove from local cache
        self.assets.remove(&normalized);
        self.fingerprint_map.remove(&normalized);
        self.stats.remove(&normalized);

        // Purge from edge cache
        if let Some(ref edge_cache) = self.edge_cache {
            edge_cache
                .purge_by_tags(vec![format!("static:{}", normalized)])
                .await?;
        }

        info!("Invalidated asset: {}", normalized);
        Ok(())
    }

    /// Invalidate all static assets
    pub async fn invalidate_all(&self) -> FusekiResult<()> {
        self.assets.clear();
        self.fingerprint_map.clear();
        self.stats.clear();

        // Purge from edge cache
        if let Some(ref edge_cache) = self.edge_cache {
            edge_cache
                .purge_by_tags(vec!["static:*".to_string()])
                .await?;
        }

        info!("Invalidated all static assets");
        Ok(())
    }

    /// Reload all static assets
    pub async fn reload(&self) -> FusekiResult<()> {
        self.invalidate_all().await?;
        self.initialize()?;
        info!("Reloaded all static assets");
        Ok(())
    }

    /// Get statistics
    pub fn get_statistics(&self) -> CdnStaticStatistics {
        let mut total_size = 0u64;
        let mut compressed_size = 0u64;
        let mut total_hits = 0u64;
        let mut total_bytes = 0u64;

        for entry in self.assets.iter() {
            total_size += entry.size as u64;
            if let Some(ref compressed) = entry.compressed_content {
                compressed_size += compressed.len() as u64;
            }
        }

        for entry in self.stats.iter() {
            total_hits += entry.hits;
            total_bytes += entry.bytes_served + entry.compressed_bytes_served;
        }

        let compression_ratio = if total_size > 0 && compressed_size > 0 {
            1.0 - (compressed_size as f64 / total_size as f64)
        } else {
            0.0
        };

        CdnStaticStatistics {
            total_assets: self.assets.len(),
            total_size_bytes: total_size,
            compressed_size_bytes: compressed_size,
            total_hits,
            total_bytes_served: total_bytes,
            compression_ratio,
        }
    }

    /// List all assets
    pub fn list_assets(&self) -> Vec<StaticAssetInfo> {
        self.assets
            .iter()
            .map(|entry| {
                let stats = self.stats.get(entry.key()).map(|s| s.clone());
                StaticAssetInfo {
                    path: entry.path.to_string_lossy().to_string(),
                    url: entry.key().clone(),
                    fingerprinted_url: self.fingerprint_map.get(entry.key()).map(|v| v.clone()),
                    content_type: entry.content_type.clone(),
                    size: entry.size,
                    compressed_size: entry.compressed_content.as_ref().map(|c| c.len()),
                    hits: stats.as_ref().map(|s| s.hits).unwrap_or(0),
                }
            })
            .collect()
    }
}

/// Static asset response type
pub enum StaticAssetResponse {
    /// 304 Not Modified
    NotModified,
    /// Full content response
    Content {
        content: Vec<u8>,
        content_type: String,
        etag: String,
        last_modified: SystemTime,
        cache_control: String,
        is_compressed: bool,
    },
}

impl IntoResponse for StaticAssetResponse {
    fn into_response(self) -> axum::response::Response {
        match self {
            StaticAssetResponse::NotModified => Response::builder()
                .status(StatusCode::NOT_MODIFIED)
                .body(Body::empty())
                .expect("response body build should succeed"),
            StaticAssetResponse::Content {
                content,
                content_type,
                etag,
                last_modified,
                cache_control,
                is_compressed,
            } => {
                let mut builder = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, content_type)
                    .header(header::ETAG, format!("\"{}\"", etag))
                    .header(header::CACHE_CONTROL, cache_control)
                    .header("X-Content-Type-Options", "nosniff");

                // Add Last-Modified header
                if let Ok(duration) = last_modified.duration_since(SystemTime::UNIX_EPOCH) {
                    let datetime = httpdate::fmt_http_date(SystemTime::UNIX_EPOCH + duration);
                    builder = builder.header(header::LAST_MODIFIED, datetime);
                }

                // Add Content-Encoding if compressed
                if is_compressed {
                    builder = builder.header(header::CONTENT_ENCODING, "gzip");
                }

                builder
                    .header(header::CONTENT_LENGTH, content.len())
                    .body(Body::from(content))
                    .expect("response body build should succeed")
            }
        }
    }
}

/// Static asset info for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticAssetInfo {
    pub path: String,
    pub url: String,
    pub fingerprinted_url: Option<String>,
    pub content_type: String,
    pub size: usize,
    pub compressed_size: Option<usize>,
    pub hits: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_manager() -> (CdnStaticManager, TempDir) {
        let temp_dir = TempDir::new().unwrap();

        // Create some test files
        let html_path = temp_dir.path().join("index.html");
        let css_path = temp_dir.path().join("style.css");
        let js_path = temp_dir.path().join("app.js");

        std::fs::write(&html_path, "<html><body>Test</body></html>").unwrap();
        std::fs::write(&css_path, "body { color: red; }").unwrap();
        std::fs::write(&js_path, "console.log('hello');").unwrap();

        let config = CdnStaticConfig {
            enabled: true,
            root_directory: temp_dir.path().to_path_buf(),
            ..Default::default()
        };

        let manager = CdnStaticManager::new(config, None);
        manager.initialize().unwrap();

        (manager, temp_dir)
    }

    #[test]
    fn test_cdn_static_manager_creation() {
        let config = CdnStaticConfig::default();
        let manager = CdnStaticManager::new(config, None);
        assert!(manager.assets.is_empty());
    }

    #[test]
    fn test_content_type_detection() {
        assert_eq!(
            CdnStaticManager::get_content_type("html"),
            "text/html; charset=utf-8"
        );
        assert_eq!(
            CdnStaticManager::get_content_type("css"),
            "text/css; charset=utf-8"
        );
        assert_eq!(
            CdnStaticManager::get_content_type("js"),
            "application/javascript; charset=utf-8"
        );
        assert_eq!(CdnStaticManager::get_content_type("png"), "image/png");
        assert_eq!(CdnStaticManager::get_content_type("woff2"), "font/woff2");
    }

    #[test]
    fn test_hash_computation() {
        let content = b"Hello, World!";
        let hash = CdnStaticManager::compute_hash(content);
        assert_eq!(hash.len(), 32); // MD5 produces 16 bytes = 32 hex chars
    }

    #[test]
    fn test_compressibility_check() {
        assert!(CdnStaticManager::is_compressible("html"));
        assert!(CdnStaticManager::is_compressible("css"));
        assert!(CdnStaticManager::is_compressible("js"));
        assert!(CdnStaticManager::is_compressible("json"));
        assert!(CdnStaticManager::is_compressible("svg"));
        assert!(!CdnStaticManager::is_compressible("png"));
        assert!(!CdnStaticManager::is_compressible("jpg"));
        assert!(!CdnStaticManager::is_compressible("woff2"));
    }

    #[test]
    fn test_cache_policy_header() {
        let policy = CachePolicy {
            max_age: 3600,
            stale_while_revalidate: Some(300),
            public: true,
            immutable: false,
        };

        let header = policy.to_header_value();
        assert!(header.contains("public"));
        assert!(header.contains("max-age=3600"));
        assert!(header.contains("stale-while-revalidate=300"));
        assert!(!header.contains("immutable"));
    }

    #[test]
    fn test_cache_policy_immutable() {
        let policy = CachePolicy {
            max_age: 31536000,
            stale_while_revalidate: None,
            public: true,
            immutable: true,
        };

        let header = policy.to_header_value();
        assert!(header.contains("immutable"));
    }

    #[test]
    fn test_asset_loading() {
        let (manager, _temp_dir) = create_test_manager();

        // Should have loaded 3 files
        assert_eq!(manager.assets.len(), 3);

        // Check that assets were loaded correctly
        let assets = manager.list_assets();
        assert_eq!(assets.len(), 3);

        // Verify content types
        for asset in &assets {
            if asset.path.ends_with(".html") {
                assert!(asset.content_type.contains("text/html"));
            } else if asset.path.ends_with(".css") {
                assert!(asset.content_type.contains("text/css"));
            } else if asset.path.ends_with(".js") {
                assert!(asset.content_type.contains("javascript"));
            }
        }
    }

    #[test]
    fn test_fingerprint_generation() {
        let config = CdnStaticConfig::default();
        let manager = CdnStaticManager::new(config, None);

        let path = "/static/app.js";
        let hash = "abc12345def67890";

        let fingerprinted = manager.generate_fingerprinted_path(path, hash);
        assert_eq!(fingerprinted, "/static/app.abc12345.js");
    }

    #[test]
    fn test_serve_asset() {
        let (manager, _temp_dir) = create_test_manager();

        let headers = HeaderMap::new();
        let result = manager.serve_asset("/static/index.html", &headers);

        assert!(result.is_ok());
        match result.unwrap() {
            StaticAssetResponse::Content { content_type, .. } => {
                assert!(content_type.contains("text/html"));
            }
            _ => panic!("Expected content response"),
        }
    }

    #[test]
    fn test_serve_asset_not_found() {
        let (manager, _temp_dir) = create_test_manager();

        let headers = HeaderMap::new();
        let result = manager.serve_asset("/static/nonexistent.html", &headers);

        assert!(result.is_err());
    }

    #[test]
    fn test_etag_handling() {
        let (manager, _temp_dir) = create_test_manager();

        // First request to get ETag
        let headers = HeaderMap::new();
        let first_response = manager.serve_asset("/static/index.html", &headers).unwrap();

        let etag = match first_response {
            StaticAssetResponse::Content { etag, .. } => etag,
            _ => panic!("Expected content response"),
        };

        // Second request with If-None-Match
        let mut headers = HeaderMap::new();
        headers.insert(
            header::IF_NONE_MATCH,
            HeaderValue::from_str(&format!("\"{}\"", etag)).unwrap(),
        );

        let second_response = manager.serve_asset("/static/index.html", &headers).unwrap();

        assert!(matches!(second_response, StaticAssetResponse::NotModified));
    }

    #[test]
    fn test_statistics() {
        let (manager, _temp_dir) = create_test_manager();

        let stats = manager.get_statistics();
        assert_eq!(stats.total_assets, 3);
        assert!(stats.total_size_bytes > 0);
    }

    #[test]
    fn test_denied_extensions() {
        let temp_dir = TempDir::new().unwrap();

        // Create a denied file type
        let exe_path = temp_dir.path().join("malware.exe");
        std::fs::write(&exe_path, "fake executable").unwrap();

        let config = CdnStaticConfig {
            enabled: true,
            root_directory: temp_dir.path().to_path_buf(),
            denied_extensions: vec!["exe".to_string()],
            ..Default::default()
        };

        let manager = CdnStaticManager::new(config, None);
        manager.initialize().unwrap();

        // Should not have loaded the exe file
        assert_eq!(manager.assets.len(), 0);
    }

    #[test]
    fn test_compression() {
        let temp_dir = TempDir::new().unwrap();

        // Create a file large enough to compress
        let large_content = "x".repeat(10000);
        let js_path = temp_dir.path().join("large.js");
        std::fs::write(&js_path, &large_content).unwrap();

        let config = CdnStaticConfig {
            enabled: true,
            root_directory: temp_dir.path().to_path_buf(),
            enable_compression: true,
            compression_threshold_bytes: 1024,
            ..Default::default()
        };

        let manager = CdnStaticManager::new(config, None);
        manager.initialize().unwrap();

        let assets = manager.list_assets();
        assert_eq!(assets.len(), 1);

        // Should have compressed content
        assert!(assets[0].compressed_size.is_some());
        assert!(assets[0].compressed_size.unwrap() < assets[0].size);
    }
}
