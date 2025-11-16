//! Model Element Resolver
//!
//! Resolves references between SAMM model elements.

use crate::error::{Result, SammError};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use url::Url;

/// Resolves SAMM model element references
pub struct ModelResolver {
    /// Models root directories for resolution
    models_roots: Vec<PathBuf>,

    /// HTTP/HTTPS base URLs for remote resolution
    remote_bases: Vec<String>,

    /// Cached resolved elements (URN -> content)
    cache: HashMap<String, String>,

    /// Cached URN to file path mappings
    path_cache: HashMap<String, PathBuf>,

    /// HTTP client for remote resolution (lazy initialized)
    http_client: Option<reqwest::Client>,

    /// HTTP request timeout in seconds
    http_timeout_secs: u64,
}

impl ModelResolver {
    /// Create a new model resolver
    pub fn new() -> Self {
        Self {
            models_roots: Vec::new(),
            remote_bases: Vec::new(),
            cache: HashMap::new(),
            path_cache: HashMap::new(),
            http_client: None,
            http_timeout_secs: 30,
        }
    }

    /// Add a models root directory
    pub fn add_models_root(&mut self, path: PathBuf) {
        self.models_roots.push(path);
    }

    /// Add a remote base URL for HTTP/HTTPS resolution
    ///
    /// Example: `https://models.example.com/samm/`
    ///
    /// The URN will be resolved to:
    /// `{base_url}/{namespace}/{version}/{element}.ttl`
    pub fn add_remote_base(&mut self, base_url: String) {
        let normalized = if base_url.ends_with('/') {
            base_url
        } else {
            format!("{}/", base_url)
        };
        self.remote_bases.push(normalized);
    }

    /// Set HTTP timeout in seconds (default: 30)
    pub fn set_http_timeout(&mut self, timeout_secs: u64) {
        self.http_timeout_secs = timeout_secs;
        // Reset HTTP client to apply new timeout
        self.http_client = None;
    }

    /// Get or create HTTP client
    fn get_http_client(&mut self) -> Result<&reqwest::Client> {
        if self.http_client.is_none() {
            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(self.http_timeout_secs))
                .build()
                .map_err(|e| SammError::Network(format!("Failed to create HTTP client: {}", e)))?;
            self.http_client = Some(client);
        }
        Ok(self.http_client.as_ref().unwrap())
    }

    /// Resolve a model element URN to a file path
    ///
    /// Follows the SAMM directory structure:
    /// `<namespace>/<version>/<element>.ttl`
    ///
    /// Example:
    /// - URN: `urn:samm:org.example:1.0.0#Movement`
    /// - Path: `<models_root>/org.example/1.0.0/Movement.ttl`
    pub fn resolve_urn(&self, urn: &str) -> Result<PathBuf> {
        // Check path cache first
        if let Some(cached_path) = self.path_cache.get(urn) {
            return Ok(cached_path.clone());
        }

        // Parse the URN
        let parts = self.parse_urn(urn)?;

        // Build relative path: namespace/version/element.ttl
        let relative_path = PathBuf::from(&parts.namespace)
            .join(&parts.version)
            .join(format!("{}.ttl", parts.element));

        // Search in all models roots
        for root in &self.models_roots {
            let full_path = root.join(&relative_path);
            if full_path.exists() {
                return Ok(full_path);
            }
        }

        Err(SammError::ResolutionError(format!(
            "Could not resolve URN '{}' in any models root directory",
            urn
        )))
    }

    /// Load and cache a model element from a URN
    ///
    /// Tries resolution in this order:
    /// 1. Check cache
    /// 2. Try file-based resolution (local models roots)
    /// 3. Try HTTP/HTTPS resolution (remote bases)
    pub async fn load_element(&mut self, urn: &str) -> Result<String> {
        // Check cache
        if let Some(cached_content) = self.cache.get(urn) {
            tracing::debug!("Cache hit for URN: {}", urn);
            return Ok(cached_content.clone());
        }

        // Try file-based resolution first
        if let Ok(file_path) = self.resolve_urn(urn) {
            let content = fs::read_to_string(&file_path).await.map_err(|e| {
                SammError::ResolutionError(format!(
                    "Failed to read file '{}': {}",
                    file_path.display(),
                    e
                ))
            })?;

            // Cache content
            self.cache.insert(urn.to_string(), content.clone());
            tracing::debug!("Loaded and cached URN: {} from {:?}", urn, file_path);

            return Ok(content);
        }

        // Try HTTP/HTTPS resolution
        if !self.remote_bases.is_empty() {
            return self.load_element_http(urn).await;
        }

        Err(SammError::ResolutionError(format!(
            "Could not resolve URN '{}' in any configured location (file or HTTP)",
            urn
        )))
    }

    /// Load a model element from HTTP/HTTPS
    async fn load_element_http(&mut self, urn: &str) -> Result<String> {
        let parts = self.parse_urn(urn)?;

        // Build relative URL path: namespace/version/element.ttl
        let relative_path = format!(
            "{}/{}/{}.ttl",
            parts.namespace, parts.version, parts.element
        );

        // Try each remote base
        let client = self.get_http_client()?.clone();

        for base_url in &self.remote_bases {
            let url = format!("{}{}", base_url, relative_path);

            tracing::debug!("Attempting HTTP resolution: {}", url);

            match client.get(&url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        match response.text().await {
                            Ok(content) => {
                                // Cache content
                                self.cache.insert(urn.to_string(), content.clone());
                                tracing::debug!("Loaded and cached URN: {} from {}", urn, url);
                                return Ok(content);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to read response body from {}: {}", url, e);
                                continue;
                            }
                        }
                    } else {
                        tracing::debug!(
                            "HTTP resolution failed for {}: status {}",
                            url,
                            response.status()
                        );
                    }
                }
                Err(e) => {
                    tracing::debug!("HTTP request failed for {}: {}", url, e);
                    continue;
                }
            }
        }

        Err(SammError::Network(format!(
            "Could not resolve URN '{}' from any remote base URL",
            urn
        )))
    }

    /// Parse a URN into its components
    fn parse_urn(&self, urn: &str) -> Result<UrnParts> {
        // Expected format: urn:samm:<namespace>:<version>#<element>
        if !urn.starts_with("urn:samm:") {
            return Err(SammError::InvalidUrn(format!(
                "URN must start with 'urn:samm:', got: {}",
                urn
            )));
        }

        // Split by '#' to separate namespace:version from element
        let parts: Vec<&str> = urn.split('#').collect();
        if parts.len() != 2 {
            return Err(SammError::InvalidUrn(format!(
                "URN must contain exactly one '#' separator, got: {}",
                urn
            )));
        }

        let element = parts[1].to_string();

        // Parse namespace and version from urn:samm:<namespace>:<version>
        let namespace_version = parts[0].strip_prefix("urn:samm:").unwrap();
        let nv_parts: Vec<&str> = namespace_version.rsplitn(2, ':').collect();

        if nv_parts.len() != 2 {
            return Err(SammError::InvalidUrn(format!(
                "URN must contain namespace and version separated by ':', got: {}",
                urn
            )));
        }

        let version = nv_parts[0].to_string();
        let namespace = nv_parts[1].to_string();

        Ok(UrnParts {
            namespace,
            version,
            element,
        })
    }

    /// Clear all caches
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.path_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            content_cache_size: self.cache.len(),
            path_cache_size: self.path_cache.len(),
        }
    }
}

/// Parsed URN components
#[derive(Debug, Clone)]
struct UrnParts {
    namespace: String,
    version: String,
    element: String,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached content entries
    pub content_cache_size: usize,
    /// Number of cached path mappings
    pub path_cache_size: usize,
}

impl Default for ModelResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_resolver_creation() {
        let resolver = ModelResolver::new();
        assert_eq!(resolver.models_roots.len(), 0);
    }

    #[test]
    fn test_add_models_root() {
        let mut resolver = ModelResolver::new();
        resolver.add_models_root(PathBuf::from("/path/to/models"));
        assert_eq!(resolver.models_roots.len(), 1);
    }

    #[test]
    fn test_parse_urn() {
        let resolver = ModelResolver::new();

        // Valid URN
        let result = resolver.parse_urn("urn:samm:org.example:1.0.0#Movement");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.namespace, "org.example");
        assert_eq!(parts.version, "1.0.0");
        assert_eq!(parts.element, "Movement");
    }

    #[test]
    fn test_parse_urn_with_nested_namespace() {
        let resolver = ModelResolver::new();

        // URN with nested namespace
        let result = resolver.parse_urn("urn:samm:org.eclipse.esmf:2.3.0#Aspect");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.namespace, "org.eclipse.esmf");
        assert_eq!(parts.version, "2.3.0");
        assert_eq!(parts.element, "Aspect");
    }

    #[test]
    fn test_parse_invalid_urn() {
        let resolver = ModelResolver::new();

        // Missing urn:samm prefix
        let result = resolver.parse_urn("org.example:1.0.0#Movement");
        assert!(result.is_err());

        // Missing # separator
        let result = resolver.parse_urn("urn:samm:org.example:1.0.0");
        assert!(result.is_err());

        // Missing version
        let result = resolver.parse_urn("urn:samm:org.example#Movement");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_resolve_urn_with_temp_file() {
        let mut resolver = ModelResolver::new();

        // Create a temporary directory structure
        let temp_dir = env::temp_dir().join("samm_resolver_test");
        let namespace_dir = temp_dir.join("org.example").join("1.0.0");
        fs::create_dir_all(&namespace_dir).await.unwrap();

        // Create a test file
        let test_file = namespace_dir.join("Movement.ttl");
        fs::write(
            &test_file,
            "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .",
        )
        .await
        .unwrap();

        // Add models root
        resolver.add_models_root(temp_dir.clone());

        // Resolve URN
        let result = resolver.resolve_urn("urn:samm:org.example:1.0.0#Movement");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_file);

        // Cleanup
        fs::remove_dir_all(temp_dir).await.unwrap();
    }

    #[tokio::test]
    async fn test_load_element_with_caching() {
        let mut resolver = ModelResolver::new();

        // Create a temporary directory structure
        let temp_dir = env::temp_dir().join("samm_resolver_cache_test");
        let namespace_dir = temp_dir.join("org.example").join("1.0.0");
        fs::create_dir_all(&namespace_dir).await.unwrap();

        // Create a test file
        let test_file = namespace_dir.join("TestAspect.ttl");
        let test_content = "@prefix samm: <urn:samm:org.eclipse.esmf.samm:meta-model:2.3.0#> .";
        fs::write(&test_file, test_content).await.unwrap();

        // Add models root
        resolver.add_models_root(temp_dir.clone());

        // Load element (first time - from file)
        let urn = "urn:samm:org.example:1.0.0#TestAspect";
        let result1 = resolver.load_element(urn).await;
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap(), test_content);

        // Check cache stats
        let stats = resolver.cache_stats();
        assert_eq!(stats.content_cache_size, 1);

        // Load element (second time - from cache)
        let result2 = resolver.load_element(urn).await;
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), test_content);

        // Cleanup
        fs::remove_dir_all(temp_dir).await.unwrap();
    }

    #[test]
    fn test_cache_stats() {
        let resolver = ModelResolver::new();
        let stats = resolver.cache_stats();
        assert_eq!(stats.content_cache_size, 0);
        assert_eq!(stats.path_cache_size, 0);
    }

    #[test]
    fn test_add_remote_base() {
        let mut resolver = ModelResolver::new();

        // Add base URL without trailing slash
        resolver.add_remote_base("https://models.example.com".to_string());
        assert_eq!(resolver.remote_bases.len(), 1);
        assert_eq!(resolver.remote_bases[0], "https://models.example.com/");

        // Add base URL with trailing slash
        resolver.add_remote_base("https://other.example.com/samm/".to_string());
        assert_eq!(resolver.remote_bases.len(), 2);
        assert_eq!(resolver.remote_bases[1], "https://other.example.com/samm/");
    }

    #[test]
    fn test_set_http_timeout() {
        let mut resolver = ModelResolver::new();
        assert_eq!(resolver.http_timeout_secs, 30); // Default

        resolver.set_http_timeout(60);
        assert_eq!(resolver.http_timeout_secs, 60);
    }

    #[tokio::test]
    async fn test_load_element_fallback_to_http() {
        let mut resolver = ModelResolver::new();

        // No local models roots configured
        // Add a mock remote base (will fail in this test, but tests the code path)
        resolver.add_remote_base("https://nonexistent.example.com/models/".to_string());

        let urn = "urn:samm:org.example:1.0.0#TestAspect";
        let result = resolver.load_element(urn).await;

        // Should fail with Network error since the URL doesn't exist
        assert!(result.is_err());
        if let Err(SammError::Network(msg)) = result {
            assert!(msg.contains("Could not resolve URN"));
        } else {
            panic!("Expected Network error");
        }
    }
}
