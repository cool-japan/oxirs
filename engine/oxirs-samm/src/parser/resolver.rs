//! Model Element Resolver
//!
//! Resolves references between SAMM model elements.

use crate::error::{Result, SammError};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;

/// Resolves SAMM model element references
pub struct ModelResolver {
    /// Models root directories for resolution
    models_roots: Vec<PathBuf>,

    /// Cached resolved elements (URN -> content)
    cache: HashMap<String, String>,

    /// Cached URN to file path mappings
    path_cache: HashMap<String, PathBuf>,
}

impl ModelResolver {
    /// Create a new model resolver
    pub fn new() -> Self {
        Self {
            models_roots: Vec::new(),
            cache: HashMap::new(),
            path_cache: HashMap::new(),
        }
    }

    /// Add a models root directory
    pub fn add_models_root(&mut self, path: PathBuf) {
        self.models_roots.push(path);
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
    pub async fn load_element(&mut self, urn: &str) -> Result<String> {
        // Check cache
        if let Some(cached_content) = self.cache.get(urn) {
            tracing::debug!("Cache hit for URN: {}", urn);
            return Ok(cached_content.clone());
        }

        // Resolve URN to file path
        let file_path = self.resolve_urn(urn)?;

        // Load file content
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

        Ok(content)
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
}
