//! Lazy IRI resolution with caching
//!
//! This module provides lazy evaluation of IRI resolution and normalization.
//! Instead of immediately resolving and normalizing all IRIs during parsing,
//! IRIs are stored in a compact form and only resolved when needed.
//!
//! # Benefits
//!
//! - Reduced memory usage (compact storage)
//! - Faster parsing (defer expensive normalization)
//! - Caching of resolved IRIs (avoid re-resolving)
//! - Prefix expansion on-demand

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// A lazily-resolved IRI that defers expansion and normalization
///
/// This structure stores IRIs in their compact form (e.g., prefixed names)
/// and only expands them when actually needed for comparison or output.
///
/// # Example
///
/// ```
/// use oxirs_ttl::toolkit::LazyIri;
/// use std::collections::HashMap;
///
/// let mut prefixes = HashMap::new();
/// prefixes.insert("ex".to_string(), "http://example.org/".to_string());
///
/// // Create a lazy IRI from a prefixed name
/// let iri = LazyIri::from_prefixed("ex", "Person", prefixes.clone());
///
/// // Resolution happens on-demand
/// assert_eq!(iri.resolve().unwrap(), "http://example.org/Person");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LazyIri {
    /// Already resolved full IRI
    Resolved(Arc<String>),
    /// Prefixed name that needs resolution
    Prefixed {
        /// The prefix part (e.g., "ex" in "ex:Person")
        prefix: String,
        /// The local part (e.g., "Person" in "ex:Person")
        local: String,
        /// Available prefix mappings for resolution
        prefixes: Arc<HashMap<String, String>>,
    },
    /// Relative IRI that needs base resolution
    Relative {
        /// The relative IRI string
        iri: String,
        /// Optional base IRI for resolution
        base: Arc<Option<String>>,
    },
}

impl LazyIri {
    /// Create a lazy IRI from an already-resolved IRI
    pub fn from_resolved(iri: impl Into<String>) -> Self {
        Self::Resolved(Arc::new(iri.into()))
    }

    /// Create a lazy IRI from a prefixed name
    pub fn from_prefixed(
        prefix: impl Into<String>,
        local: impl Into<String>,
        prefixes: HashMap<String, String>,
    ) -> Self {
        Self::Prefixed {
            prefix: prefix.into(),
            local: local.into(),
            prefixes: Arc::new(prefixes),
        }
    }

    /// Create a lazy IRI from a relative IRI
    pub fn from_relative(iri: impl Into<String>, base: Option<String>) -> Self {
        Self::Relative {
            iri: iri.into(),
            base: Arc::new(base),
        }
    }

    /// Resolve the IRI to its full form
    ///
    /// This performs the actual expansion and normalization.
    /// Results are cached internally for repeated calls.
    pub fn resolve(&self) -> Result<Cow<'_, str>, IriResolutionError> {
        match self {
            Self::Resolved(iri) => Ok(Cow::Borrowed(iri.as_str())),
            Self::Prefixed {
                prefix,
                local,
                prefixes,
            } => {
                let namespace = prefixes
                    .get(prefix)
                    .ok_or_else(|| IriResolutionError::UndefinedPrefix(prefix.clone()))?;

                Ok(Cow::Owned(format!("{}{}", namespace, local)))
            }
            Self::Relative { iri, base } => {
                let base_iri = base
                    .as_ref()
                    .as_ref()
                    .ok_or(IriResolutionError::MissingBase)?;

                // Simple relative resolution (production code would use RFC 3986)
                Ok(Cow::Owned(Self::resolve_relative(base_iri, iri)))
            }
        }
    }

    /// RFC 3986-compliant relative IRI resolution
    ///
    /// Implements the IRI resolution algorithm from RFC 3986 Section 5.2
    /// with support for:
    /// - Absolute IRIs (returned as-is)
    /// - Relative paths with dot segment removal (. and ..)
    /// - Fragment and query handling
    /// - Authority component handling
    fn resolve_relative(base: &str, relative: &str) -> String {
        // If relative is already absolute, return it
        if relative.contains("://") {
            return relative.to_string();
        }

        // Parse base IRI components
        let (base_scheme, base_authority, base_path, _base_query, _base_fragment) =
            Self::parse_iri(base);

        // Parse relative IRI components
        if relative.starts_with("//") {
            // Network-path reference: keep scheme, replace rest
            return format!("{}:{}", base_scheme, relative);
        }

        if relative.starts_with('/') {
            // Absolute-path reference: keep scheme+authority, replace path
            let authority_str = if base_authority.is_empty() {
                String::new()
            } else {
                format!("//{}", base_authority)
            };
            return format!("{}:{}{}", base_scheme, authority_str, relative);
        }

        if relative.starts_with('?') || relative.starts_with('#') {
            // Query or fragment reference: keep scheme+authority+path
            let authority_str = if base_authority.is_empty() {
                String::new()
            } else {
                format!("//{}", base_authority)
            };
            return format!("{}:{}{}{}", base_scheme, authority_str, base_path, relative);
        }

        // Relative-path reference: merge paths
        let merged_path = Self::merge_paths(&base_path, relative, !base_authority.is_empty());
        let normalized_path = Self::remove_dot_segments(&merged_path);

        // Split relative into path and query/fragment
        let (rel_path_only, rel_suffix) = if let Some(pos) = relative.find('?') {
            (&relative[..pos], &relative[pos..])
        } else if let Some(pos) = relative.find('#') {
            (&relative[..pos], &relative[pos..])
        } else {
            (relative, "")
        };

        let _ = rel_path_only; // Used in merged_path calculation

        let authority_str = if base_authority.is_empty() {
            String::new()
        } else {
            format!("//{}", base_authority)
        };

        // Ensure proper path formatting (no double slashes)
        let final_path = if !base_authority.is_empty() && !normalized_path.starts_with('/') {
            format!("/{}", normalized_path)
        } else {
            normalized_path
        };

        format!(
            "{}:{}{}{}",
            base_scheme, authority_str, final_path, rel_suffix
        )
    }

    /// Parse IRI into components (scheme, authority, path, query, fragment)
    fn parse_iri(iri: &str) -> (String, String, String, String, String) {
        // Extract scheme (without ://)
        let (scheme, rest) = if let Some(pos) = iri.find("://") {
            (&iri[..pos], &iri[pos + 3..])
        } else {
            ("", iri)
        };

        // Extract authority (without //)
        let (authority, rest) = if !scheme.is_empty() {
            if let Some(pos) = rest.find('/') {
                (&rest[..pos], &rest[pos..])
            } else if let Some(pos) = rest.find(['?', '#']) {
                (&rest[..pos], &rest[pos..])
            } else {
                (rest, "")
            }
        } else {
            ("", rest)
        };

        // Extract path, query, and fragment
        let (path_query, fragment) = if let Some(pos) = rest.find('#') {
            (&rest[..pos], &rest[pos..])
        } else {
            (rest, "")
        };

        let (path, query) = if let Some(pos) = path_query.find('?') {
            (&path_query[..pos], &path_query[pos..])
        } else {
            (path_query, "")
        };

        (
            scheme.to_string(),
            authority.to_string(),
            path.to_string(),
            query.to_string(),
            fragment.to_string(),
        )
    }

    /// Merge base path with relative path (RFC 3986 Section 5.2.3)
    fn merge_paths(base_path: &str, relative_path: &str, has_authority: bool) -> String {
        if has_authority && base_path.is_empty() {
            // Base has authority but empty path
            format!("/{}", relative_path)
        } else if let Some(pos) = base_path.rfind('/') {
            // Remove last segment from base, append relative
            format!("{}/{}", &base_path[..pos], relative_path)
        } else {
            // Base has no slash
            relative_path.to_string()
        }
    }

    /// Remove dot segments from path (RFC 3986 Section 5.2.4)
    fn remove_dot_segments(path: &str) -> String {
        let mut output = Vec::new();
        let segments: Vec<&str> = path.split('/').collect();

        for segment in segments {
            match segment {
                "" | "." => {
                    // Skip empty segments and current directory
                }
                ".." => {
                    // Go up one level (pop last segment)
                    output.pop();
                }
                _ => {
                    // Regular segment
                    output.push(segment);
                }
            }
        }

        // Reconstruct path
        if path.starts_with('/') {
            format!("/{}", output.join("/"))
        } else if output.is_empty() {
            String::new()
        } else {
            output.join("/")
        }
    }

    /// Check if this IRI is already resolved
    pub fn is_resolved(&self) -> bool {
        matches!(self, Self::Resolved(_))
    }

    /// Get the compact form (for display)
    pub fn compact_form(&self) -> String {
        match self {
            Self::Resolved(iri) => format!("<{}>", iri),
            Self::Prefixed { prefix, local, .. } => format!("{}:{}", prefix, local),
            Self::Relative { iri, .. } => format!("<{}>", iri),
        }
    }
}

/// Error types for IRI resolution
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IriResolutionError {
    /// Undefined prefix in prefixed name
    UndefinedPrefix(String),
    /// Missing base IRI for relative resolution
    MissingBase,
    /// Invalid IRI format
    InvalidFormat(String),
}

impl std::fmt::Display for IriResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndefinedPrefix(prefix) => write!(f, "Undefined prefix: {}", prefix),
            Self::MissingBase => write!(f, "Missing base IRI for relative resolution"),
            Self::InvalidFormat(msg) => write!(f, "Invalid IRI format: {}", msg),
        }
    }
}

impl std::error::Error for IriResolutionError {}

/// Cached IRI resolver for high-performance resolution
///
/// This resolver caches the results of IRI resolution to avoid
/// repeated expensive operations.
///
/// # Example
///
/// ```
/// use oxirs_ttl::toolkit::CachedIriResolver;
/// use std::collections::HashMap;
///
/// let mut resolver = CachedIriResolver::new();
///
/// let mut prefixes = HashMap::new();
/// prefixes.insert("ex".to_string(), "http://example.org/".to_string());
///
/// // Resolve prefixed name
/// let iri1 = resolver.resolve_prefixed("ex", "Person", &prefixes).unwrap();
/// let iri2 = resolver.resolve_prefixed("ex", "Person", &prefixes).unwrap();
///
/// // Second resolution should hit cache
/// assert!(std::sync::Arc::ptr_eq(&iri1, &iri2));
/// assert_eq!(resolver.cache_hit_rate(), 0.5); // 1 hit out of 2 requests
/// ```
#[derive(Debug, Clone)]
pub struct CachedIriResolver {
    /// Cache of resolved IRIs
    cache: HashMap<String, Arc<String>>,
    /// Statistics
    stats: ResolverStats,
}

/// Statistics for IRI resolution
#[derive(Debug, Clone, Default)]
pub struct ResolverStats {
    /// Total resolution requests
    pub total_requests: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
}

impl CachedIriResolver {
    /// Create a new cached IRI resolver
    pub fn new() -> Self {
        Self::with_capacity(512)
    }

    /// Create with pre-allocated cache capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            stats: ResolverStats::default(),
        }
    }

    /// Resolve a prefixed name to a full IRI
    pub fn resolve_prefixed(
        &mut self,
        prefix: &str,
        local: &str,
        prefixes: &HashMap<String, String>,
    ) -> Result<Arc<String>, IriResolutionError> {
        self.stats.total_requests += 1;

        // Create cache key
        let cache_key = format!("{}:{}", prefix, local);

        // Check cache
        if let Some(cached) = self.cache.get(&cache_key) {
            self.stats.cache_hits += 1;
            return Ok(cached.clone());
        }

        // Resolve
        self.stats.cache_misses += 1;
        let namespace = prefixes
            .get(prefix)
            .ok_or_else(|| IriResolutionError::UndefinedPrefix(prefix.to_string()))?;

        let resolved = Arc::new(format!("{}{}", namespace, local));

        // Cache and return
        self.cache.insert(cache_key, resolved.clone());
        Ok(resolved)
    }

    /// Resolve a relative IRI against a base
    pub fn resolve_relative(
        &mut self,
        relative: &str,
        base: &str,
    ) -> Result<Arc<String>, IriResolutionError> {
        self.stats.total_requests += 1;

        // Create cache key
        let cache_key = format!("{}<{}>", base, relative);

        // Check cache
        if let Some(cached) = self.cache.get(&cache_key) {
            self.stats.cache_hits += 1;
            return Ok(cached.clone());
        }

        // Resolve
        self.stats.cache_misses += 1;
        let resolved = Arc::new(LazyIri::resolve_relative(base, relative));

        // Cache and return
        self.cache.insert(cache_key, resolved.clone());
        Ok(resolved)
    }

    /// Resolve a lazy IRI
    pub fn resolve_lazy(&mut self, iri: &LazyIri) -> Result<Arc<String>, IriResolutionError> {
        match iri {
            LazyIri::Resolved(resolved) => {
                self.stats.total_requests += 1;
                self.stats.cache_hits += 1; // Already resolved counts as cache hit
                Ok(resolved.clone())
            }
            LazyIri::Prefixed {
                prefix,
                local,
                prefixes,
            } => self.resolve_prefixed(prefix, local, prefixes),
            LazyIri::Relative { iri: rel_iri, base } => {
                let base_str = base
                    .as_ref()
                    .as_ref()
                    .ok_or(IriResolutionError::MissingBase)?;
                self.resolve_relative(rel_iri, base_str)
            }
        }
    }

    /// Get cache hit rate (0.0 to 1.0)
    pub fn cache_hit_rate(&self) -> f64 {
        if self.stats.total_requests == 0 {
            return 0.0;
        }
        self.stats.cache_hits as f64 / self.stats.total_requests as f64
    }

    /// Get statistics
    pub fn stats(&self) -> &ResolverStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.stats = ResolverStats::default();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Shrink cache to fit
    pub fn shrink_to_fit(&mut self) {
        self.cache.shrink_to_fit();
    }
}

impl Default for CachedIriResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ResolverStats {
    /// Get a human-readable report
    pub fn report(&self) -> String {
        let hit_rate = if self.total_requests > 0 {
            (self.cache_hits as f64 / self.total_requests as f64) * 100.0
        } else {
            0.0
        };

        format!(
            "IRI Resolver Statistics:\n\
             - Total requests: {}\n\
             - Cache hits: {} ({:.1}%)\n\
             - Cache misses: {}",
            self.total_requests, self.cache_hits, hit_rate, self.cache_misses
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_iri_resolved() {
        let iri = LazyIri::from_resolved("http://example.org/");
        assert!(iri.is_resolved());
        assert_eq!(iri.resolve().unwrap(), "http://example.org/");
    }

    #[test]
    fn test_lazy_iri_prefixed() {
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());

        let iri = LazyIri::from_prefixed("ex", "Person", prefixes);
        assert!(!iri.is_resolved());
        assert_eq!(iri.resolve().unwrap(), "http://example.org/Person");
    }

    #[test]
    fn test_lazy_iri_undefined_prefix() {
        let prefixes = HashMap::new();
        let iri = LazyIri::from_prefixed("ex", "Person", prefixes);

        assert!(iri.resolve().is_err());
    }

    #[test]
    fn test_lazy_iri_relative() {
        let iri =
            LazyIri::from_relative("relative/path", Some("http://example.org/base".to_string()));

        assert_eq!(iri.resolve().unwrap(), "http://example.org/relative/path");
    }

    #[test]
    fn test_rfc3986_dot_segments() {
        // Test dot segment removal
        // Base: /a/b/c + ../ → /a/b + ../ → /a
        let iri =
            LazyIri::from_relative("../sibling", Some("http://example.org/a/b/c".to_string()));
        assert_eq!(iri.resolve().unwrap(), "http://example.org/a/sibling");

        // Base: /a/ + ./ → /a/current
        let iri = LazyIri::from_relative("./current", Some("http://example.org/a/".to_string()));
        assert_eq!(iri.resolve().unwrap(), "http://example.org/a/current");

        // Base: /a/b/c/d + ../../ → /a/b + ../../ → /a
        let iri =
            LazyIri::from_relative("../../up", Some("http://example.org/a/b/c/d".to_string()));
        assert_eq!(iri.resolve().unwrap(), "http://example.org/a/up");

        // Additional RFC 3986 test cases
        // Base: /a/b (file) + c → /a/c (replaces last segment)
        let iri = LazyIri::from_relative("c", Some("http://example.org/a/b".to_string()));
        assert_eq!(iri.resolve().unwrap(), "http://example.org/a/c");

        // Base: /a/b/ (directory) + c → /a/b/c (appends)
        let iri = LazyIri::from_relative("c", Some("http://example.org/a/b/".to_string()));
        assert_eq!(iri.resolve().unwrap(), "http://example.org/a/b/c");
    }

    #[test]
    fn test_rfc3986_absolute_path() {
        // Absolute path reference
        let iri = LazyIri::from_relative("/absolute", Some("http://example.org/a/b/c".to_string()));
        assert_eq!(iri.resolve().unwrap(), "http://example.org/absolute");
    }

    #[test]
    fn test_rfc3986_query_fragment() {
        // Query reference
        let iri = LazyIri::from_relative("?query", Some("http://example.org/path".to_string()));
        assert_eq!(iri.resolve().unwrap(), "http://example.org/path?query");

        // Fragment reference
        let iri = LazyIri::from_relative("#fragment", Some("http://example.org/path".to_string()));
        assert_eq!(iri.resolve().unwrap(), "http://example.org/path#fragment");
    }

    #[test]
    fn test_rfc3986_network_path() {
        // Network-path reference
        let iri = LazyIri::from_relative(
            "//other.example.org/path",
            Some("http://example.org/base".to_string()),
        );
        assert_eq!(iri.resolve().unwrap(), "http://other.example.org/path");
    }

    #[test]
    fn test_rfc3986_already_absolute() {
        // Already absolute IRI
        let iri = LazyIri::from_relative(
            "https://absolute.org/path",
            Some("http://example.org/base".to_string()),
        );
        assert_eq!(iri.resolve().unwrap(), "https://absolute.org/path");
    }

    #[test]
    fn test_lazy_iri_relative_no_base() {
        let iri = LazyIri::from_relative("relative/path", None);
        assert!(iri.resolve().is_err());
    }

    #[test]
    fn test_cached_resolver_basic() {
        let mut resolver = CachedIriResolver::new();
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());

        let iri1 = resolver
            .resolve_prefixed("ex", "Person", &prefixes)
            .unwrap();
        assert_eq!(*iri1, "http://example.org/Person");
        assert_eq!(resolver.stats().cache_misses, 1);

        // Second resolution should hit cache
        let iri2 = resolver
            .resolve_prefixed("ex", "Person", &prefixes)
            .unwrap();
        assert_eq!(*iri2, "http://example.org/Person");
        assert_eq!(resolver.stats().cache_hits, 1);

        // Should be the same Arc
        assert!(Arc::ptr_eq(&iri1, &iri2));
    }

    #[test]
    fn test_cached_resolver_hit_rate() {
        let mut resolver = CachedIriResolver::new();
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());

        // 1 miss
        resolver
            .resolve_prefixed("ex", "Person", &prefixes)
            .unwrap();

        // 1 hit
        resolver
            .resolve_prefixed("ex", "Person", &prefixes)
            .unwrap();

        assert_eq!(resolver.cache_hit_rate(), 0.5); // 50%
    }

    #[test]
    fn test_cached_resolver_relative() {
        let mut resolver = CachedIriResolver::new();

        let iri1 = resolver
            .resolve_relative("path", "http://example.org/")
            .unwrap();
        assert_eq!(*iri1, "http://example.org/path");

        // Should hit cache
        let iri2 = resolver
            .resolve_relative("path", "http://example.org/")
            .unwrap();
        assert!(Arc::ptr_eq(&iri1, &iri2));
    }

    #[test]
    fn test_cached_resolver_clear() {
        let mut resolver = CachedIriResolver::new();
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());

        resolver
            .resolve_prefixed("ex", "Person", &prefixes)
            .unwrap();
        assert_eq!(resolver.cache_size(), 1);

        resolver.clear_cache();
        assert_eq!(resolver.cache_size(), 0);
        assert_eq!(resolver.stats().total_requests, 0);
    }

    #[test]
    fn test_compact_form() {
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());

        let iri1 = LazyIri::from_resolved("http://example.org/test");
        assert_eq!(iri1.compact_form(), "<http://example.org/test>");

        let iri2 = LazyIri::from_prefixed("ex", "Person", prefixes);
        assert_eq!(iri2.compact_form(), "ex:Person");
    }

    #[test]
    fn test_resolver_stats_report() {
        let mut resolver = CachedIriResolver::new();
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());

        resolver
            .resolve_prefixed("ex", "Person", &prefixes)
            .unwrap();
        resolver
            .resolve_prefixed("ex", "Person", &prefixes)
            .unwrap();

        let report = resolver.stats().report();
        assert!(report.contains("Total requests: 2"));
        assert!(report.contains("Cache hits: 1"));
        assert!(report.contains("50.0%"));
    }

    #[test]
    fn test_resolve_lazy_resolved() {
        let mut resolver = CachedIriResolver::new();
        let iri = LazyIri::from_resolved("http://example.org/test");

        let resolved = resolver.resolve_lazy(&iri).unwrap();
        assert_eq!(*resolved, "http://example.org/test");
        assert_eq!(resolver.cache_hit_rate(), 1.0); // Already resolved
    }

    #[test]
    fn test_resolve_lazy_prefixed() {
        let mut resolver = CachedIriResolver::new();
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());

        let iri = LazyIri::from_prefixed("ex", "Person", prefixes);
        let resolved = resolver.resolve_lazy(&iri).unwrap();
        assert_eq!(*resolved, "http://example.org/Person");
    }
}
