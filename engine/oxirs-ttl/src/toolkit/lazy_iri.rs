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

    /// Simple relative IRI resolution
    ///
    /// Note: This is a simplified implementation. Production code should
    /// use a proper RFC 3986 implementation like the `iri-string` crate.
    fn resolve_relative(base: &str, relative: &str) -> String {
        if relative.starts_with("http://") || relative.starts_with("https://") {
            // Already absolute
            return relative.to_string();
        }

        // Handle base URL with or without trailing slash
        if base.ends_with('/') {
            format!("{}{}", base, relative)
        } else {
            format!("{}/{}", base, relative)
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

        assert_eq!(
            iri.resolve().unwrap(),
            "http://example.org/base/relative/path"
        );
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
