//! NGSI-LD @context Handling
//!
//! Manages JSON-LD context resolution, caching, and expansion.

use super::types::{NgsiContext, NgsiError};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Default NGSI-LD Core Context
pub const CORE_CONTEXT_URL: &str = "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld";

/// Context cache for resolved contexts
pub struct ContextCache {
    cache: Arc<RwLock<HashMap<String, CachedContext>>>,
    max_size: usize,
}

/// Cached context entry
#[derive(Debug, Clone)]
pub struct CachedContext {
    pub context: HashMap<String, String>,
    pub cached_at: std::time::Instant,
    pub ttl_seconds: u64,
}

impl Default for ContextCache {
    fn default() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size: 100,
        }
    }
}

impl ContextCache {
    /// Create a new context cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }

    /// Get a cached context
    pub async fn get(&self, url: &str) -> Option<CachedContext> {
        let cache = self.cache.read().await;
        cache.get(url).cloned().filter(|c| !c.is_expired())
    }

    /// Store a context in cache
    pub async fn put(&self, url: String, context: HashMap<String, String>, ttl_seconds: u64) {
        let mut cache = self.cache.write().await;

        // Evict old entries if cache is full
        if cache.len() >= self.max_size {
            let expired: Vec<String> = cache
                .iter()
                .filter(|(_, v)| v.is_expired())
                .map(|(k, _)| k.clone())
                .collect();

            for key in expired {
                cache.remove(&key);
            }

            // If still full, remove oldest
            if cache.len() >= self.max_size {
                if let Some(oldest) = cache
                    .iter()
                    .min_by_key(|(_, v)| v.cached_at)
                    .map(|(k, _)| k.clone())
                {
                    cache.remove(&oldest);
                }
            }
        }

        cache.insert(
            url,
            CachedContext {
                context,
                cached_at: std::time::Instant::now(),
                ttl_seconds,
            },
        );
    }

    /// Clear the cache
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

impl CachedContext {
    /// Check if this cached entry is expired
    pub fn is_expired(&self) -> bool {
        self.cached_at.elapsed().as_secs() > self.ttl_seconds
    }
}

/// Context resolver for NGSI-LD
pub struct ContextResolver {
    cache: ContextCache,
    default_context: HashMap<String, String>,
}

impl Default for ContextResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextResolver {
    /// Create a new context resolver
    pub fn new() -> Self {
        let mut default_context = HashMap::new();

        // NGSI-LD Core Context terms
        default_context.insert("id".to_string(), "@id".to_string());
        default_context.insert("type".to_string(), "@type".to_string());
        default_context.insert(
            "Property".to_string(),
            "https://uri.etsi.org/ngsi-ld/Property".to_string(),
        );
        default_context.insert(
            "Relationship".to_string(),
            "https://uri.etsi.org/ngsi-ld/Relationship".to_string(),
        );
        default_context.insert(
            "GeoProperty".to_string(),
            "https://uri.etsi.org/ngsi-ld/GeoProperty".to_string(),
        );
        default_context.insert(
            "value".to_string(),
            "https://uri.etsi.org/ngsi-ld/hasValue".to_string(),
        );
        default_context.insert(
            "object".to_string(),
            "https://uri.etsi.org/ngsi-ld/hasObject".to_string(),
        );
        default_context.insert(
            "observedAt".to_string(),
            "https://uri.etsi.org/ngsi-ld/observedAt".to_string(),
        );
        default_context.insert(
            "unitCode".to_string(),
            "https://uri.etsi.org/ngsi-ld/unitCode".to_string(),
        );
        default_context.insert(
            "datasetId".to_string(),
            "https://uri.etsi.org/ngsi-ld/datasetId".to_string(),
        );
        default_context.insert(
            "createdAt".to_string(),
            "https://uri.etsi.org/ngsi-ld/createdAt".to_string(),
        );
        default_context.insert(
            "modifiedAt".to_string(),
            "https://uri.etsi.org/ngsi-ld/modifiedAt".to_string(),
        );
        default_context.insert(
            "location".to_string(),
            "https://uri.etsi.org/ngsi-ld/location".to_string(),
        );

        Self {
            cache: ContextCache::default(),
            default_context,
        }
    }

    /// Expand a term using context
    pub fn expand_term(&self, term: &str, context: &NgsiContext) -> String {
        // First check the provided context
        if let Some(expanded) = self.lookup_in_context(term, context) {
            return expanded;
        }

        // Then check default context
        if let Some(expanded) = self.default_context.get(term) {
            return expanded.clone();
        }

        // Return as-is if not found
        term.to_string()
    }

    /// Compact a URI using context
    pub fn compact_uri(&self, uri: &str, context: &NgsiContext) -> String {
        // First check the provided context
        if let Some(compacted) = self.reverse_lookup_in_context(uri, context) {
            return compacted;
        }

        // Then check default context
        for (term, expanded) in &self.default_context {
            if uri == expanded {
                return term.clone();
            }
        }

        // Return as-is if not found
        uri.to_string()
    }

    /// Look up a term in the provided context
    fn lookup_in_context(&self, term: &str, context: &NgsiContext) -> Option<String> {
        match context {
            NgsiContext::Uri(_) => None, // Would need to fetch
            NgsiContext::Object(obj) => obj.get(term).and_then(|v| v.as_str().map(String::from)),
            NgsiContext::Array(arr) => {
                for ctx in arr {
                    if let Some(expanded) = self.lookup_in_context(term, ctx) {
                        return Some(expanded);
                    }
                }
                None
            }
        }
    }

    /// Reverse lookup in context
    fn reverse_lookup_in_context(&self, uri: &str, context: &NgsiContext) -> Option<String> {
        match context {
            NgsiContext::Uri(_) => None,
            NgsiContext::Object(obj) => {
                for (term, value) in obj {
                    if value.as_str() == Some(uri) {
                        return Some(term.clone());
                    }
                }
                None
            }
            NgsiContext::Array(arr) => {
                for ctx in arr {
                    if let Some(term) = self.reverse_lookup_in_context(uri, ctx) {
                        return Some(term);
                    }
                }
                None
            }
        }
    }

    /// Resolve and merge contexts
    pub async fn resolve_context(
        &self,
        context: &NgsiContext,
    ) -> Result<HashMap<String, String>, NgsiError> {
        let mut resolved = self.default_context.clone();

        match context {
            NgsiContext::Uri(url) => {
                // Check cache first
                if let Some(cached) = self.cache.get(url).await {
                    resolved.extend(cached.context);
                }
                // Note: In production, would fetch from URL here
            }
            NgsiContext::Object(obj) => {
                for (key, value) in obj {
                    if let Some(v) = value.as_str() {
                        resolved.insert(key.clone(), v.to_string());
                    }
                }
            }
            NgsiContext::Array(arr) => {
                for ctx in arr {
                    let sub_resolved = Box::pin(self.resolve_context(ctx)).await?;
                    resolved.extend(sub_resolved);
                }
            }
        }

        Ok(resolved)
    }

    /// Get the default NGSI-LD context
    pub fn default_context(&self) -> NgsiContext {
        NgsiContext::Uri(CORE_CONTEXT_URL.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_resolver_expand() {
        let resolver = ContextResolver::new();
        let context = NgsiContext::default();

        let expanded = resolver.expand_term("Property", &context);
        assert_eq!(expanded, "https://uri.etsi.org/ngsi-ld/Property");

        let expanded = resolver.expand_term("unknown", &context);
        assert_eq!(expanded, "unknown");
    }

    #[test]
    fn test_context_resolver_compact() {
        let resolver = ContextResolver::new();
        let context = NgsiContext::default();

        let compacted = resolver.compact_uri("https://uri.etsi.org/ngsi-ld/Property", &context);
        assert_eq!(compacted, "Property");
    }

    #[tokio::test]
    async fn test_context_cache() {
        let cache = ContextCache::new(10);

        let mut ctx = HashMap::new();
        ctx.insert("test".to_string(), "http://example.org/test".to_string());

        cache
            .put("http://example.org/context".to_string(), ctx.clone(), 3600)
            .await;

        let cached = cache.get("http://example.org/context").await;
        assert!(cached.is_some());
        assert_eq!(
            cached.unwrap().context.get("test"),
            Some(&"http://example.org/test".to_string())
        );
    }

    #[test]
    fn test_cached_context_expiry() {
        let ctx = CachedContext {
            context: HashMap::new(),
            cached_at: std::time::Instant::now() - std::time::Duration::from_secs(100),
            ttl_seconds: 50,
        };

        assert!(ctx.is_expired());

        let ctx = CachedContext {
            context: HashMap::new(),
            cached_at: std::time::Instant::now(),
            ttl_seconds: 3600,
        };

        assert!(!ctx.is_expired());
    }
}
