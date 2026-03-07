//! Universal DID Resolver

#[cfg(feature = "did-web")]
use super::methods::DidWebMethod;
use super::methods::{DidKeyMethod, DidMethod};
use super::{Did, DidDocument};
use crate::{DidError, DidResult};
use lru::LruCache;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Universal DID Resolver
pub struct DidResolver {
    /// Registered method resolvers
    methods: HashMap<String, Arc<dyn DidMethod>>,
    /// Resolution cache
    cache: Arc<RwLock<LruCache<String, DidDocument>>>,
    /// Cache TTL in seconds
    cache_ttl_secs: u64,
}

impl Default for DidResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DidResolver {
    /// Create a new resolver with default methods
    pub fn new() -> Self {
        let mut methods: HashMap<String, Arc<dyn DidMethod>> = HashMap::new();

        // Register default methods
        methods.insert("key".to_string(), Arc::new(DidKeyMethod::new()));
        #[cfg(feature = "did-web")]
        methods.insert("web".to_string(), Arc::new(DidWebMethod::new()));

        Self {
            methods,
            cache: Arc::new(RwLock::new(LruCache::new(
                NonZeroUsize::new(1000).expect("cache size must be non-zero"),
            ))),
            cache_ttl_secs: 300, // 5 minutes
        }
    }

    /// Create resolver with only did:key (no network)
    pub fn offline() -> Self {
        let mut methods: HashMap<String, Arc<dyn DidMethod>> = HashMap::new();
        methods.insert("key".to_string(), Arc::new(DidKeyMethod::new()));

        Self {
            methods,
            cache: Arc::new(RwLock::new(LruCache::new(
                NonZeroUsize::new(100).expect("cache size must be non-zero"),
            ))),
            cache_ttl_secs: 3600, // 1 hour for offline
        }
    }

    /// Register a custom method resolver
    pub fn register_method(&mut self, method: impl DidMethod + 'static) {
        self.methods
            .insert(method.method_name().to_string(), Arc::new(method));
    }

    /// Set cache TTL
    pub fn with_cache_ttl(mut self, ttl_secs: u64) -> Self {
        self.cache_ttl_secs = ttl_secs;
        self
    }

    /// Resolve a DID to its DID Document
    pub async fn resolve(&self, did: &Did) -> DidResult<DidDocument> {
        // Check cache
        {
            let cache = self.cache.read().await;
            if let Some(doc) = cache.peek(did.as_str()) {
                return Ok(doc.clone());
            }
        }

        // Find appropriate method resolver
        let method_name = did.method();
        let resolver = self.methods.get(method_name).ok_or_else(|| {
            DidError::UnsupportedMethod(format!("No resolver for method: {}", method_name))
        })?;

        // Resolve
        let doc = resolver.resolve(did).await?;

        // Cache result
        {
            let mut cache = self.cache.write().await;
            cache.put(did.as_str().to_string(), doc.clone());
        }

        Ok(doc)
    }

    /// Resolve DID from string
    pub async fn resolve_str(&self, did_str: &str) -> DidResult<DidDocument> {
        let did = Did::new(did_str)?;
        self.resolve(&did).await
    }

    /// Check if a method is supported
    pub fn supports(&self, method: &str) -> bool {
        self.methods.contains_key(method)
    }

    /// Get list of supported methods
    pub fn supported_methods(&self) -> Vec<&str> {
        self.methods.keys().map(|s| s.as_str()).collect()
    }

    /// Clear the resolution cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resolve_did_key() {
        let resolver = DidResolver::new();

        let public_key = [0u8; 32];
        let did = Did::new_key_ed25519(&public_key).unwrap();

        let doc = resolver.resolve(&did).await.unwrap();
        assert_eq!(doc.id, did);
    }

    #[tokio::test]
    async fn test_resolve_caching() {
        let resolver = DidResolver::new();

        let public_key = [0u8; 32];
        let did = Did::new_key_ed25519(&public_key).unwrap();

        // First resolve
        let doc1 = resolver.resolve(&did).await.unwrap();

        // Second resolve (should be cached)
        let doc2 = resolver.resolve(&did).await.unwrap();

        assert_eq!(doc1.id, doc2.id);
    }

    #[tokio::test]
    async fn test_unsupported_method() {
        let resolver = DidResolver::new();

        let did = Did::new("did:unknown:abc123").unwrap();
        let result = resolver.resolve(&did).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DidError::UnsupportedMethod(_)
        ));
    }

    #[test]
    fn test_supported_methods() {
        let resolver = DidResolver::new();

        assert!(resolver.supports("key"));
        #[cfg(feature = "did-web")]
        assert!(resolver.supports("web"));
        assert!(!resolver.supports("unknown"));
    }
}
