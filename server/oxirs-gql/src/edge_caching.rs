//! Edge Caching for GraphQL Responses
//!
//! This module implements edge caching strategies to improve GraphQL API performance
//! by caching responses at the edge (CDN or edge servers):
//! - **Cache-Control Headers**: Automatic generation of cache headers
//! - **ETags**: Entity tags for validation-based caching
//! - **Vary Headers**: Proper cache key computation
//! - **Purging**: Selective cache invalidation
//! - **CDN Integration**: Support for major CDN providers
//! - **Query Analysis**: Automatic cache policy selection
//!
//! ## Features
//!
//! ### Cache-Control Header Generation
//! Automatically generates appropriate `Cache-Control` headers based on:
//! - Query type (query vs mutation vs subscription)
//! - Field volatility (static vs dynamic data)
//! - User authentication status
//! - Custom directives (@cacheControl)
//!
//! ### CDN Support
//! - Cloudflare
//! - Fastly
//! - AWS CloudFront
//! - Akamai
//! - Generic CDN
//!
//! ## Usage
//!
//! ```rust,no_run
//! use oxirs_gql::edge_caching::{EdgeCache, CachePolicy, CdnProvider};
//!
//! // Create edge cache manager
//! let cache = EdgeCache::new(CdnProvider::Cloudflare);
//!
//! // Determine cache policy for query
//! let policy = cache.analyze_query("{ user { name } }");
//!
//! // Generate cache headers
//! let headers = policy.to_headers();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// CDN provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CdnProvider {
    /// Cloudflare CDN
    Cloudflare,
    /// Fastly CDN
    Fastly,
    /// AWS CloudFront
    CloudFront,
    /// Akamai
    Akamai,
    /// Generic CDN
    Generic,
}

impl CdnProvider {
    /// Get provider-specific cache tag header name
    pub fn cache_tag_header(&self) -> &'static str {
        match self {
            Self::Cloudflare => "Cache-Tag",
            Self::Fastly => "Surrogate-Key",
            Self::CloudFront => "x-amz-meta-cache-tags",
            Self::Akamai => "Edge-Cache-Tag",
            Self::Generic => "X-Cache-Tags",
        }
    }

    /// Get provider-specific purge API endpoint pattern
    pub fn purge_api_pattern(&self) -> &'static str {
        match self {
            Self::Cloudflare => "https://api.cloudflare.com/client/v4/zones/{zone}/purge_cache",
            Self::Fastly => "https://api.fastly.com/service/{service}/purge/{key}",
            Self::CloudFront => "cloudfront:CreateInvalidation",
            Self::Akamai => "https://api.akamai.com/ccu/v3/invalidate/tag/{network}",
            Self::Generic => "/purge",
        }
    }

    /// Check if provider supports cache tags
    pub fn supports_cache_tags(&self) -> bool {
        matches!(
            self,
            Self::Cloudflare | Self::Fastly | Self::Akamai | Self::Generic
        )
    }
}

/// Cache directive type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheDirective {
    /// public - response can be cached by any cache
    Public,
    /// private - response is for single user
    Private,
    /// no-cache - must revalidate with server
    NoCache,
    /// no-store - must not be cached
    NoStore,
    /// immutable - response will never change
    Immutable,
    /// must-revalidate - cache must revalidate stale responses
    MustRevalidate,
    /// proxy-revalidate - shared caches must revalidate
    ProxyRevalidate,
}

impl CacheDirective {
    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Private => "private",
            Self::NoCache => "no-cache",
            Self::NoStore => "no-store",
            Self::Immutable => "immutable",
            Self::MustRevalidate => "must-revalidate",
            Self::ProxyRevalidate => "proxy-revalidate",
        }
    }
}

/// Cache policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePolicy {
    /// Cache directives
    pub directives: Vec<CacheDirective>,
    /// Max age in seconds
    pub max_age: Option<Duration>,
    /// Shared max age (s-maxage)
    pub shared_max_age: Option<Duration>,
    /// Stale-while-revalidate window
    pub stale_while_revalidate: Option<Duration>,
    /// Stale-if-error window
    pub stale_if_error: Option<Duration>,
    /// Cache tags for selective purging
    pub cache_tags: Vec<String>,
    /// Vary headers
    pub vary_headers: Vec<String>,
    /// ETag generation enabled
    pub enable_etag: bool,
}

impl CachePolicy {
    /// Create a new cache policy
    pub fn new() -> Self {
        Self {
            directives: Vec::new(),
            max_age: None,
            shared_max_age: None,
            stale_while_revalidate: None,
            stale_if_error: None,
            cache_tags: Vec::new(),
            vary_headers: Vec::new(),
            enable_etag: false,
        }
    }

    /// Create a public cache policy
    pub fn public(max_age: Duration) -> Self {
        Self {
            directives: vec![CacheDirective::Public],
            max_age: Some(max_age),
            shared_max_age: None,
            stale_while_revalidate: None,
            stale_if_error: None,
            cache_tags: Vec::new(),
            vary_headers: vec!["Accept-Encoding".to_string()],
            enable_etag: true,
        }
    }

    /// Create a private cache policy
    pub fn private(max_age: Duration) -> Self {
        Self {
            directives: vec![CacheDirective::Private],
            max_age: Some(max_age),
            shared_max_age: None,
            stale_while_revalidate: None,
            stale_if_error: None,
            cache_tags: Vec::new(),
            vary_headers: vec!["Authorization".to_string(), "Accept-Encoding".to_string()],
            enable_etag: true,
        }
    }

    /// Create a no-cache policy
    pub fn no_cache() -> Self {
        Self {
            directives: vec![CacheDirective::NoCache, CacheDirective::NoStore],
            max_age: Some(Duration::from_secs(0)),
            shared_max_age: None,
            stale_while_revalidate: None,
            stale_if_error: None,
            cache_tags: Vec::new(),
            vary_headers: Vec::new(),
            enable_etag: false,
        }
    }

    /// Create an immutable cache policy (for versioned resources)
    pub fn immutable(max_age: Duration) -> Self {
        Self {
            directives: vec![CacheDirective::Public, CacheDirective::Immutable],
            max_age: Some(max_age),
            shared_max_age: Some(max_age),
            stale_while_revalidate: None,
            stale_if_error: None,
            cache_tags: Vec::new(),
            vary_headers: vec!["Accept-Encoding".to_string()],
            enable_etag: false, // Immutable resources don't need ETags
        }
    }

    /// Add cache tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.cache_tags.contains(&tag) {
            self.cache_tags.push(tag);
        }
    }

    /// Add vary header
    pub fn add_vary(&mut self, header: String) {
        if !self.vary_headers.contains(&header) {
            self.vary_headers.push(header);
        }
    }

    /// Generate Cache-Control header value
    pub fn to_cache_control(&self) -> String {
        let mut parts = Vec::new();

        // Add directives
        for directive in &self.directives {
            parts.push(directive.as_str().to_string());
        }

        // Add max-age
        if let Some(max_age) = self.max_age {
            parts.push(format!("max-age={}", max_age.as_secs()));
        }

        // Add s-maxage
        if let Some(shared_max_age) = self.shared_max_age {
            parts.push(format!("s-maxage={}", shared_max_age.as_secs()));
        }

        // Add stale-while-revalidate
        if let Some(stale) = self.stale_while_revalidate {
            parts.push(format!("stale-while-revalidate={}", stale.as_secs()));
        }

        // Add stale-if-error
        if let Some(stale) = self.stale_if_error {
            parts.push(format!("stale-if-error={}", stale.as_secs()));
        }

        parts.join(", ")
    }

    /// Generate headers map
    pub fn to_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        // Cache-Control header
        let cache_control = self.to_cache_control();
        if !cache_control.is_empty() {
            headers.insert("Cache-Control".to_string(), cache_control);
        }

        // Vary header
        if !self.vary_headers.is_empty() {
            headers.insert("Vary".to_string(), self.vary_headers.join(", "));
        }

        headers
    }

    /// Check if response is cacheable
    pub fn is_cacheable(&self) -> bool {
        !self.directives.contains(&CacheDirective::NoStore)
            && !self.directives.contains(&CacheDirective::NoCache)
    }
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Query cache analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheability {
    /// Whether query is cacheable
    pub cacheable: bool,
    /// Recommended cache policy
    pub policy: CachePolicy,
    /// Reasons why query is/isn't cacheable
    pub reasons: Vec<String>,
    /// Detected entity types
    pub entity_types: Vec<String>,
}

/// Edge cache manager
#[derive(Debug, Clone)]
pub struct EdgeCache {
    /// CDN provider
    provider: CdnProvider,
    /// Default cache policy
    default_policy: CachePolicy,
    /// Query-specific policies
    query_policies: HashMap<String, CachePolicy>,
}

impl EdgeCache {
    /// Create a new edge cache manager
    pub fn new(provider: CdnProvider) -> Self {
        Self {
            provider,
            default_policy: CachePolicy::public(Duration::from_secs(300)), // 5 minutes
            query_policies: HashMap::new(),
        }
    }

    /// Set default cache policy
    pub fn set_default_policy(&mut self, policy: CachePolicy) {
        self.default_policy = policy;
    }

    /// Set policy for specific query pattern
    pub fn set_query_policy(&mut self, query_pattern: String, policy: CachePolicy) {
        self.query_policies.insert(query_pattern, policy);
    }

    /// Analyze query for cacheability
    pub fn analyze_query(&self, query: &str) -> QueryCacheability {
        let mut reasons = Vec::new();
        let mut entity_types = Vec::new();

        // Check for mutations
        if query.trim().starts_with("mutation") {
            return QueryCacheability {
                cacheable: false,
                policy: CachePolicy::no_cache(),
                reasons: vec!["Mutations are not cacheable".to_string()],
                entity_types,
            };
        }

        // Check for subscriptions
        if query.trim().starts_with("subscription") {
            return QueryCacheability {
                cacheable: false,
                policy: CachePolicy::no_cache(),
                reasons: vec!["Subscriptions are not cacheable".to_string()],
                entity_types,
            };
        }

        // Extract entity types from query
        entity_types = self.extract_entity_types(query);

        // Determine cache policy
        let mut policy = self.default_policy.clone();

        // Add cache tags for entity types
        for entity_type in &entity_types {
            policy.add_tag(entity_type.clone());
        }

        // Check for authentication requirements
        if self.requires_authentication(query) {
            policy.directives = vec![CacheDirective::Private];
            policy.add_vary("Authorization".to_string());
            reasons.push("Query requires authentication - using private cache".to_string());
        } else {
            reasons.push("Public cache policy applied".to_string());
        }

        // Check for volatile fields
        if self.has_volatile_fields(query) {
            policy.max_age = Some(Duration::from_secs(60)); // 1 minute
            reasons.push("Query contains volatile fields - shorter TTL".to_string());
        }

        QueryCacheability {
            cacheable: true,
            policy,
            reasons,
            entity_types,
        }
    }

    /// Extract entity types from query
    fn extract_entity_types(&self, query: &str) -> Vec<String> {
        let mut types = Vec::new();

        // Simple regex-like extraction (in production, use proper GraphQL parsing)
        for word in query.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean.is_empty()
                && clean.chars().next().unwrap().is_uppercase()
                && !["Query", "Mutation", "Subscription"].contains(&clean)
            {
                types.push(clean.to_string());
            }
        }

        types
    }

    /// Check if query requires authentication
    fn requires_authentication(&self, query: &str) -> bool {
        // Check for common authenticated fields
        let auth_indicators = ["currentUser", "me", "myProfile", "private"];
        auth_indicators
            .iter()
            .any(|&indicator| query.contains(indicator))
    }

    /// Check if query has volatile fields
    fn has_volatile_fields(&self, query: &str) -> bool {
        // Check for commonly volatile fields
        let volatile_fields = ["timestamp", "now", "current", "random", "live"];
        volatile_fields.iter().any(|&field| query.contains(field))
    }

    /// Generate ETag for response
    pub fn generate_etag(response: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(response.as_bytes());
        let result = hasher.finalize();
        format!("\"{}\"", hex::encode(&result[..16])) // Use first 16 bytes
    }

    /// Check if ETag matches
    pub fn etag_matches(etag: &str, if_none_match: &str) -> bool {
        if_none_match
            .split(',')
            .map(|s| s.trim())
            .any(|tag| tag == etag || tag == "*")
    }

    /// Create purge request for cache tags
    pub fn create_purge_request(&self, tags: Vec<String>) -> PurgeRequest {
        PurgeRequest {
            provider: self.provider,
            tags,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// Get CDN provider
    pub fn provider(&self) -> CdnProvider {
        self.provider
    }
}

/// Cache purge request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurgeRequest {
    /// CDN provider
    pub provider: CdnProvider,
    /// Cache tags to purge
    pub tags: Vec<String>,
    /// Timestamp of purge request
    pub timestamp: std::time::SystemTime,
}

impl PurgeRequest {
    /// Convert to provider-specific API request
    pub fn to_api_request(&self, _credentials: &CdnCredentials) -> Result<String, String> {
        match self.provider {
            CdnProvider::Cloudflare => {
                let tags_json = serde_json::to_string(&self.tags)
                    .map_err(|e| format!("Failed to serialize tags: {}", e))?;
                Ok(format!(r#"{{"tags":{}}}"#, tags_json))
            }
            CdnProvider::Fastly => {
                // Fastly purges by surrogate key
                Ok(self.tags.join(","))
            }
            CdnProvider::CloudFront => {
                // CloudFront uses invalidation paths
                let paths: Vec<String> = self.tags.iter().map(|t| format!("/*{}*", t)).collect();
                let paths_json = serde_json::to_string(&paths)
                    .map_err(|e| format!("Failed to serialize paths: {}", e))?;
                Ok(format!(
                    r#"{{"Paths":{{"Quantity":{},"Items":{}}}}}"#,
                    paths.len(),
                    paths_json
                ))
            }
            CdnProvider::Akamai => {
                let tags_json = serde_json::to_string(&self.tags)
                    .map_err(|e| format!("Failed to serialize tags: {}", e))?;
                Ok(format!(r#"{{"objects":{}}}"#, tags_json))
            }
            CdnProvider::Generic => {
                let tags_json = serde_json::to_string(&self.tags)
                    .map_err(|e| format!("Failed to serialize tags: {}", e))?;
                Ok(tags_json)
            }
        }
    }
}

/// CDN credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdnCredentials {
    /// API key or token
    pub api_key: String,
    /// Zone ID (Cloudflare) or Service ID (Fastly)
    pub zone_or_service_id: Option<String>,
    /// Additional credentials
    pub additional: HashMap<String, String>,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total requests
    pub total_requests: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Stale hits
    pub stale_hits: u64,
    /// Total purge requests
    pub purge_requests: u64,
    /// Total bytes cached
    pub bytes_cached: u64,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.cache_hits as f64) / (self.total_requests as f64)
        }
    }

    /// Calculate miss rate
    pub fn miss_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            (self.cache_misses as f64) / (self.total_requests as f64)
        }
    }

    /// Record cache hit
    pub fn record_hit(&mut self, bytes: u64) {
        self.total_requests += 1;
        self.cache_hits += 1;
        self.bytes_cached += bytes;
    }

    /// Record cache miss
    pub fn record_miss(&mut self) {
        self.total_requests += 1;
        self.cache_misses += 1;
    }

    /// Record stale hit
    pub fn record_stale_hit(&mut self, bytes: u64) {
        self.total_requests += 1;
        self.stale_hits += 1;
        self.bytes_cached += bytes;
    }

    /// Record purge request
    pub fn record_purge(&mut self) {
        self.purge_requests += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdn_provider_cache_tag_header() {
        assert_eq!(CdnProvider::Cloudflare.cache_tag_header(), "Cache-Tag");
        assert_eq!(CdnProvider::Fastly.cache_tag_header(), "Surrogate-Key");
        assert_eq!(
            CdnProvider::CloudFront.cache_tag_header(),
            "x-amz-meta-cache-tags"
        );
        assert_eq!(CdnProvider::Akamai.cache_tag_header(), "Edge-Cache-Tag");
    }

    #[test]
    fn test_cdn_provider_supports_cache_tags() {
        assert!(CdnProvider::Cloudflare.supports_cache_tags());
        assert!(CdnProvider::Fastly.supports_cache_tags());
        assert!(!CdnProvider::CloudFront.supports_cache_tags());
        assert!(CdnProvider::Akamai.supports_cache_tags());
    }

    #[test]
    fn test_cache_directive_as_str() {
        assert_eq!(CacheDirective::Public.as_str(), "public");
        assert_eq!(CacheDirective::Private.as_str(), "private");
        assert_eq!(CacheDirective::NoCache.as_str(), "no-cache");
        assert_eq!(CacheDirective::Immutable.as_str(), "immutable");
    }

    #[test]
    fn test_cache_policy_public() {
        let policy = CachePolicy::public(Duration::from_secs(300));
        assert!(policy.directives.contains(&CacheDirective::Public));
        assert_eq!(policy.max_age, Some(Duration::from_secs(300)));
        assert!(policy.enable_etag);
    }

    #[test]
    fn test_cache_policy_private() {
        let policy = CachePolicy::private(Duration::from_secs(60));
        assert!(policy.directives.contains(&CacheDirective::Private));
        assert_eq!(policy.max_age, Some(Duration::from_secs(60)));
        assert!(policy.vary_headers.contains(&"Authorization".to_string()));
    }

    #[test]
    fn test_cache_policy_no_cache() {
        let policy = CachePolicy::no_cache();
        assert!(policy.directives.contains(&CacheDirective::NoCache));
        assert!(policy.directives.contains(&CacheDirective::NoStore));
        assert!(!policy.enable_etag);
    }

    #[test]
    fn test_cache_policy_immutable() {
        let policy = CachePolicy::immutable(Duration::from_secs(31536000));
        assert!(policy.directives.contains(&CacheDirective::Immutable));
        assert_eq!(policy.max_age, Some(Duration::from_secs(31536000)));
        assert!(!policy.enable_etag); // Immutable resources don't need ETags
    }

    #[test]
    fn test_cache_policy_to_cache_control() {
        let policy = CachePolicy::public(Duration::from_secs(300));
        let cache_control = policy.to_cache_control();
        assert!(cache_control.contains("public"));
        assert!(cache_control.contains("max-age=300"));
    }

    #[test]
    fn test_cache_policy_with_stale_while_revalidate() {
        let mut policy = CachePolicy::public(Duration::from_secs(300));
        policy.stale_while_revalidate = Some(Duration::from_secs(60));

        let cache_control = policy.to_cache_control();
        assert!(cache_control.contains("stale-while-revalidate=60"));
    }

    #[test]
    fn test_cache_policy_add_tag() {
        let mut policy = CachePolicy::new();
        policy.add_tag("User".to_string());
        policy.add_tag("Post".to_string());
        policy.add_tag("User".to_string()); // Duplicate

        assert_eq!(policy.cache_tags.len(), 2);
        assert!(policy.cache_tags.contains(&"User".to_string()));
        assert!(policy.cache_tags.contains(&"Post".to_string()));
    }

    #[test]
    fn test_cache_policy_is_cacheable() {
        let public_policy = CachePolicy::public(Duration::from_secs(300));
        assert!(public_policy.is_cacheable());

        let no_cache_policy = CachePolicy::no_cache();
        assert!(!no_cache_policy.is_cacheable());
    }

    #[test]
    fn test_edge_cache_analyze_query_mutation() {
        let cache = EdgeCache::new(CdnProvider::Cloudflare);
        let result = cache.analyze_query("mutation { updateUser(id: 1) { name } }");

        assert!(!result.cacheable);
        assert!(!result.policy.is_cacheable());
    }

    #[test]
    fn test_edge_cache_analyze_query_subscription() {
        let cache = EdgeCache::new(CdnProvider::Fastly);
        let result = cache.analyze_query("subscription { userUpdated { id name } }");

        assert!(!result.cacheable);
    }

    #[test]
    fn test_edge_cache_analyze_query_public() {
        let cache = EdgeCache::new(CdnProvider::Cloudflare);
        let result = cache.analyze_query("{ posts { title author } }");

        assert!(result.cacheable);
        assert!(result.policy.is_cacheable());
    }

    #[test]
    fn test_edge_cache_analyze_query_authenticated() {
        let cache = EdgeCache::new(CdnProvider::Cloudflare);
        let result = cache.analyze_query("{ currentUser { email } }");

        assert!(result.cacheable);
        assert!(result.policy.directives.contains(&CacheDirective::Private));
    }

    #[test]
    fn test_edge_cache_extract_entity_types() {
        let cache = EdgeCache::new(CdnProvider::Cloudflare);
        let types = cache.extract_entity_types("{ User { name } Post { title } }");

        assert!(types.contains(&"User".to_string()));
        assert!(types.contains(&"Post".to_string()));
    }

    #[test]
    fn test_edge_cache_requires_authentication() {
        let cache = EdgeCache::new(CdnProvider::Cloudflare);

        assert!(cache.requires_authentication("{ currentUser { name } }"));
        assert!(cache.requires_authentication("{ me { email } }"));
        assert!(!cache.requires_authentication("{ posts { title } }"));
    }

    #[test]
    fn test_edge_cache_has_volatile_fields() {
        let cache = EdgeCache::new(CdnProvider::Cloudflare);

        assert!(cache.has_volatile_fields("{ posts { timestamp } }"));
        assert!(cache.has_volatile_fields("{ stats { current } }"));
        assert!(!cache.has_volatile_fields("{ posts { title } }"));
    }

    #[test]
    fn test_edge_cache_generate_etag() {
        let response1 = r#"{"data":{"user":{"name":"Alice"}}}"#;
        let response2 = r#"{"data":{"user":{"name":"Bob"}}}"#;

        let etag1 = EdgeCache::generate_etag(response1);
        let etag2 = EdgeCache::generate_etag(response2);

        assert_ne!(etag1, etag2);
        assert!(etag1.starts_with('"'));
        assert!(etag1.ends_with('"'));

        // Same content should produce same ETag
        assert_eq!(etag1, EdgeCache::generate_etag(response1));
    }

    #[test]
    fn test_edge_cache_etag_matches() {
        let etag = r#""abc123""#;

        assert!(EdgeCache::etag_matches(etag, r#""abc123""#));
        assert!(EdgeCache::etag_matches(etag, "*"));
        assert!(EdgeCache::etag_matches(etag, r#""abc123", "def456""#));
        assert!(!EdgeCache::etag_matches(etag, r#""def456""#));
    }

    #[test]
    fn test_purge_request_cloudflare() {
        let purge = PurgeRequest {
            provider: CdnProvider::Cloudflare,
            tags: vec!["User".to_string(), "Post".to_string()],
            timestamp: std::time::SystemTime::now(),
        };

        let credentials = CdnCredentials {
            api_key: "test".to_string(),
            zone_or_service_id: Some("zone123".to_string()),
            additional: HashMap::new(),
        };

        let request = purge.to_api_request(&credentials).unwrap();
        assert!(request.contains("tags"));
        assert!(request.contains("User"));
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let mut stats = CacheStats::default();
        stats.record_hit(100);
        stats.record_hit(200);
        stats.record_miss();

        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 1);
        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cache_stats_miss_rate() {
        let mut stats = CacheStats::default();
        stats.record_hit(100);
        stats.record_miss();
        stats.record_miss();

        assert_eq!(stats.total_requests, 3);
        assert!((stats.miss_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cache_stats_empty() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
        assert_eq!(stats.miss_rate(), 0.0);
    }
}
