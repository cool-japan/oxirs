// Edge caching integration framework for OxiRS Fuseki
//
// Provides integration with edge caching providers (Cloudflare, Fastly, etc.)
// for caching SPARQL query results at the edge for improved global performance.

use crate::error::{FusekiError, FusekiResult};
use dashmap::DashMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Edge caching provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeCacheProvider {
    /// Cloudflare CDN caching
    Cloudflare,
    /// Fastly CDN caching
    Fastly,
    /// AWS CloudFront
    CloudFront,
    /// Akamai
    Akamai,
    /// Custom provider
    Custom,
}

/// Cache control directives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControl {
    /// Cache enabled
    pub enabled: bool,
    /// Maximum age in seconds
    pub max_age: u64,
    /// Stale-while-revalidate duration in seconds
    pub stale_while_revalidate: Option<u64>,
    /// Stale-if-error duration in seconds
    pub stale_if_error: Option<u64>,
    /// Public vs private cache
    pub public: bool,
    /// Allow cache transformation
    pub no_transform: bool,
    /// Must revalidate
    pub must_revalidate: bool,
}

impl Default for CacheControl {
    fn default() -> Self {
        Self {
            enabled: true,
            max_age: 3600,                     // 1 hour default
            stale_while_revalidate: Some(300), // 5 minutes
            stale_if_error: Some(86400),       // 24 hours
            public: true,
            no_transform: true,
            must_revalidate: false,
        }
    }
}

impl CacheControl {
    /// Convert to Cache-Control header value
    pub fn to_header_value(&self) -> String {
        let mut directives = Vec::new();

        if self.public {
            directives.push("public".to_string());
        } else {
            directives.push("private".to_string());
        }

        directives.push(format!("max-age={}", self.max_age));

        if let Some(swr) = self.stale_while_revalidate {
            directives.push(format!("stale-while-revalidate={}", swr));
        }

        if let Some(sie) = self.stale_if_error {
            directives.push(format!("stale-if-error={}", sie));
        }

        if self.no_transform {
            directives.push("no-transform".to_string());
        }

        if self.must_revalidate {
            directives.push("must-revalidate".to_string());
        }

        directives.join(", ")
    }
}

/// Edge caching configuration
#[derive(Debug, Clone)]
pub struct EdgeCacheConfig {
    pub provider: EdgeCacheProvider,
    pub enabled: bool,
    /// API key/token for the provider
    pub api_key: Option<String>,
    /// Zone ID (Cloudflare) or Service ID (Fastly)
    pub zone_id: Option<String>,
    /// Default cache control
    pub default_cache_control: CacheControl,
    /// Custom cache key patterns
    pub cache_key_patterns: Vec<String>,
    /// Cache tags for purge groups
    pub enable_cache_tags: bool,
    /// Minimum query execution time to cache (ms)
    pub min_execution_time_ms: u64,
    /// Maximum cacheable response size (bytes)
    pub max_response_size_bytes: usize,
}

impl Default for EdgeCacheConfig {
    fn default() -> Self {
        Self {
            provider: EdgeCacheProvider::Custom,
            enabled: false,
            api_key: None,
            zone_id: None,
            default_cache_control: CacheControl::default(),
            cache_key_patterns: vec!["query:{}".to_string()],
            enable_cache_tags: true,
            min_execution_time_ms: 100,
            max_response_size_bytes: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// Cache invalidation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvalidationStrategy {
    /// Purge all cache
    PurgeAll,
    /// Purge by cache tags
    PurgeByTags(Vec<String>),
    /// Purge by URL pattern
    PurgeByUrls(Vec<String>),
    /// Purge by cache key
    PurgeByKeys(Vec<String>),
}

/// Edge cache manager
pub struct EdgeCacheManager {
    config: EdgeCacheConfig,
    cache_metadata: Arc<DashMap<String, CacheMetadata>>,
    purge_queue: Arc<DashMap<String, PurgeRequest>>,
}

/// Cache metadata
#[derive(Debug, Clone)]
struct CacheMetadata {
    cache_key: String,
    tags: Vec<String>,
    created_at: Instant,
    ttl: Duration,
    size_bytes: usize,
}

/// Purge request
#[derive(Debug, Clone)]
struct PurgeRequest {
    strategy: InvalidationStrategy,
    requested_at: Instant,
    status: PurgeStatus,
}

#[derive(Debug, Clone)]
enum PurgeStatus {
    Pending,
    InProgress,
    Completed,
    Failed(String),
}

impl EdgeCacheManager {
    /// Create a new edge cache manager
    pub fn new(config: EdgeCacheConfig) -> Self {
        Self {
            config,
            cache_metadata: Arc::new(DashMap::new()),
            purge_queue: Arc::new(DashMap::new()),
        }
    }

    /// Get cache control headers for a SPARQL query
    pub fn get_cache_headers(
        &self,
        query: &str,
        execution_time_ms: u64,
        response_size: usize,
    ) -> Option<HashMap<String, String>> {
        if !self.config.enabled {
            return None;
        }

        // Check if query should be cached
        if execution_time_ms < self.config.min_execution_time_ms {
            return None;
        }

        if response_size > self.config.max_response_size_bytes {
            return None;
        }

        // Determine cache control based on query type
        let cache_control = self.determine_cache_control(query);

        let mut headers = HashMap::new();
        headers.insert("Cache-Control".to_string(), cache_control.to_header_value());

        // Add Vary header to vary cache by Accept header
        headers.insert("Vary".to_string(), "Accept, Accept-Encoding".to_string());

        // Add cache tags if enabled
        if self.config.enable_cache_tags {
            if let Some(tags) = self.generate_cache_tags(query) {
                headers.insert("Cache-Tag".to_string(), tags.join(","));
            }
        }

        // Provider-specific headers
        match self.config.provider {
            EdgeCacheProvider::Cloudflare => {
                // Cloudflare-specific cache headers
                headers.insert("CF-Cache-Status".to_string(), "MISS".to_string());
                headers.insert(
                    "CDN-Cache-Control".to_string(),
                    format!("max-age={}", cache_control.max_age),
                );
            }
            EdgeCacheProvider::Fastly => {
                // Fastly-specific cache headers
                headers.insert(
                    "Surrogate-Control".to_string(),
                    format!("max-age={}", cache_control.max_age),
                );
            }
            EdgeCacheProvider::CloudFront => {
                // CloudFront cache headers
                headers.insert(
                    "CloudFront-Cache-Control".to_string(),
                    cache_control.to_header_value(),
                );
            }
            _ => {}
        }

        Some(headers)
    }

    /// Determine cache control for a query
    fn determine_cache_control(&self, query: &str) -> CacheControl {
        let query_upper = query.to_uppercase();

        // Read-only queries (SELECT, ASK, DESCRIBE, CONSTRUCT) can be cached longer
        if query_upper.contains("SELECT")
            || query_upper.contains("ASK")
            || query_upper.contains("DESCRIBE")
            || query_upper.contains("CONSTRUCT")
        {
            // Check for volatile functions that make query non-cacheable
            if query_upper.contains("NOW()")
                || query_upper.contains("RAND()")
                || query_upper.contains("UUID()")
                || query_upper.contains("STRUUID()")
            {
                // Short cache for volatile queries
                return CacheControl {
                    max_age: 60,
                    stale_while_revalidate: Some(30),
                    ..self.config.default_cache_control.clone()
                };
            }

            // Long cache for stable queries
            return self.config.default_cache_control.clone();
        }

        // Update queries should not be cached
        if query_upper.contains("INSERT")
            || query_upper.contains("DELETE")
            || query_upper.contains("LOAD")
            || query_upper.contains("CLEAR")
        {
            return CacheControl {
                enabled: false,
                max_age: 0,
                public: false,
                must_revalidate: true,
                ..Default::default()
            };
        }

        // Default
        self.config.default_cache_control.clone()
    }

    /// Generate cache tags for purging
    fn generate_cache_tags(&self, query: &str) -> Option<Vec<String>> {
        let mut tags = Vec::new();

        // Add query type tag
        let query_upper = query.to_uppercase();
        if query_upper.contains("SELECT") {
            tags.push("sparql:select".to_string());
        }
        if query_upper.contains("CONSTRUCT") {
            tags.push("sparql:construct".to_string());
        }
        if query_upper.contains("DESCRIBE") {
            tags.push("sparql:describe".to_string());
        }
        if query_upper.contains("ASK") {
            tags.push("sparql:ask".to_string());
        }

        // Extract graph URIs for targeted purging
        if let Some(graph_uris) = self.extract_graph_uris(query) {
            for uri in graph_uris {
                tags.push(format!("graph:{}", uri));
            }
        }

        if tags.is_empty() {
            None
        } else {
            Some(tags)
        }
    }

    /// Extract graph URIs from query for cache tagging
    ///
    /// Parses SPARQL queries to extract GRAPH clause URIs for targeted cache invalidation.
    /// Supports patterns like:
    /// - `GRAPH <http://example.org/graph> { ... }`
    /// - `FROM NAMED <http://example.org/graph>`
    fn extract_graph_uris(&self, query: &str) -> Option<Vec<String>> {
        let mut graph_uris = Vec::new();

        // Pattern 1: GRAPH <uri> { ... }
        // Matches GRAPH keyword followed by a URI in angle brackets
        let graph_pattern = Regex::new(r"(?i)\bGRAPH\s+<([^>]+)>").ok()?;
        for cap in graph_pattern.captures_iter(query) {
            if let Some(uri) = cap.get(1) {
                graph_uris.push(uri.as_str().to_string());
            }
        }

        // Pattern 2: FROM NAMED <uri>
        // Matches named graphs specified in the query prologue
        let from_named_pattern = Regex::new(r"(?i)\bFROM\s+NAMED\s+<([^>]+)>").ok()?;
        for cap in from_named_pattern.captures_iter(query) {
            if let Some(uri) = cap.get(1) {
                let uri_str = uri.as_str().to_string();
                if !graph_uris.contains(&uri_str) {
                    graph_uris.push(uri_str);
                }
            }
        }

        // Pattern 3: WITH <uri> for SPARQL Update
        // Matches graph context in UPDATE queries
        let with_pattern = Regex::new(r"(?i)\bWITH\s+<([^>]+)>").ok()?;
        for cap in with_pattern.captures_iter(query) {
            if let Some(uri) = cap.get(1) {
                let uri_str = uri.as_str().to_string();
                if !graph_uris.contains(&uri_str) {
                    graph_uris.push(uri_str);
                }
            }
        }

        // Pattern 4: INTO <uri> for INSERT DATA
        let into_pattern = Regex::new(r"(?i)\bINTO\s+<([^>]+)>").ok()?;
        for cap in into_pattern.captures_iter(query) {
            if let Some(uri) = cap.get(1) {
                let uri_str = uri.as_str().to_string();
                if !graph_uris.contains(&uri_str) {
                    graph_uris.push(uri_str);
                }
            }
        }

        if graph_uris.is_empty() {
            None
        } else {
            Some(graph_uris)
        }
    }

    /// Purge cache for a dataset
    pub async fn purge_dataset(&self, dataset: &str) -> FusekiResult<()> {
        let purge_id = uuid::Uuid::new_v4().to_string();

        let strategy = InvalidationStrategy::PurgeByTags(vec![format!("dataset:{}", dataset)]);

        self.purge_queue.insert(
            purge_id.clone(),
            PurgeRequest {
                strategy: strategy.clone(),
                requested_at: Instant::now(),
                status: PurgeStatus::Pending,
            },
        );

        // Execute purge based on provider
        match self.config.provider {
            EdgeCacheProvider::Cloudflare => {
                self.purge_cloudflare(&strategy).await?;
            }
            EdgeCacheProvider::Fastly => {
                self.purge_fastly(&strategy).await?;
            }
            EdgeCacheProvider::CloudFront => {
                self.purge_cloudfront(&strategy).await?;
            }
            _ => {
                // Custom provider - log for manual processing
                tracing::info!("Purge requested for custom provider: {:?}", strategy);
            }
        }

        // Update status
        if let Some(mut req) = self.purge_queue.get_mut(&purge_id) {
            req.status = PurgeStatus::Completed;
        }

        Ok(())
    }

    /// Purge cache by tags (for dataset updates)
    pub async fn purge_by_tags(&self, tags: Vec<String>) -> FusekiResult<()> {
        let purge_id = uuid::Uuid::new_v4().to_string();

        let strategy = InvalidationStrategy::PurgeByTags(tags);

        self.purge_queue.insert(
            purge_id.clone(),
            PurgeRequest {
                strategy: strategy.clone(),
                requested_at: Instant::now(),
                status: PurgeStatus::Pending,
            },
        );

        match self.config.provider {
            EdgeCacheProvider::Cloudflare => {
                self.purge_cloudflare(&strategy).await?;
            }
            EdgeCacheProvider::Fastly => {
                self.purge_fastly(&strategy).await?;
            }
            EdgeCacheProvider::CloudFront => {
                self.purge_cloudfront(&strategy).await?;
            }
            _ => {
                tracing::info!("Purge requested for custom provider: {:?}", strategy);
            }
        }

        if let Some(mut req) = self.purge_queue.get_mut(&purge_id) {
            req.status = PurgeStatus::Completed;
        }

        Ok(())
    }

    /// Purge all cache
    pub async fn purge_all(&self) -> FusekiResult<()> {
        let strategy = InvalidationStrategy::PurgeAll;

        match self.config.provider {
            EdgeCacheProvider::Cloudflare => {
                self.purge_cloudflare(&strategy).await?;
            }
            EdgeCacheProvider::Fastly => {
                self.purge_fastly(&strategy).await?;
            }
            EdgeCacheProvider::CloudFront => {
                self.purge_cloudfront(&strategy).await?;
            }
            _ => {
                tracing::info!("Purge all requested for custom provider");
            }
        }

        Ok(())
    }

    // Provider-specific purge implementations

    async fn purge_cloudflare(&self, strategy: &InvalidationStrategy) -> FusekiResult<()> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("Cloudflare API key not configured"))?;

        let zone_id = self
            .config
            .zone_id
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("Cloudflare zone ID not configured"))?;

        let url = format!(
            "https://api.cloudflare.com/client/v4/zones/{}/purge_cache",
            zone_id
        );

        let body = match strategy {
            InvalidationStrategy::PurgeAll => {
                serde_json::json!({ "purge_everything": true })
            }
            InvalidationStrategy::PurgeByTags(tags) => {
                serde_json::json!({ "tags": tags })
            }
            InvalidationStrategy::PurgeByUrls(urls) => {
                serde_json::json!({ "files": urls })
            }
            _ => {
                return Err(FusekiError::bad_request(
                    "Unsupported purge strategy for Cloudflare",
                ))
            }
        };

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                FusekiError::service_unavailable(format!("Cloudflare API error: {}", e))
            })?;

        if !response.status().is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(FusekiError::service_unavailable(format!(
                "Cloudflare purge failed: {}",
                error_text
            )));
        }

        Ok(())
    }

    async fn purge_fastly(&self, strategy: &InvalidationStrategy) -> FusekiResult<()> {
        let api_key = self
            .config
            .api_key
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("Fastly API key not configured"))?;

        let service_id = self
            .config
            .zone_id
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("Fastly service ID not configured"))?;

        match strategy {
            InvalidationStrategy::PurgeAll => {
                let url = format!("https://api.fastly.com/service/{}/purge_all", service_id);

                let client = reqwest::Client::new();
                let response = client
                    .post(&url)
                    .header("Fastly-Key", api_key)
                    .send()
                    .await
                    .map_err(|e| {
                        FusekiError::service_unavailable(format!("Fastly API error: {}", e))
                    })?;

                if !response.status().is_success() {
                    return Err(FusekiError::service_unavailable("Fastly purge all failed"));
                }
            }
            InvalidationStrategy::PurgeByTags(tags) => {
                // Fastly uses surrogate keys
                for tag in tags {
                    let url = format!(
                        "https://api.fastly.com/service/{}/purge/{}",
                        service_id, tag
                    );

                    let client = reqwest::Client::new();
                    client
                        .post(&url)
                        .header("Fastly-Key", api_key)
                        .send()
                        .await
                        .map_err(|e| {
                            FusekiError::service_unavailable(format!("Fastly API error: {}", e))
                        })?;
                }
            }
            _ => {
                return Err(FusekiError::bad_request(
                    "Unsupported purge strategy for Fastly",
                ))
            }
        }

        Ok(())
    }

    async fn purge_cloudfront(&self, strategy: &InvalidationStrategy) -> FusekiResult<()> {
        // CloudFront purge would use AWS SDK
        // For now, just log
        tracing::info!("CloudFront purge requested: {:?}", strategy);
        Ok(())
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> EdgeCacheStatistics {
        EdgeCacheStatistics {
            total_cached_items: self.cache_metadata.len(),
            pending_purges: self
                .purge_queue
                .iter()
                .filter(|e| matches!(e.value().status, PurgeStatus::Pending))
                .count(),
            completed_purges: self
                .purge_queue
                .iter()
                .filter(|e| matches!(e.value().status, PurgeStatus::Completed))
                .count(),
            failed_purges: self
                .purge_queue
                .iter()
                .filter(|e| matches!(e.value().status, PurgeStatus::Failed(_)))
                .count(),
        }
    }
}

/// Edge cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCacheStatistics {
    pub total_cached_items: usize,
    pub pending_purges: usize,
    pub completed_purges: usize,
    pub failed_purges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_control_header() {
        let cache_control = CacheControl {
            enabled: true,
            max_age: 3600,
            stale_while_revalidate: Some(300),
            stale_if_error: Some(86400),
            public: true,
            no_transform: true,
            must_revalidate: false,
        };

        let header = cache_control.to_header_value();
        assert!(header.contains("public"));
        assert!(header.contains("max-age=3600"));
        assert!(header.contains("stale-while-revalidate=300"));
        assert!(header.contains("no-transform"));
    }

    #[test]
    fn test_cache_control_for_select_query() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query = "SELECT * WHERE { ?s ?p ?o } LIMIT 10";
        let cache_control = manager.determine_cache_control(query);

        assert!(cache_control.enabled);
        assert!(cache_control.max_age > 0);
    }

    #[test]
    fn test_no_cache_for_update_query() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query =
            "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }";
        let cache_control = manager.determine_cache_control(query);

        assert!(!cache_control.enabled);
        assert_eq!(cache_control.max_age, 0);
    }

    #[test]
    fn test_short_cache_for_volatile_query() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config.clone());

        let query = "SELECT * WHERE { ?s ?p ?o . BIND(NOW() AS ?time) }";
        let cache_control = manager.determine_cache_control(query);

        assert!(cache_control.enabled);
        assert!(cache_control.max_age < config.default_cache_control.max_age);
    }

    #[test]
    fn test_cache_tags_generation() {
        let config = EdgeCacheConfig {
            enable_cache_tags: true,
            ..Default::default()
        };
        let manager = EdgeCacheManager::new(config);

        let query = "SELECT * WHERE { ?s ?p ?o }";
        let tags = manager.generate_cache_tags(query);

        assert!(tags.is_some());
        let tags = tags.unwrap();
        assert!(tags.contains(&"sparql:select".to_string()));
    }

    #[test]
    fn test_cache_headers_generation() {
        let config = EdgeCacheConfig {
            enabled: true,
            min_execution_time_ms: 50,
            ..Default::default()
        };
        let manager = EdgeCacheManager::new(config);

        let query = "SELECT * WHERE { ?s ?p ?o }";
        let headers = manager.get_cache_headers(query, 100, 1024);

        assert!(headers.is_some());
        let headers = headers.unwrap();
        assert!(headers.contains_key("Cache-Control"));
        assert!(headers.contains_key("Vary"));
    }

    #[test]
    fn test_no_cache_for_fast_queries() {
        let config = EdgeCacheConfig {
            enabled: true,
            min_execution_time_ms: 100,
            ..Default::default()
        };
        let manager = EdgeCacheManager::new(config);

        let query = "SELECT * WHERE { ?s ?p ?o }";
        let headers = manager.get_cache_headers(query, 50, 1024); // 50ms < 100ms threshold

        assert!(headers.is_none());
    }

    #[test]
    fn test_extract_graph_uris_graph_pattern() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query = r#"
            SELECT * WHERE {
                GRAPH <http://example.org/graph1> {
                    ?s ?p ?o
                }
            }
        "#;
        let uris = manager.extract_graph_uris(query);
        assert!(uris.is_some());
        let uris = uris.unwrap();
        assert_eq!(uris.len(), 1);
        assert!(uris.contains(&"http://example.org/graph1".to_string()));
    }

    #[test]
    fn test_extract_graph_uris_from_named_pattern() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query = r#"
            SELECT * FROM NAMED <http://example.org/named-graph>
            WHERE { GRAPH ?g { ?s ?p ?o } }
        "#;
        let uris = manager.extract_graph_uris(query);
        assert!(uris.is_some());
        let uris = uris.unwrap();
        assert_eq!(uris.len(), 1);
        assert!(uris.contains(&"http://example.org/named-graph".to_string()));
    }

    #[test]
    fn test_extract_graph_uris_with_pattern() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query = r#"
            WITH <http://example.org/update-graph>
            DELETE { ?s ?p ?o }
            WHERE { ?s ?p ?o }
        "#;
        let uris = manager.extract_graph_uris(query);
        assert!(uris.is_some());
        let uris = uris.unwrap();
        assert_eq!(uris.len(), 1);
        assert!(uris.contains(&"http://example.org/update-graph".to_string()));
    }

    #[test]
    fn test_extract_graph_uris_into_pattern() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query = r#"
            INSERT DATA INTO <http://example.org/target-graph> {
                <http://example.org/s> <http://example.org/p> <http://example.org/o>
            }
        "#;
        let uris = manager.extract_graph_uris(query);
        assert!(uris.is_some());
        let uris = uris.unwrap();
        assert_eq!(uris.len(), 1);
        assert!(uris.contains(&"http://example.org/target-graph".to_string()));
    }

    #[test]
    fn test_extract_graph_uris_multiple_graphs() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query = r#"
            SELECT * FROM NAMED <http://example.org/graph1>
            FROM NAMED <http://example.org/graph2>
            WHERE {
                GRAPH <http://example.org/graph3> {
                    ?s ?p ?o
                }
            }
        "#;
        let uris = manager.extract_graph_uris(query);
        assert!(uris.is_some());
        let uris = uris.unwrap();
        assert_eq!(uris.len(), 3);
        assert!(uris.contains(&"http://example.org/graph1".to_string()));
        assert!(uris.contains(&"http://example.org/graph2".to_string()));
        assert!(uris.contains(&"http://example.org/graph3".to_string()));
    }

    #[test]
    fn test_extract_graph_uris_no_graphs() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query = "SELECT * WHERE { ?s ?p ?o }";
        let uris = manager.extract_graph_uris(query);
        assert!(uris.is_none());
    }

    #[test]
    fn test_extract_graph_uris_case_insensitive() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query = r#"
            SELECT * WHERE {
                graph <http://example.org/lowercase>
                { ?s ?p ?o }
            }
        "#;
        let uris = manager.extract_graph_uris(query);
        assert!(uris.is_some());
        let uris = uris.unwrap();
        assert_eq!(uris.len(), 1);
        assert!(uris.contains(&"http://example.org/lowercase".to_string()));
    }

    #[test]
    fn test_extract_graph_uris_no_duplicates() {
        let config = EdgeCacheConfig::default();
        let manager = EdgeCacheManager::new(config);

        let query = r#"
            SELECT * FROM NAMED <http://example.org/same>
            WHERE {
                GRAPH <http://example.org/same> {
                    ?s ?p ?o
                }
            }
        "#;
        let uris = manager.extract_graph_uris(query);
        assert!(uris.is_some());
        let uris = uris.unwrap();
        // Should deduplicate
        assert_eq!(uris.len(), 1);
        assert!(uris.contains(&"http://example.org/same".to_string()));
    }
}
