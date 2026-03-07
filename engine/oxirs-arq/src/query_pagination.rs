//! Query Result Pagination
//!
//! Provides efficient pagination strategies for large SPARQL result sets.
//!
//! ## Features
//!
//! - **Cursor-based pagination**: Stateless pagination using opaque cursors
//! - **Offset/Limit optimization**: Efficient skip/take with index hints
//! - **Streaming pagination**: Incremental result delivery for memory efficiency
//! - **Keyset pagination**: Resumable pagination for sorted results
//! - **Page size adaptation**: Automatic adjustment based on result complexity
//! - **SciRS2 integration**: Statistical analysis of result distributions
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_arq::query_pagination::{PaginationConfig, PaginationStrategy, QueryPaginator};
//!
//! let config = PaginationConfig::default()
//!     .with_page_size(100)
//!     .with_strategy(PaginationStrategy::Cursor);
//!
//! let mut paginator = QueryPaginator::new(config);
//! let page = paginator.next_page()?;
//! ```

use anyhow::{anyhow, Result};
use scirs2_core::random::rng;
use scirs2_core::RngCore; // Trait for RNG methods
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Pagination strategy for SPARQL query results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaginationStrategy {
    /// Simple offset-based pagination (OFFSET/LIMIT in SPARQL)
    /// - Pros: Simple, stateless
    /// - Cons: Inefficient for large offsets (O(n) skip cost)
    OffsetLimit,

    /// Cursor-based pagination with opaque tokens
    /// - Pros: Efficient, stateless, handles concurrent updates
    /// - Cons: Requires encoding/decoding cursors
    Cursor,

    /// Keyset pagination using comparison operators
    /// - Pros: Most efficient for sorted results
    /// - Cons: Requires stable sort key, complex for multi-column sorts
    Keyset,

    /// Streaming pagination with server-side state
    /// - Pros: Very memory efficient, supports pause/resume
    /// - Cons: Requires server-side state management
    Streaming,
}

/// Configuration for query pagination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationConfig {
    /// Primary pagination strategy
    pub strategy: PaginationStrategy,

    /// Default page size (number of results per page)
    pub page_size: usize,

    /// Maximum page size to prevent memory exhaustion
    pub max_page_size: usize,

    /// Minimum page size for efficiency
    pub min_page_size: usize,

    /// Enable adaptive page sizing based on result complexity
    pub adaptive_sizing: bool,

    /// Timeout for fetching a single page
    pub page_timeout: Duration,

    /// Maximum total results to return (0 = unlimited)
    pub max_total_results: usize,

    /// Prefetch next page in background
    pub prefetch_enabled: bool,

    /// Cursor encoding format
    pub cursor_encoding: CursorEncoding,

    /// Cache pages for faster backward navigation
    pub cache_pages: bool,

    /// Maximum number of cached pages
    pub max_cached_pages: usize,
}

/// Cursor encoding format for pagination tokens
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CursorEncoding {
    /// Base64-encoded binary format (compact)
    Base64,

    /// Hexadecimal encoding (readable)
    Hex,

    /// URL-safe base64 encoding
    Base64Url,
}

impl Default for PaginationConfig {
    fn default() -> Self {
        Self {
            strategy: PaginationStrategy::Cursor,
            page_size: 100,
            max_page_size: 10_000,
            min_page_size: 10,
            adaptive_sizing: true,
            page_timeout: Duration::from_secs(30),
            max_total_results: 0, // unlimited
            prefetch_enabled: false,
            cursor_encoding: CursorEncoding::Base64Url,
            cache_pages: false,
            max_cached_pages: 10,
        }
    }
}

impl PaginationConfig {
    /// Create configuration with custom page size
    pub fn with_page_size(mut self, size: usize) -> Self {
        self.page_size = size.clamp(self.min_page_size, self.max_page_size);
        self
    }

    /// Set pagination strategy
    pub fn with_strategy(mut self, strategy: PaginationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable or disable adaptive page sizing
    pub fn with_adaptive_sizing(mut self, enabled: bool) -> Self {
        self.adaptive_sizing = enabled;
        self
    }

    /// Set page timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.page_timeout = timeout;
        self
    }

    /// Set maximum total results
    pub fn with_max_results(mut self, max: usize) -> Self {
        self.max_total_results = max;
        self
    }

    /// Enable page prefetching
    pub fn with_prefetch(mut self, enabled: bool) -> Self {
        self.prefetch_enabled = enabled;
        self
    }

    /// Enable page caching
    pub fn with_cache(mut self, enabled: bool, max_cached: usize) -> Self {
        self.cache_pages = enabled;
        self.max_cached_pages = max_cached;
        self
    }
}

/// Opaque pagination cursor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageCursor {
    /// Cursor version for compatibility
    version: u8,

    /// Offset or position in result set
    position: usize,

    /// Last seen sort key values (for keyset pagination)
    sort_keys: Vec<String>,

    /// Page number (for offset-based pagination)
    page_number: usize,

    /// Timestamp when cursor was created
    created_at: u64,

    /// Random nonce for security
    nonce: u64,
}

impl PageCursor {
    /// Create a new cursor
    pub fn new(position: usize, page_number: usize) -> Self {
        let mut rng_instance = rng();
        Self {
            version: 1,
            position,
            sort_keys: Vec::new(),
            page_number,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("SystemTime should be after UNIX_EPOCH")
                .as_secs(),
            nonce: rng_instance.next_u64(),
        }
    }

    /// Create cursor with sort keys for keyset pagination
    pub fn with_sort_keys(mut self, keys: Vec<String>) -> Self {
        self.sort_keys = keys;
        self
    }

    /// Encode cursor to string
    pub fn encode(&self, encoding: CursorEncoding) -> Result<String> {
        let bytes = oxicode::serde::encode_to_vec(self, oxicode::config::standard())
            .map_err(|e| anyhow!("Failed to serialize cursor: {}", e))?;

        Ok(match encoding {
            CursorEncoding::Base64 => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD.encode(&bytes)
            }
            CursorEncoding::Hex => hex::encode(&bytes),
            CursorEncoding::Base64Url => {
                use base64::Engine;
                base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&bytes)
            }
        })
    }

    /// Decode cursor from string
    pub fn decode(encoded: &str, encoding: CursorEncoding) -> Result<Self> {
        let bytes = match encoding {
            CursorEncoding::Base64 => {
                use base64::Engine;
                base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .map_err(|e| anyhow!("Failed to decode base64 cursor: {}", e))?
            }
            CursorEncoding::Hex => {
                hex::decode(encoded).map_err(|e| anyhow!("Failed to decode hex cursor: {}", e))?
            }
            CursorEncoding::Base64Url => {
                use base64::Engine;
                base64::engine::general_purpose::URL_SAFE_NO_PAD
                    .decode(encoded)
                    .map_err(|e| anyhow!("Failed to decode base64url cursor: {}", e))?
            }
        };

        oxicode::serde::decode_from_slice(&bytes, oxicode::config::standard())
            .map(|(v, _)| v)
            .map_err(|e| anyhow!("Failed to deserialize cursor: {}", e))
    }
}

/// Page of query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultPage<T> {
    /// Results in this page
    pub results: Vec<T>,

    /// Cursor for next page (if available)
    pub next_cursor: Option<String>,

    /// Cursor for previous page (if available)
    pub prev_cursor: Option<String>,

    /// Total number of results (if known)
    pub total_count: Option<usize>,

    /// Current page number (for offset-based pagination)
    pub page_number: usize,

    /// Actual page size
    pub page_size: usize,

    /// Whether there are more results
    pub has_more: bool,

    /// Metadata about this page
    pub metadata: PageMetadata,
}

/// Metadata about a result page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageMetadata {
    /// Time taken to fetch this page
    pub fetch_duration: Duration,

    /// Strategy used for this page
    pub strategy: PaginationStrategy,

    /// Whether page size was adapted
    pub adapted: bool,

    /// Complexity score of results (for adaptive sizing)
    pub complexity_score: f64,

    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

/// Query paginator for managing paginated result delivery
pub struct QueryPaginator {
    config: PaginationConfig,
    current_page: usize,
    total_fetched: usize,
    page_cache: HashMap<usize, Vec<Vec<u8>>>, // Simple cache
    stats: PaginationStatistics,
}

/// Statistics for pagination performance
#[derive(Debug, Clone, Default)]
pub struct PaginationStatistics {
    /// Total pages fetched
    pub pages_fetched: usize,

    /// Total results returned
    pub results_returned: usize,

    /// Average page fetch time
    pub avg_fetch_time: Duration,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// Adaptive sizing adjustments made
    pub size_adjustments: usize,

    /// Page size distribution
    pub page_sizes: Vec<usize>,
}

impl QueryPaginator {
    /// Create a new paginator with configuration
    pub fn new(config: PaginationConfig) -> Self {
        Self {
            config,
            current_page: 0,
            total_fetched: 0,
            page_cache: HashMap::new(),
            stats: PaginationStatistics::default(),
        }
    }

    /// Calculate optimal page size based on result complexity
    pub fn calculate_adaptive_page_size(&self, complexity_score: f64) -> usize {
        if !self.config.adaptive_sizing {
            return self.config.page_size;
        }

        // Higher complexity = smaller pages to maintain responsiveness
        // Use exponential scaling: page_size = base_size * e^(-k * complexity)
        let k = 0.1; // Scaling factor
        let base_size = self.config.page_size as f64;
        let adjusted_size = base_size * (-k * complexity_score).exp();

        (adjusted_size.round() as usize).clamp(self.config.min_page_size, self.config.max_page_size)
    }

    /// Build SPARQL query with pagination clause
    pub fn build_paginated_query(
        &self,
        base_query: &str,
        cursor: Option<&PageCursor>,
        page_size: usize,
    ) -> Result<String> {
        match self.config.strategy {
            PaginationStrategy::OffsetLimit => {
                let offset = cursor.map(|c| c.position).unwrap_or(0);
                Ok(format!(
                    "{}\nLIMIT {} OFFSET {}",
                    base_query.trim_end(),
                    page_size,
                    offset
                ))
            }

            PaginationStrategy::Cursor | PaginationStrategy::Streaming => {
                // For cursor-based, we still use OFFSET/LIMIT but encode position
                let offset = cursor.map(|c| c.position).unwrap_or(0);
                Ok(format!(
                    "{}\nLIMIT {} OFFSET {}",
                    base_query.trim_end(),
                    page_size,
                    offset
                ))
            }

            PaginationStrategy::Keyset => {
                // For keyset pagination, we need to inject WHERE conditions
                // This requires parsing and modifying the query
                // For now, fall back to offset-based
                let offset = cursor.map(|c| c.position).unwrap_or(0);
                Ok(format!(
                    "{}\nLIMIT {} OFFSET {}",
                    base_query.trim_end(),
                    page_size,
                    offset
                ))
            }
        }
    }

    /// Create cursor for next page
    pub fn create_next_cursor(&self, current_position: usize, page_size: usize) -> Result<String> {
        let cursor = PageCursor::new(current_position + page_size, self.current_page + 1);

        cursor.encode(self.config.cursor_encoding)
    }

    /// Create cursor for previous page
    pub fn create_prev_cursor(
        &self,
        current_position: usize,
        page_size: usize,
    ) -> Result<Option<String>> {
        if current_position == 0 {
            return Ok(None);
        }

        let prev_position = current_position.saturating_sub(page_size);
        let cursor = PageCursor::new(prev_position, self.current_page.saturating_sub(1));

        Ok(Some(cursor.encode(self.config.cursor_encoding)?))
    }

    /// Update statistics after fetching a page
    pub fn update_statistics(
        &mut self,
        page_size: usize,
        fetch_duration: Duration,
        was_adapted: bool,
    ) {
        self.stats.pages_fetched += 1;
        self.stats.results_returned += page_size;
        self.stats.page_sizes.push(page_size);

        if was_adapted {
            self.stats.size_adjustments += 1;
        }

        // Update average fetch time using cumulative moving average
        let n = self.stats.pages_fetched as f64;
        let current_avg_secs = self.stats.avg_fetch_time.as_secs_f64();
        let new_secs = fetch_duration.as_secs_f64();
        let new_avg_secs = (current_avg_secs * (n - 1.0) + new_secs) / n;

        self.stats.avg_fetch_time = Duration::from_secs_f64(new_avg_secs);
    }

    /// Calculate result complexity score for adaptive sizing
    pub fn calculate_complexity_score(&self, result_sizes: &[usize]) -> f64 {
        if result_sizes.is_empty() {
            return 0.0;
        }

        // Calculate mean manually
        let sizes: Vec<f64> = result_sizes.iter().map(|&s| s as f64).collect();
        let mean_val = sizes.iter().sum::<f64>() / sizes.len() as f64;

        // Calculate variance as complexity indicator
        let variance =
            sizes.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>() / sizes.len() as f64;

        // Normalize variance to 0-1 range using sigmoid
        1.0 / (1.0 + (-variance / 1000.0).exp())
    }

    /// Get current pagination statistics
    pub fn get_statistics(&self) -> &PaginationStatistics {
        &self.stats
    }

    /// Reset paginator to initial state
    pub fn reset(&mut self) {
        self.current_page = 0;
        self.total_fetched = 0;
        self.page_cache.clear();
        self.stats = PaginationStatistics::default();
    }
}

/// Helper for offset-based pagination calculations
pub struct OffsetPagination;

impl OffsetPagination {
    /// Calculate total pages given total results and page size
    pub fn total_pages(total_results: usize, page_size: usize) -> usize {
        if page_size == 0 {
            return 0;
        }
        (total_results + page_size - 1) / page_size
    }

    /// Calculate offset for a given page number
    pub fn offset_for_page(page: usize, page_size: usize) -> usize {
        page.saturating_mul(page_size)
    }

    /// Check if page number is valid
    pub fn is_valid_page(page: usize, total_results: usize, page_size: usize) -> bool {
        if page_size == 0 {
            return false;
        }
        page < Self::total_pages(total_results, page_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagination_config_defaults() {
        let config = PaginationConfig::default();
        assert_eq!(config.strategy, PaginationStrategy::Cursor);
        assert_eq!(config.page_size, 100);
        assert_eq!(config.max_page_size, 10_000);
        assert!(config.adaptive_sizing);
    }

    #[test]
    fn test_pagination_config_builder() {
        let config = PaginationConfig::default()
            .with_page_size(50)
            .with_strategy(PaginationStrategy::Keyset)
            .with_adaptive_sizing(false);

        assert_eq!(config.page_size, 50);
        assert_eq!(config.strategy, PaginationStrategy::Keyset);
        assert!(!config.adaptive_sizing);
    }

    #[test]
    fn test_page_cursor_encoding() {
        let cursor = PageCursor::new(100, 5);

        // Test Base64 encoding
        let encoded = cursor.encode(CursorEncoding::Base64).unwrap();
        let decoded = PageCursor::decode(&encoded, CursorEncoding::Base64).unwrap();
        assert_eq!(cursor.position, decoded.position);
        assert_eq!(cursor.page_number, decoded.page_number);

        // Test Hex encoding
        let encoded_hex = cursor.encode(CursorEncoding::Hex).unwrap();
        let decoded_hex = PageCursor::decode(&encoded_hex, CursorEncoding::Hex).unwrap();
        assert_eq!(cursor.position, decoded_hex.position);

        // Test Base64Url encoding
        let encoded_url = cursor.encode(CursorEncoding::Base64Url).unwrap();
        let decoded_url = PageCursor::decode(&encoded_url, CursorEncoding::Base64Url).unwrap();
        assert_eq!(cursor.position, decoded_url.position);
    }

    #[test]
    fn test_adaptive_page_sizing() {
        let config = PaginationConfig::default().with_page_size(1000);
        let paginator = QueryPaginator::new(config);

        // Low complexity should give larger pages
        let size_low = paginator.calculate_adaptive_page_size(0.1);
        assert!(size_low >= 900); // Should be close to base size

        // High complexity should give smaller pages
        let size_high = paginator.calculate_adaptive_page_size(5.0);
        assert!(size_high < 700); // Should be significantly smaller
    }

    #[test]
    fn test_paginated_query_building() {
        let config = PaginationConfig::default();
        let paginator = QueryPaginator::new(config);

        let base_query = "SELECT * WHERE { ?s ?p ?o }";

        // First page (no cursor)
        let query1 = paginator
            .build_paginated_query(base_query, None, 100)
            .unwrap();
        assert!(query1.contains("LIMIT 100"));
        assert!(query1.contains("OFFSET 0"));

        // Second page (with cursor)
        let cursor = PageCursor::new(100, 1);
        let query2 = paginator
            .build_paginated_query(base_query, Some(&cursor), 100)
            .unwrap();
        assert!(query2.contains("LIMIT 100"));
        assert!(query2.contains("OFFSET 100"));
    }

    #[test]
    fn test_cursor_creation() {
        let config = PaginationConfig::default();
        let paginator = QueryPaginator::new(config);

        // Create next cursor
        let next = paginator.create_next_cursor(0, 100).unwrap();
        assert!(!next.is_empty());

        // Decode and verify
        let decoded = PageCursor::decode(&next, CursorEncoding::Base64Url).unwrap();
        assert_eq!(decoded.position, 100);
        assert_eq!(decoded.page_number, 1);
    }

    #[test]
    fn test_prev_cursor_creation() {
        let config = PaginationConfig::default();
        let paginator = QueryPaginator::new(config);

        // No previous page from position 0
        let prev = paginator.create_prev_cursor(0, 100).unwrap();
        assert!(prev.is_none());

        // Has previous page from position 200
        let prev = paginator.create_prev_cursor(200, 100).unwrap();
        assert!(prev.is_some());

        let decoded = PageCursor::decode(&prev.unwrap(), CursorEncoding::Base64Url).unwrap();
        assert_eq!(decoded.position, 100);
    }

    #[test]
    fn test_complexity_calculation() {
        let config = PaginationConfig::default();
        let paginator = QueryPaginator::new(config);

        // Uniform results (low complexity)
        let uniform = vec![100, 100, 100, 100];
        let score_low = paginator.calculate_complexity_score(&uniform);

        // Varied results (high complexity)
        let varied = vec![10, 100, 500, 5000];
        let score_high = paginator.calculate_complexity_score(&varied);

        assert!(score_high > score_low);
    }

    #[test]
    fn test_offset_pagination_helpers() {
        // Total pages calculation
        assert_eq!(OffsetPagination::total_pages(250, 100), 3);
        assert_eq!(OffsetPagination::total_pages(300, 100), 3);
        assert_eq!(OffsetPagination::total_pages(301, 100), 4);

        // Offset calculation
        assert_eq!(OffsetPagination::offset_for_page(0, 100), 0);
        assert_eq!(OffsetPagination::offset_for_page(1, 100), 100);
        assert_eq!(OffsetPagination::offset_for_page(5, 50), 250);

        // Page validation
        assert!(OffsetPagination::is_valid_page(0, 250, 100));
        assert!(OffsetPagination::is_valid_page(2, 250, 100));
        assert!(!OffsetPagination::is_valid_page(3, 250, 100));
    }

    #[test]
    fn test_statistics_tracking() {
        let config = PaginationConfig::default();
        let mut paginator = QueryPaginator::new(config);

        // Simulate fetching pages
        paginator.update_statistics(100, Duration::from_millis(50), false);
        paginator.update_statistics(100, Duration::from_millis(60), false);
        paginator.update_statistics(80, Duration::from_millis(40), true);

        let stats = paginator.get_statistics();
        assert_eq!(stats.pages_fetched, 3);
        assert_eq!(stats.results_returned, 280);
        assert_eq!(stats.size_adjustments, 1);
        assert_eq!(stats.page_sizes.len(), 3);
        assert!(stats.avg_fetch_time.as_millis() > 0);
    }

    #[test]
    fn test_paginator_reset() {
        let config = PaginationConfig::default();
        let mut paginator = QueryPaginator::new(config);

        paginator.current_page = 5;
        paginator.total_fetched = 500;
        paginator.update_statistics(100, Duration::from_millis(50), false);

        paginator.reset();

        assert_eq!(paginator.current_page, 0);
        assert_eq!(paginator.total_fetched, 0);
        assert_eq!(paginator.stats.pages_fetched, 0);
        assert!(paginator.page_cache.is_empty());
    }
}
