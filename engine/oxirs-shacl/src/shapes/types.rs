//! Supporting types and statistics for SHACL shape processing

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Statistics about shape parsing cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeCacheStats {
    pub entries: usize,
    pub total_constraints: usize,
}

/// Shape validation context during parsing
#[derive(Debug, Clone)]
pub struct ShapeParsingContext {
    /// Current parsing depth
    pub depth: usize,

    /// Visited shape IRIs (for circular reference detection)
    pub visited: HashSet<String>,

    /// Parsing configuration
    pub config: ShapeParsingConfig,

    /// Performance statistics
    pub stats: ShapeParsingStats,
}

impl Default for ShapeParsingContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ShapeParsingContext {
    /// Create a new parsing context with default configuration
    pub fn new() -> Self {
        Self {
            depth: 0,
            visited: HashSet::new(),
            config: ShapeParsingConfig::default(),
            stats: ShapeParsingStats::new(),
        }
    }
}

/// Configuration for shape parsing
#[derive(Debug, Clone)]
pub struct ShapeParsingConfig {
    /// Maximum recursion depth for shape parsing
    pub max_depth: usize,
    /// Enable strict parsing mode
    pub strict_mode: bool,
    /// Enable performance tracking
    pub enable_performance_tracking: bool,
    /// Cache parsed shapes
    pub enable_caching: bool,
    /// Namespace prefixes for IRI resolution
    pub namespaces: HashMap<String, String>,
}

impl Default for ShapeParsingConfig {
    fn default() -> Self {
        Self {
            max_depth: 50,
            strict_mode: false,
            enable_performance_tracking: true,
            enable_caching: true,
            namespaces: HashMap::new(),
        }
    }
}

/// Shape parsing performance statistics
#[derive(Debug, Clone)]
pub struct ShapeParsingStats {
    pub total_shapes_parsed: usize,
    pub total_constraints_parsed: usize,
    pub parsing_time: Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl ShapeParsingStats {
    pub fn new() -> Self {
        Self {
            total_shapes_parsed: 0,
            total_constraints_parsed: 0,
            parsing_time: Duration::from_millis(0),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_requests as f64
        }
    }

    /// Average parsing time per shape
    pub fn avg_parsing_time_per_shape(&self) -> Duration {
        if self.total_shapes_parsed == 0 {
            Duration::from_millis(0)
        } else {
            self.parsing_time / self.total_shapes_parsed as u32
        }
    }

    /// Update statistics after parsing a shape
    pub fn update_shape_parsed(&mut self, constraints_count: usize, duration: Duration) {
        self.total_shapes_parsed += 1;
        self.total_constraints_parsed += constraints_count;
        self.parsing_time += duration;
    }

    /// Record cache hit
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record cache miss
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }
}

impl Default for ShapeParsingStats {
    fn default() -> Self {
        Self::new()
    }
}
