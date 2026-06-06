//! Storage-related OxiRS configuration types.
//!
//! Contains `IndexingConfig`, `ParsingConfig`, `SerializationConfig`,
//! `CachingConfig`, and all subordinate types + Default impls.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────
// IndexingConfig
// ─────────────────────────────────────────────────────────────

/// Indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    /// Default indexing strategy
    pub default_strategy: IndexingStrategy,
    /// Strategy-specific configurations
    pub strategy_configs: HashMap<String, IndexStrategyConfig>,
    /// Adaptive indexing settings
    pub adaptive: AdaptiveIndexingConfig,
    /// Index persistence settings
    pub persistence: IndexPersistenceConfig,
}

/// Indexing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexingStrategy {
    /// No indexing
    None,
    /// Single index (SPO only)
    Single,
    /// Dual index (SPO + POS)
    Dual,
    /// Triple index (SPO + POS + OSP)
    Triple,
    /// Adaptive multi-index
    AdaptiveMulti,
    /// Custom indexing strategy
    Custom,
}

/// Index strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStrategyConfig {
    /// Strategy name
    pub name: String,
    /// Index types to create
    pub index_types: Vec<IndexType>,
    /// Bloom filter settings
    pub bloom_filter: BloomFilterConfig,
    /// Index compaction settings
    pub compaction: IndexCompactionConfig,
}

/// Index types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    /// Subject-Predicate-Object index
    SPO,
    /// Predicate-Object-Subject index
    POS,
    /// Object-Subject-Predicate index
    OSP,
    /// Subject-Object-Predicate index
    SOP,
    /// Predicate-Subject-Object index
    PSO,
    /// Object-Predicate-Subject index
    OPS,
}

/// Bloom filter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilterConfig {
    /// Enable bloom filters for indexes
    pub enabled: bool,
    /// Expected number of items
    pub expected_items: usize,
    /// False positive probability
    pub false_positive_rate: f64,
    /// Hash function count
    pub hash_functions: usize,
}

/// Index compaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexCompactionConfig {
    /// Enable automatic compaction
    pub enabled: bool,
    /// Compaction threshold (fragmentation %)
    pub threshold: f64,
    /// Compaction interval
    pub interval: Duration,
    /// Concurrent compaction
    pub concurrent: bool,
}

/// Adaptive indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveIndexingConfig {
    /// Enable adaptive indexing
    pub enabled: bool,
    /// Query pattern analysis window
    pub analysis_window: Duration,
    /// Minimum query frequency for index creation
    pub min_query_frequency: f64,
    /// Index effectiveness threshold
    pub effectiveness_threshold: f64,
}

/// Index persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexPersistenceConfig {
    /// Enable index persistence
    pub enabled: bool,
    /// Persistence directory
    pub directory: PathBuf,
    /// Sync interval
    pub sync_interval: Duration,
    /// Compression enabled
    pub compression: bool,
}

// ─────────────────────────────────────────────────────────────
// ParsingConfig
// ─────────────────────────────────────────────────────────────

/// Parser configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsingConfig {
    /// Default buffer sizes for different formats
    pub buffer_sizes: HashMap<String, usize>,
    /// Parser-specific configurations
    pub parsers: HashMap<String, ParserConfig>,
    /// Error handling configuration
    pub error_handling: ParserErrorConfig,
    /// Validation settings
    pub validation: ValidationConfig,
}

/// Individual parser configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParserConfig {
    /// Parser name
    pub name: String,
    /// Enable streaming mode
    pub enable_streaming: bool,
    /// Chunk size for streaming
    pub chunk_size: usize,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Worker thread count
    pub worker_threads: usize,
    /// Parser-specific options
    pub options: HashMap<String, serde_json::Value>,
}

/// Parser error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParserErrorConfig {
    /// Error tolerance (percentage of errors allowed)
    pub tolerance: f64,
    /// Continue parsing after errors
    pub continue_on_error: bool,
    /// Collect error details
    pub collect_errors: bool,
    /// Maximum error count before stopping
    pub max_errors: usize,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable IRI validation
    pub enable_iri_validation: bool,
    /// Enable literal validation
    pub enable_literal_validation: bool,
    /// Enable language tag validation
    pub enable_language_validation: bool,
    /// Custom validation rules
    pub custom_rules: Vec<ValidationRule>,
}

/// Custom validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule pattern (regex)
    pub pattern: String,
    /// Error message
    pub error_message: String,
    /// Rule enabled
    pub enabled: bool,
}

// ─────────────────────────────────────────────────────────────
// SerializationConfig
// ─────────────────────────────────────────────────────────────

/// Serialization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationConfig {
    /// Default serialization format
    pub default_format: SerializationFormat,
    /// Format-specific configurations
    pub formats: HashMap<String, FormatConfig>,
    /// Output settings
    pub output: OutputConfig,
    /// Compression settings
    pub compression: CompressionConfig,
}

/// Serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationFormat {
    NTriples,
    Turtle,
    RdfXml,
    JsonLd,
    NQuads,
    TriG,
}

/// Format-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatConfig {
    /// Format name
    pub name: String,
    /// Pretty printing enabled
    pub pretty_print: bool,
    /// Indentation size
    pub indent_size: usize,
    /// Line length limit
    pub line_length: usize,
    /// Format-specific options
    pub options: HashMap<String, serde_json::Value>,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Default encoding
    pub encoding: String,
    /// Buffer size
    pub buffer_size: usize,
    /// Enable buffering
    pub enable_buffering: bool,
    /// Flush interval
    pub flush_interval: Duration,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub level: u8,
    /// Minimum size for compression
    pub min_size: usize,
}

/// Compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Bzip2,
    Lz4,
    Zstd,
}

// ─────────────────────────────────────────────────────────────
// CachingConfig
// ─────────────────────────────────────────────────────────────

/// Caching configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Cache configurations by name
    pub caches: HashMap<String, CacheConfig>,
    /// Global cache settings
    pub global: GlobalCacheConfig,
}

/// Individual cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Cache name
    pub name: String,
    /// Cache type
    pub cache_type: CacheType,
    /// Maximum size
    pub max_size: usize,
    /// TTL (time to live)
    pub ttl: Duration,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
}

/// Cache types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheType {
    LRU,
    LFU,
    FIFO,
    Random,
    Adaptive,
}

/// Eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    TTL,
    Size,
    Custom,
}

/// Global cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCacheConfig {
    /// Total cache memory limit
    pub memory_limit: usize,
    /// Enable cache statistics
    pub enable_statistics: bool,
    /// Cache warming enabled
    pub enable_warming: bool,
}

// ─────────────────────────────────────────────────────────────
// Default implementations
// ─────────────────────────────────────────────────────────────

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            default_strategy: IndexingStrategy::AdaptiveMulti,
            strategy_configs: HashMap::new(),
            adaptive: AdaptiveIndexingConfig::default(),
            persistence: IndexPersistenceConfig::default(),
        }
    }
}

impl Default for AdaptiveIndexingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analysis_window: Duration::from_secs(3600),
            min_query_frequency: 0.1,
            effectiveness_threshold: 0.8,
        }
    }
}

impl Default for IndexPersistenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            directory: PathBuf::from("./indexes"),
            sync_interval: Duration::from_secs(300),
            compression: true,
        }
    }
}

impl Default for ParsingConfig {
    fn default() -> Self {
        Self {
            buffer_sizes: HashMap::from([
                ("ntriples".to_string(), 64 * 1024),
                ("turtle".to_string(), 32 * 1024),
                ("rdfxml".to_string(), 128 * 1024),
                ("jsonld".to_string(), 64 * 1024),
            ]),
            parsers: HashMap::new(),
            error_handling: ParserErrorConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

impl Default for ParserErrorConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.01,
            continue_on_error: true,
            collect_errors: true,
            max_errors: 1000,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_iri_validation: true,
            enable_literal_validation: true,
            enable_language_validation: true,
            custom_rules: Vec::new(),
        }
    }
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            default_format: SerializationFormat::Turtle,
            formats: HashMap::new(),
            output: OutputConfig::default(),
            compression: CompressionConfig::default(),
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            encoding: "UTF-8".to_string(),
            buffer_size: 8192,
            enable_buffering: true,
            flush_interval: Duration::from_millis(100),
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionAlgorithm::Gzip,
            level: 6,
            min_size: 1024,
        }
    }
}

impl Default for GlobalCacheConfig {
    fn default() -> Self {
        Self {
            memory_limit: 256 * 1024 * 1024, // 256MB
            enable_statistics: true,
            enable_warming: true,
        }
    }
}
