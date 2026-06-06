//! Core performance, memory, and interning configuration types for OxiRS.
//!
//! Contains `PerformanceConfig`, `PerformanceProfile`, `PerformanceValue`,
//! `MemoryConfig`, `InterningConfig`, and all subordinate types + Default impls.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::time::Duration;

// ─────────────────────────────────────────────────────────────
// PerformanceConfig
// ─────────────────────────────────────────────────────────────

/// Performance profile configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Selected performance profile
    pub profile: PerformanceProfile,
    /// Custom performance settings (overrides profile defaults)
    pub custom_settings: HashMap<String, PerformanceValue>,
    /// Enable auto-tuning based on runtime metrics
    pub enable_auto_tuning: bool,
    /// Auto-tuning sensitivity (0.0 to 1.0)
    pub auto_tuning_sensitivity: f64,
    /// Performance monitoring interval
    pub monitoring_interval: Duration,
}

/// Available performance profiles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceProfile {
    /// Development profile - prioritizes compilation speed and debugging
    Development,
    /// Balanced profile - good performance with reasonable resource usage
    Balanced,
    /// High performance profile - optimized for speed
    HighPerformance,
    /// Maximum throughput profile - all optimizations enabled
    MaxThroughput,
    /// Memory efficient profile - minimizes memory usage
    MemoryEfficient,
    /// Low latency profile - optimized for quick response times
    LowLatency,
    /// Batch processing profile - optimized for large dataset processing
    BatchProcessing,
    /// Real-time profile - deterministic performance with bounded latency
    RealTime,
    /// Edge computing profile - optimized for resource-constrained environments
    EdgeComputing,
    /// Custom profile - user-defined settings
    Custom,
}

/// Performance configuration values
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PerformanceValue {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Duration(u64), // milliseconds
}

// ─────────────────────────────────────────────────────────────
// MemoryConfig
// ─────────────────────────────────────────────────────────────

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Arena allocator settings
    pub arena: ArenaConfig,
    /// Garbage collection settings
    pub gc: GcConfig,
    /// Memory pressure thresholds
    pub pressure_thresholds: MemoryPressureConfig,
    /// Enable memory tracking
    pub enable_tracking: bool,
    /// Memory pool configurations
    pub pools: HashMap<String, MemoryPoolConfig>,
}

/// Arena allocator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaConfig {
    /// Initial arena size in bytes
    pub initial_size: usize,
    /// Maximum arena size in bytes
    pub max_size: usize,
    /// Arena growth factor
    pub growth_factor: f64,
    /// Enable arena compaction
    pub enable_compaction: bool,
    /// Compaction threshold (utilization %)
    pub compaction_threshold: f64,
}

/// Garbage collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcConfig {
    /// GC strategy
    pub strategy: GcStrategy,
    /// GC trigger threshold (memory utilization %)
    pub trigger_threshold: f64,
    /// Maximum GC pause time
    pub max_pause_time: Duration,
    /// Enable concurrent GC
    pub enable_concurrent: bool,
    /// GC worker thread count
    pub worker_threads: usize,
}

/// Garbage collection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GcStrategy {
    /// No automatic garbage collection
    None,
    /// Reference counting with cycle detection
    ReferenceCounting,
    /// Mark and sweep collector
    MarkAndSweep,
    /// Generational garbage collector
    Generational,
    /// Incremental garbage collector
    Incremental,
}

/// Memory pressure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureConfig {
    /// Low pressure threshold (% of available memory)
    pub low_threshold: f64,
    /// Medium pressure threshold
    pub medium_threshold: f64,
    /// High pressure threshold
    pub high_threshold: f64,
    /// Critical pressure threshold
    pub critical_threshold: f64,
    /// Actions to take at each pressure level
    pub pressure_actions: HashMap<String, Vec<MemoryPressureAction>>,
}

/// Memory pressure actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPressureAction {
    /// Force garbage collection
    ForceGc,
    /// Compact arenas
    CompactArenas,
    /// Clear caches
    ClearCaches,
    /// Reduce buffer sizes
    ReduceBuffers,
    /// Suspend background tasks
    SuspendBackgroundTasks,
    /// Send alert
    SendAlert,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Pool name
    pub name: String,
    /// Initial pool size
    pub initial_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Object size for this pool
    pub object_size: usize,
    /// Enable pre-allocation
    pub enable_preallocation: bool,
}

// ─────────────────────────────────────────────────────────────
// InterningConfig
// ─────────────────────────────────────────────────────────────

/// String interning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterningConfig {
    /// Global interner settings
    pub global: GlobalInternerConfig,
    /// Scoped interner settings
    pub scoped: ScopedInternerConfig,
    /// Interner cleanup settings
    pub cleanup: InternerCleanupConfig,
    /// Enable interner statistics
    pub enable_statistics: bool,
}

/// Global interner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalInternerConfig {
    /// Initial capacity
    pub initial_capacity: usize,
    /// Load factor threshold for resizing
    pub load_factor: f64,
    /// Enable weak references
    pub enable_weak_references: bool,
    /// LRU cache size for frequently accessed strings
    pub lru_cache_size: usize,
}

/// Scoped interner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedInternerConfig {
    /// Default scope capacity
    pub default_capacity: usize,
    /// Maximum number of active scopes
    pub max_scopes: usize,
    /// Scope timeout (automatic cleanup)
    pub scope_timeout: Duration,
}

/// Interner cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternerCleanupConfig {
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Cleanup threshold (unused string percentage)
    pub cleanup_threshold: f64,
    /// Enable automatic cleanup
    pub enable_automatic: bool,
}

// ─────────────────────────────────────────────────────────────
// Default implementations
// ─────────────────────────────────────────────────────────────

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            profile: PerformanceProfile::Balanced,
            custom_settings: HashMap::new(),
            enable_auto_tuning: false,
            auto_tuning_sensitivity: 0.5,
            monitoring_interval: Duration::from_secs(60),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            arena: ArenaConfig::default(),
            gc: GcConfig::default(),
            pressure_thresholds: MemoryPressureConfig::default(),
            enable_tracking: true,
            pools: HashMap::new(),
        }
    }
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            initial_size: 1024 * 1024,  // 1MB
            max_size: 64 * 1024 * 1024, // 64MB
            growth_factor: 2.0,
            enable_compaction: true,
            compaction_threshold: 0.5,
        }
    }
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            strategy: GcStrategy::MarkAndSweep,
            trigger_threshold: 0.8,
            max_pause_time: Duration::from_millis(10),
            enable_concurrent: true,
            worker_threads: 2,
        }
    }
}

impl Default for MemoryPressureConfig {
    fn default() -> Self {
        Self {
            low_threshold: 0.6,
            medium_threshold: 0.75,
            high_threshold: 0.9,
            critical_threshold: 0.95,
            pressure_actions: HashMap::new(),
        }
    }
}

impl Default for InterningConfig {
    fn default() -> Self {
        Self {
            global: GlobalInternerConfig::default(),
            scoped: ScopedInternerConfig::default(),
            cleanup: InternerCleanupConfig::default(),
            enable_statistics: true,
        }
    }
}

impl Default for GlobalInternerConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 10000,
            load_factor: 0.75,
            enable_weak_references: true,
            lru_cache_size: 1000,
        }
    }
}

impl Default for ScopedInternerConfig {
    fn default() -> Self {
        Self {
            default_capacity: 1000,
            max_scopes: 100,
            scope_timeout: Duration::from_secs(3600),
        }
    }
}

impl Default for InternerCleanupConfig {
    fn default() -> Self {
        Self {
            cleanup_interval: Duration::from_secs(300),
            cleanup_threshold: 0.5,
            enable_automatic: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Display / method impls for PerformanceProfile
// ─────────────────────────────────────────────────────────────

impl PerformanceProfile {
    /// Get performance configuration for this profile
    pub fn get_config(&self) -> HashMap<String, PerformanceValue> {
        let mut config = HashMap::new();

        match self {
            Self::Development => {
                config.insert("enable_debug".to_string(), PerformanceValue::Boolean(true));
                config.insert(
                    "optimization_level".to_string(),
                    PerformanceValue::Integer(0),
                );
                config.insert("enable_simd".to_string(), PerformanceValue::Boolean(false));
                config.insert("thread_count".to_string(), PerformanceValue::Integer(2));
            }
            Self::Balanced => {
                config.insert(
                    "optimization_level".to_string(),
                    PerformanceValue::Integer(2),
                );
                config.insert("enable_simd".to_string(), PerformanceValue::Boolean(true));
                config.insert("thread_count".to_string(), PerformanceValue::Integer(4));
                config.insert(
                    "memory_limit_mb".to_string(),
                    PerformanceValue::Integer(1024),
                );
            }
            Self::HighPerformance => {
                config.insert(
                    "optimization_level".to_string(),
                    PerformanceValue::Integer(3),
                );
                config.insert("enable_simd".to_string(), PerformanceValue::Boolean(true));
                config.insert(
                    "enable_zero_copy".to_string(),
                    PerformanceValue::Boolean(true),
                );
                config.insert("thread_count".to_string(), PerformanceValue::Integer(8));
                config.insert(
                    "memory_limit_mb".to_string(),
                    PerformanceValue::Integer(4096),
                );
            }
            Self::MaxThroughput => {
                config.insert(
                    "optimization_level".to_string(),
                    PerformanceValue::Integer(3),
                );
                config.insert("enable_simd".to_string(), PerformanceValue::Boolean(true));
                config.insert(
                    "enable_zero_copy".to_string(),
                    PerformanceValue::Boolean(true),
                );
                config.insert(
                    "enable_prefetching".to_string(),
                    PerformanceValue::Boolean(true),
                );
                config.insert("thread_count".to_string(), PerformanceValue::Integer(16));
                config.insert(
                    "memory_limit_mb".to_string(),
                    PerformanceValue::Integer(8192),
                );
            }
            Self::MemoryEfficient => {
                config.insert(
                    "optimization_level".to_string(),
                    PerformanceValue::Integer(1),
                );
                config.insert(
                    "enable_compression".to_string(),
                    PerformanceValue::Boolean(true),
                );
                config.insert(
                    "memory_limit_mb".to_string(),
                    PerformanceValue::Integer(256),
                );
                config.insert("gc_frequency".to_string(), PerformanceValue::Integer(10));
            }
            Self::LowLatency => {
                config.insert(
                    "optimization_level".to_string(),
                    PerformanceValue::Integer(3),
                );
                config.insert("enable_simd".to_string(), PerformanceValue::Boolean(true));
                config.insert(
                    "enable_zero_copy".to_string(),
                    PerformanceValue::Boolean(true),
                );
                config.insert(
                    "gc_strategy".to_string(),
                    PerformanceValue::String("incremental".to_string()),
                );
                config.insert(
                    "response_timeout_ms".to_string(),
                    PerformanceValue::Duration(100),
                );
            }
            Self::BatchProcessing => {
                config.insert(
                    "optimization_level".to_string(),
                    PerformanceValue::Integer(3),
                );
                config.insert(
                    "enable_parallel".to_string(),
                    PerformanceValue::Boolean(true),
                );
                config.insert("batch_size".to_string(), PerformanceValue::Integer(10000));
                config.insert("thread_count".to_string(), PerformanceValue::Integer(32));
            }
            Self::RealTime => {
                config.insert(
                    "enable_deterministic".to_string(),
                    PerformanceValue::Boolean(true),
                );
                config.insert("max_latency_ms".to_string(), PerformanceValue::Duration(10));
                config.insert(
                    "priority".to_string(),
                    PerformanceValue::String("realtime".to_string()),
                );
            }
            Self::EdgeComputing => {
                config.insert(
                    "optimization_level".to_string(),
                    PerformanceValue::Integer(2),
                );
                config.insert(
                    "memory_limit_mb".to_string(),
                    PerformanceValue::Integer(128),
                );
                config.insert("thread_count".to_string(), PerformanceValue::Integer(2));
                config.insert(
                    "enable_compression".to_string(),
                    PerformanceValue::Boolean(true),
                );
            }
            Self::Custom => {
                // Custom profiles are user-defined
            }
        }

        config
    }

    /// Get profile description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Development => {
                "Optimized for development with fast compilation and debugging support"
            }
            Self::Balanced => "Balanced configuration for general-purpose use",
            Self::HighPerformance => "High-performance configuration with advanced optimizations",
            Self::MaxThroughput => {
                "Maximum throughput configuration with all optimizations enabled"
            }
            Self::MemoryEfficient => {
                "Memory-efficient configuration for resource-constrained environments"
            }
            Self::LowLatency => "Low-latency configuration for real-time applications",
            Self::BatchProcessing => "Optimized for large-scale batch processing",
            Self::RealTime => "Real-time configuration with deterministic performance",
            Self::EdgeComputing => "Optimized for edge computing and IoT devices",
            Self::Custom => "User-defined custom configuration",
        }
    }
}

impl Display for PerformanceProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Development => "Development",
            Self::Balanced => "Balanced",
            Self::HighPerformance => "High Performance",
            Self::MaxThroughput => "Max Throughput",
            Self::MemoryEfficient => "Memory Efficient",
            Self::LowLatency => "Low Latency",
            Self::BatchProcessing => "Batch Processing",
            Self::RealTime => "Real Time",
            Self::EdgeComputing => "Edge Computing",
            Self::Custom => "Custom",
        };
        write!(f, "{}", name)
    }
}
