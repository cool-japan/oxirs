//! Configuration type definitions for OxiRS Core — thin facade module.
//!
//! All types are split across three sibling modules:
//! - `config_types_core`     — performance, memory, interning
//! - `config_types_storage`  — indexing, parsing, serialization, caching
//! - `config_types_network`  — concurrency, monitoring, security, optimization,
//!   environment / config-management infrastructure
//!
//! This module composes them into `OxirsConfig` and `ConfigurationManager`
//! and re-exports all public items.

use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

pub use crate::config_types_core::*;
pub use crate::config_types_network::*;
pub use crate::config_types_storage::*;

// ─────────────────────────────────────────────────────────────
// ConfigurationManager
// ─────────────────────────────────────────────────────────────

/// Central configuration manager for OxiRS Core
pub struct ConfigurationManager {
    #[allow(dead_code)]
    pub(crate) config: Arc<RwLock<OxirsConfig>>,
    #[allow(dead_code)]
    pub(crate) environment: Environment,
    #[allow(dead_code)]
    pub(crate) config_sources: Vec<ConfigSource>,
    #[allow(dead_code)]
    pub(crate) watchers: Vec<ConfigWatcher>,
}

/// Configuration watchers
pub struct ConfigWatcher {
    #[allow(dead_code)]
    pub(crate) source: ConfigSource,
    #[allow(dead_code)]
    pub(crate) callback: Box<dyn Fn(&OxirsConfig) + Send + Sync>,
}

// ─────────────────────────────────────────────────────────────
// OxirsConfig — top-level composite
// ─────────────────────────────────────────────────────────────

/// Main configuration structure for OxiRS Core
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OxirsConfig {
    /// Performance profile configuration
    pub performance: PerformanceConfig,
    /// Memory management configuration
    pub memory: MemoryConfig,
    /// String interning configuration
    pub interning: InterningConfig,
    /// Indexing strategy configuration
    pub indexing: IndexingConfig,
    /// Parser configuration
    pub parsing: ParsingConfig,
    /// Serializer configuration
    pub serialization: SerializationConfig,
    /// Concurrency configuration
    pub concurrency: ConcurrencyConfig,
    /// Monitoring and observability configuration
    pub monitoring: MonitoringConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Advanced optimization configuration
    pub optimization: OptimizationConfig,
}
