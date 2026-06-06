//! String interning configuration for OxiRS Core.

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterningConfig {
    pub global: GlobalInternerConfig,
    pub scoped: ScopedInternerConfig,
    pub cleanup: InternerCleanupConfig,
    pub enable_statistics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalInternerConfig {
    pub initial_capacity: usize,
    pub load_factor: f64,
    pub enable_weak_references: bool,
    pub lru_cache_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedInternerConfig {
    pub default_capacity: usize,
    pub max_scopes: usize,
    pub scope_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternerCleanupConfig {
    pub cleanup_interval: Duration,
    pub cleanup_threshold: f64,
    pub enable_automatic: bool,
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
