//! Memory management configuration for OxiRS Core.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub arena: ArenaConfig,
    pub gc: GcConfig,
    pub pressure_thresholds: MemoryPressureConfig,
    pub enable_tracking: bool,
    pub pools: HashMap<String, MemoryPoolConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaConfig {
    pub initial_size: usize,
    pub max_size: usize,
    pub growth_factor: f64,
    pub enable_compaction: bool,
    pub compaction_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcConfig {
    pub strategy: GcStrategy,
    pub trigger_threshold: f64,
    pub max_pause_time: Duration,
    pub enable_concurrent: bool,
    pub worker_threads: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GcStrategy {
    None,
    ReferenceCounting,
    MarkAndSweep,
    Generational,
    Incremental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPressureConfig {
    pub low_threshold: f64,
    pub medium_threshold: f64,
    pub high_threshold: f64,
    pub critical_threshold: f64,
    pub pressure_actions: HashMap<String, Vec<MemoryPressureAction>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryPressureAction {
    ForceGc,
    CompactArenas,
    ClearCaches,
    ReduceBuffers,
    SuspendBackgroundTasks,
    SendAlert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    pub name: String,
    pub initial_size: usize,
    pub max_size: usize,
    pub object_size: usize,
    pub enable_preallocation: bool,
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
            initial_size: 1024 * 1024,
            max_size: 64 * 1024 * 1024,
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
