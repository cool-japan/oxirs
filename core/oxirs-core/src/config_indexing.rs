//! Indexing configuration for OxiRS Core.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingConfig {
    pub default_strategy: IndexingStrategy,
    pub strategy_configs: HashMap<String, IndexStrategyConfig>,
    pub adaptive: AdaptiveIndexingConfig,
    pub persistence: IndexPersistenceConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexingStrategy {
    None,
    Single,
    Dual,
    Triple,
    AdaptiveMulti,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStrategyConfig {
    pub name: String,
    pub index_types: Vec<IndexType>,
    pub bloom_filter: BloomFilterConfig,
    pub compaction: IndexCompactionConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    SPO,
    POS,
    OSP,
    SOP,
    PSO,
    OPS,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilterConfig {
    pub enabled: bool,
    pub expected_items: usize,
    pub false_positive_rate: f64,
    pub hash_functions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexCompactionConfig {
    pub enabled: bool,
    pub threshold: f64,
    pub interval: Duration,
    pub concurrent: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveIndexingConfig {
    pub enabled: bool,
    pub analysis_window: Duration,
    pub min_query_frequency: f64,
    pub effectiveness_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexPersistenceConfig {
    pub enabled: bool,
    pub directory: PathBuf,
    pub sync_interval: Duration,
    pub compression: bool,
}

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
