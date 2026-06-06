//! Advanced optimization configuration: SIMD, zero-copy, prefetching, caching.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub simd: SimdConfig,
    pub zero_copy: ZeroCopyConfig,
    pub prefetching: PrefetchingConfig,
    pub caching: CachingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdConfig {
    pub enabled: bool,
    pub instruction_set: SimdInstructionSet,
    pub fallback_to_scalar: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimdInstructionSet {
    None,
    SSE2,
    AVX,
    AVX2,
    AVX512,
    NEON,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroCopyConfig {
    pub enabled: bool,
    pub arena_size: usize,
    pub reference_counting: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchingConfig {
    pub enabled: bool,
    pub distance: usize,
    pub strategy: PrefetchStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    Sequential,
    Random,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    pub caches: HashMap<String, CacheConfig>,
    pub global: GlobalCacheConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub name: String,
    pub cache_type: CacheType,
    pub max_size: usize,
    pub ttl: Duration,
    pub eviction_policy: EvictionPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheType {
    LRU,
    LFU,
    FIFO,
    Random,
    Adaptive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    TTL,
    Size,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCacheConfig {
    pub memory_limit: usize,
    pub enable_statistics: bool,
    pub enable_warming: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            simd: SimdConfig::default(),
            zero_copy: ZeroCopyConfig::default(),
            prefetching: PrefetchingConfig::default(),
            caching: CachingConfig::default(),
        }
    }
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            instruction_set: SimdInstructionSet::Auto,
            fallback_to_scalar: true,
        }
    }
}

impl Default for ZeroCopyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            arena_size: 1024 * 1024,
            reference_counting: true,
        }
    }
}

impl Default for PrefetchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            distance: 64,
            strategy: PrefetchStrategy::Adaptive,
        }
    }
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            caches: HashMap::new(),
            global: GlobalCacheConfig::default(),
        }
    }
}

impl Default for GlobalCacheConfig {
    fn default() -> Self {
        Self {
            memory_limit: 256 * 1024 * 1024,
            enable_statistics: true,
            enable_warming: true,
        }
    }
}
