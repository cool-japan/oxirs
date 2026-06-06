//! Performance profile configuration for OxiRS Core.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub profile: PerformanceProfile,
    pub custom_settings: HashMap<String, PerformanceValue>,
    pub enable_auto_tuning: bool,
    pub auto_tuning_sensitivity: f64,
    pub monitoring_interval: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceProfile {
    Development,
    Balanced,
    HighPerformance,
    MaxThroughput,
    MemoryEfficient,
    LowLatency,
    BatchProcessing,
    RealTime,
    EdgeComputing,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PerformanceValue {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Duration(u64),
}

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

impl PerformanceProfile {
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
                config.insert(
                    "max_latency_ms".to_string(),
                    PerformanceValue::Duration(10),
                );
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
            Self::Custom => {}
        }

        config
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Development => {
                "Optimized for development with fast compilation and debugging support"
            }
            Self::Balanced => "Balanced configuration for general-purpose use",
            Self::HighPerformance => "High-performance configuration with advanced optimizations",
            Self::MaxThroughput => "Maximum throughput configuration with all optimizations enabled",
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
