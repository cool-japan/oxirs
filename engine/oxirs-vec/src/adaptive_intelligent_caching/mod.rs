//! Adaptive Intelligent Caching System for OxiRS Vector Search
//!
//! This module provides advanced caching strategies that adapt based on query patterns,
//! vector characteristics, and performance metrics. It implements machine learning-driven
//! cache optimization and predictive prefetching.

#![allow(dead_code)]

pub mod access_tracker;
pub mod cache;
pub mod config;
pub mod eviction;
pub mod metrics;
pub mod ml_models;
pub mod optimizer;
pub mod pattern_analyzer;
pub mod prefetcher;
pub mod storage;
pub mod tier;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export main types for convenience
pub use cache::AdaptiveIntelligentCache;
pub use config::CacheConfiguration;
pub use metrics::CachePerformanceMetrics;
pub use ml_models::MLModels;
pub use optimizer::CacheOptimizer;
pub use pattern_analyzer::AccessPatternAnalyzer;
pub use prefetcher::PredictivePrefetcher;
pub use tier::CacheTier;
pub use types::{CacheKey, CacheStatistics, CacheValue, ExportFormat};
