//! # Hot/Warm/Cold Tiering System for Vector Indices
//!
//! This module provides a production-grade tiering system that automatically manages
//! vector indices across different storage tiers based on access patterns, size, and
//! performance requirements.
//!
//! ## Tiering Strategy
//!
//! - **Hot Tier**: In-memory, frequently accessed indices (sub-millisecond latency)
//! - **Warm Tier**: Memory-mapped or SSD-backed indices (millisecond latency)
//! - **Cold Tier**: Compressed, disk-based indices (second latency, loaded on demand)
//!
//! ## Features
//!
//! - Automatic tier promotion/demotion based on access patterns
//! - Configurable policies (LRU, LFU, cost-based, adaptive)
//! - Integration with monitoring and analytics
//! - Support for gradual tier transitions
//! - Predictive tier management using ML
//! - Multi-tenant resource allocation
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_vec::tiering::{TieringManager, TieringConfig, TieringPolicy};
//!
//! let config = TieringConfig {
//!     hot_tier_capacity_gb: 16.0,
//!     warm_tier_capacity_gb: 128.0,
//!     cold_tier_capacity_gb: 1024.0,
//!     policy: TieringPolicy::Adaptive,
//!     ..Default::default()
//! };
//!
//! let mut manager = TieringManager::new(config)?;
//!
//! // Add an index
//! manager.register_index("embeddings_v1", index)?;
//!
//! // Access triggers automatic tier management
//! let results = manager.query_index("embeddings_v1", query_vector, k)?;
//!
//! // Check tier statistics
//! let stats = manager.get_tier_statistics();
//! println!("Hot tier: {:.2}% utilized", stats.hot_tier_utilization * 100.0);
//! ```

#![allow(dead_code)]

pub mod access_tracker;
pub mod config;
pub mod manager;
pub mod metrics;
pub mod policies;
pub mod storage_backends;
pub mod tier_optimizer;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export main types
pub use config::TieringConfig;
pub use manager::TieringManager;
pub use metrics::TierMetrics;
pub use policies::{TierTransitionReason, TieringPolicy};
pub use types::{IndexMetadata, StorageTier, TierStatistics};
