//! # Online Index Compaction System
//!
//! This module provides production-grade online compaction for vector indices,
//! enabling efficient memory reclamation and defragmentation without downtime.
//!
//! ## Features
//!
//! - **Online Compaction**: No downtime during compaction operations
//! - **Incremental Processing**: Process in small batches to minimize impact
//! - **Multiple Strategies**: Time-based, size-based, and adaptive compaction
//! - **Background Operation**: Runs in background thread with low priority
//! - **Metrics & Monitoring**: Track fragmentation, reclaimed space, and performance
//! - **Graceful Degradation**: Continues serving queries during compaction
//!
//! ## Compaction Strategies
//!
//! - **Periodic**: Compact at regular time intervals
//! - **Threshold-Based**: Compact when fragmentation exceeds threshold
//! - **Adaptive**: Automatically adjust based on workload patterns
//! - **Manual**: Trigger compaction on demand
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_vec::compaction::{CompactionManager, CompactionConfig, CompactionStrategy};
//!
//! let config = CompactionConfig {
//!     strategy: CompactionStrategy::Adaptive,
//!     fragmentation_threshold: 0.3,
//!     compaction_interval: Duration::from_secs(3600),
//!     batch_size: 1000,
//!     ..Default::default()
//! };
//!
//! let mut manager = CompactionManager::new(config)?;
//!
//! // Start background compaction
//! manager.start_background_compaction()?;
//!
//! // Or trigger manual compaction
//! manager.compact_now()?;
//!
//! // Get compaction metrics
//! let stats = manager.get_statistics();
//! println!("Fragmentation: {:.2}%", stats.fragmentation_ratio * 100.0);
//! ```

#![allow(dead_code)]

pub mod config;
pub mod manager;
pub mod metrics;
pub mod strategies;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export main types
pub use config::CompactionConfig;
pub use manager::CompactionManager;
pub use metrics::CompactionMetrics;
pub use strategies::CompactionStrategy;
pub use types::{CompactionResult, CompactionState, CompactionStatistics};
