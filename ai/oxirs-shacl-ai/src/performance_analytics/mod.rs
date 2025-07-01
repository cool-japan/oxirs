//! Performance Analytics with Real-time Monitoring and Optimization
//!
//! This module implements comprehensive performance analytics capabilities including
//! real-time monitoring, performance optimization, and intelligent performance insights.

pub mod alerts;
pub mod config;
pub mod dashboard;
pub mod engine;
pub mod metrics;
pub mod monitoring;
pub mod optimization;
pub mod types;

// Re-export main types and functions
pub use alerts::*;
pub use config::*;
pub use dashboard::*;
pub use engine::*;
pub use metrics::*;
pub use monitoring::*;
pub use optimization::*;
pub use types::*;