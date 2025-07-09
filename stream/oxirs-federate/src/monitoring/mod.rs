//! Federation Monitoring and Metrics
//!
//! This module provides comprehensive monitoring and metrics collection for federated
//! query processing, including performance tracking, error monitoring, and observability.

pub mod config;
pub mod metrics;
pub mod monitor;
pub mod resilience;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export main types to maintain API compatibility
pub use config::*;
pub use metrics::*;
pub use monitor::*;
pub use resilience::*;
pub use types::*;