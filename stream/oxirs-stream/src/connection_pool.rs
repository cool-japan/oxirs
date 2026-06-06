//! Connection Pool — facade module
//!
//! This module re-exports the full public API from the split sub-modules.

pub use crate::circuit_breaker::CircuitBreakerConfig;
pub use crate::connection_pool_manager::{ConnectionPool, PooledConnectionHandle};
pub use crate::connection_pool_types::{
    ConnectionFactory, DetailedPoolMetrics, LoadBalancingStrategy, PoolConfig, PoolStats,
    PoolStatus, PooledConnection,
};
