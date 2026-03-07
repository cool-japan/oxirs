//! Connection pool subsystem for OxiRS Fuseki
//!
//! Provides generic adaptive connection pooling with per-dataset registries
//! and backpressure control.

pub mod adaptive_connection_pool;
pub mod adaptive_pool;
pub mod backpressure;
pub mod connection_pool;

pub use adaptive_connection_pool::{
    AdaptiveConnectionPool, AdaptiveConnectionPoolConfig, AdaptivePoolGuard, AdaptivePoolMetrics,
};
pub use adaptive_pool::{
    AdaptivePool, DatasetPoolRegistry, PoolConfig, PoolStats, PooledConnection,
};
pub use backpressure::{
    BackpressureConfig, BackpressureController, BackpressureDecision, BackpressureStats,
};
pub use connection_pool::ConnectionPool;
