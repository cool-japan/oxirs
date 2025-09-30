//! # NATS Streaming Backend - Modular Architecture
//!
//! NATS support for streaming RDF data with advanced features.
//!
//! This module provides enterprise-grade NATS integration with:
//! - JetStream for persistence and delivery guarantees
//! - Advanced connection pooling and health monitoring
//! - Circuit breaker patterns and compression
//! - Real-time performance analytics
//!
//! ## Architecture
//!
//! The NATS backend is organized into specialized modules:
//! - `connection_pool`: Advanced connection management
//! - `health_monitor`: Predictive health monitoring
//! - `circuit_breaker`: ML-based failure detection
//! - `compression`: Adaptive compression algorithms
//! - `config`: Configuration management
//! - `producer`/`consumer`: Message handling

use crate::backend::{StreamBackend as StreamBackendTrait, StreamBackendConfig};
use crate::error::{StreamError, StreamResult};
use crate::types::{Offset, PartitionId, StreamPosition, TopicName};
use crate::{EventMetadata, StreamEvent};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

// Import modular components
pub mod circuit_breaker;
pub mod compression;
pub mod config;
pub mod connection_pool;
pub mod health_monitor;

// Re-export key types for backward compatibility
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState};
pub use compression::{CompressionAlgorithm, CompressionConfig, CompressionManager, CompressionStats};
pub use config::{NatsConfig, NatsConsumerConfig, NatsProducerConfig, NatsStorageType, NatsRetentionPolicy};
pub use connection_pool::{ConnectionPool, ConnectionWrapper, HealthStatus as PoolHealthStatus};
pub use health_monitor::{HealthMonitor, HealthMonitorConfig, HealthStatus, HealthMetrics};

#[cfg(feature = "nats")]
use async_nats::{
    jetstream::{self, consumer::PullConsumer, stream::Stream},
    Client, ConnectOptions,
};

/// NATS streaming backend implementation with enterprise features
#[derive(Debug)]
pub struct NatsBackend {
    config: NatsConfig,
    connection_pool: Arc<RwLock<Option<ConnectionPool>>>,
    health_monitor: Arc<HealthMonitor>,
    circuit_breaker: Arc<RwLock<CircuitBreaker>>,
    compression_manager: Arc<CompressionManager>,
    #[cfg(feature = "nats")]
    jetstream: Arc<RwLock<Option<jetstream::Context>>>,
}

impl NatsBackend {
    /// Create a new NATS backend with configuration
    pub fn new(config: NatsConfig) -> Self {
        let health_monitor = Arc::new(HealthMonitor::new(HealthMonitorConfig::default()));
        let circuit_breaker = Arc::new(RwLock::new(CircuitBreaker::new(CircuitBreakerConfig::default())));
        let compression_manager = Arc::new(CompressionManager::new(CompressionConfig::default()));

        Self {
            config,
            connection_pool: Arc::new(RwLock::new(None)),
            health_monitor,
            circuit_breaker,
            compression_manager,
            #[cfg(feature = "nats")]
            jetstream: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize the NATS connection with advanced features
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing NATS backend with enterprise features");

        // Initialize connection pool
        let pool = ConnectionPool::new(&self.config, self.health_monitor.clone()).await?;
        *self.connection_pool.write().await = Some(pool);

        // Start health monitoring
        self.health_monitor.start_monitoring().await?;

        #[cfg(feature = "nats")]
        {
            // Establish JetStream context
            let pool_guard = self.connection_pool.read().await;
            if let Some(ref pool) = *pool_guard {
                let client = pool.get_connection().await?;
                let jetstream = jetstream::new(client);
                *self.jetstream.write().await = Some(jetstream);
            }
        }

        info!("NATS backend initialization complete");
        Ok(())
    }

    /// Get connection pool statistics
    pub async fn get_pool_stats(&self) -> Option<HashMap<String, f64>> {
        let pool_guard = self.connection_pool.read().await;
        if let Some(ref pool) = *pool_guard {
            Some(pool.get_statistics().await)
        } else {
            None
        }
    }

    /// Get health status
    pub async fn get_health_status(&self) -> HealthStatus {
        self.health_monitor.get_current_status().await
    }

    /// Get circuit breaker status
    pub async fn get_circuit_breaker_status(&self) -> CircuitState {
        let cb_guard = self.circuit_breaker.read().await;
        cb_guard.get_state().await
    }

    /// Get compression statistics
    pub async fn get_compression_stats(&self) -> CompressionStats {
        self.compression_manager.get_statistics().await
    }

    /// Compress data using the configured algorithm
    pub async fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.compression_manager.compress(data).await
    }

    /// Decompress data
    pub async fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.compression_manager.decompress(data).await
    }

    /// Shutdown the backend gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down NATS backend");

        // Stop health monitoring
        self.health_monitor.stop_monitoring().await;

        // Close connection pool
        let mut pool_guard = self.connection_pool.write().await;
        if let Some(pool) = pool_guard.take() {
            pool.shutdown().await?;
        }

        #[cfg(feature = "nats")]
        {
            *self.jetstream.write().await = None;
        }

        info!("NATS backend shutdown complete");
        Ok(())
    }
}

#[async_trait]
impl StreamBackendTrait for NatsBackend {
    type Config = NatsConfig;
    type Producer = NatsProducer;
    type Consumer = NatsConsumer;

    async fn new_with_config(config: Self::Config) -> StreamResult<Self> {
        let backend = Self::new(config);
        backend.initialize().await.map_err(|e| {
            StreamError::BackendError(format!("Failed to initialize NATS backend: {}", e))
        })?;
        Ok(backend)
    }

    async fn create_producer(&self, config: &StreamBackendConfig) -> StreamResult<Self::Producer> {
        let pool_guard = self.connection_pool.read().await;
        let pool = pool_guard.as_ref().ok_or_else(|| {
            StreamError::BackendError("Connection pool not initialized".to_string())
        })?;

        let producer_config = NatsProducerConfig::from_backend_config(config);
        NatsProducer::new(producer_config, pool.clone(), self.compression_manager.clone()).await
    }

    async fn create_consumer(&self, config: &StreamBackendConfig) -> StreamResult<Self::Consumer> {
        let pool_guard = self.connection_pool.read().await;
        let pool = pool_guard.as_ref().ok_or_else(|| {
            StreamError::BackendError("Connection pool not initialized".to_string())
        })?;

        let consumer_config = NatsConsumerConfig::from_backend_config(config);
        NatsConsumer::new(consumer_config, pool.clone(), self.compression_manager.clone()).await
    }

    async fn health_check(&self) -> StreamResult<bool> {
        let health_status = self.get_health_status().await;
        Ok(matches!(health_status, HealthStatus::Healthy))
    }

    async fn get_stats(&self) -> StreamResult<HashMap<String, serde_json::Value>> {
        let mut stats = HashMap::new();

        // Pool statistics
        if let Some(pool_stats) = self.get_pool_stats().await {
            stats.insert("connection_pool".to_string(), serde_json::to_value(pool_stats)?);
        }

        // Health status
        let health_status = self.get_health_status().await;
        stats.insert("health_status".to_string(), serde_json::to_value(health_status)?);

        // Circuit breaker status
        let cb_status = self.get_circuit_breaker_status().await;
        stats.insert("circuit_breaker".to_string(), serde_json::to_value(cb_status)?);

        // Compression statistics
        let compression_stats = self.get_compression_stats().await;
        stats.insert("compression".to_string(), serde_json::to_value(compression_stats)?);

        Ok(stats)
    }
}

/// NATS producer implementation
#[derive(Debug)]
pub struct NatsProducer {
    config: NatsProducerConfig,
    connection_pool: ConnectionPool,
    compression_manager: Arc<CompressionManager>,
}

impl NatsProducer {
    pub async fn new(
        config: NatsProducerConfig,
        connection_pool: ConnectionPool,
        compression_manager: Arc<CompressionManager>,
    ) -> StreamResult<Self> {
        Ok(Self {
            config,
            connection_pool,
            compression_manager,
        })
    }

    pub async fn send(&self, event: &StreamEvent) -> StreamResult<()> {
        // Implementation would handle actual message sending
        // This is a placeholder for the modular architecture
        debug!("Sending event via NATS producer: {:?}", event.id);
        Ok(())
    }
}

/// NATS consumer implementation
#[derive(Debug)]
pub struct NatsConsumer {
    config: NatsConsumerConfig,
    connection_pool: ConnectionPool,
    compression_manager: Arc<CompressionManager>,
}

impl NatsConsumer {
    pub async fn new(
        config: NatsConsumerConfig,
        connection_pool: ConnectionPool,
        compression_manager: Arc<CompressionManager>,
    ) -> StreamResult<Self> {
        Ok(Self {
            config,
            connection_pool,
            compression_manager,
        })
    }

    pub async fn receive(&self) -> StreamResult<Option<StreamEvent>> {
        // Implementation would handle actual message receiving
        // This is a placeholder for the modular architecture
        debug!("Receiving event via NATS consumer");
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_nats_backend_creation() {
        let config = NatsConfig::default();
        let backend = NatsBackend::new(config);
        
        let health_status = backend.get_health_status().await;
        // Should start in Unknown state until monitoring begins
        assert!(matches!(health_status, HealthStatus::Unknown));
    }

    #[tokio::test]
    async fn test_compression_roundtrip() {
        let config = NatsConfig::default();
        let backend = NatsBackend::new(config);
        
        let test_data = b"Hello, NATS compression!";
        let compressed = backend.compress_data(test_data).await.unwrap();
        let decompressed = backend.decompress_data(&compressed).await.unwrap();
        
        assert_eq!(test_data, decompressed.as_slice());
    }
}