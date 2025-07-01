//! Stream processing components for the real-time embedding pipeline

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::real_time_embedding_pipeline::{
    traits::{ContentItem, HealthStatus},
    types::{StreamState, StreamStatus},
};

/// Configuration for stream processors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Stream identifier
    pub stream_id: String,
    /// Buffer size for the stream
    pub buffer_size: usize,
    /// Processing timeout
    pub timeout: Duration,
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Enable compression
    pub enable_compression: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            stream_id: "default".to_string(),
            buffer_size: 1000,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            enable_compression: false,
        }
    }
}

/// Stream processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamProcessorConfig {
    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,
    /// Stream timeout settings
    pub timeout_config: TimeoutConfig,
    /// Buffer management settings
    pub buffer_config: BufferConfig,
    /// Error handling settings
    pub error_config: ErrorConfig,
}

impl Default for StreamProcessorConfig {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 10,
            timeout_config: TimeoutConfig::default(),
            buffer_config: BufferConfig::default(),
            error_config: ErrorConfig::default(),
        }
    }
}

/// Timeout configuration for stream processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Read timeout
    pub read_timeout: Duration,
    /// Write timeout
    pub write_timeout: Duration,
    /// Idle timeout
    pub idle_timeout: Duration,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connection_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(30),
            write_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
        }
    }
}

/// Buffer configuration for stream processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Initial buffer size
    pub initial_size: usize,
    /// Maximum buffer size
    pub max_size: usize,
    /// Growth factor when resizing
    pub growth_factor: f64,
    /// Enable adaptive sizing
    pub adaptive_sizing: bool,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            initial_size: 1000,
            max_size: 100000,
            growth_factor: 1.5,
            adaptive_sizing: true,
        }
    }
}

/// Error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorConfig {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Exponential backoff factor
    pub backoff_factor: f64,
    /// Maximum retry delay
    pub max_retry_delay: Duration,
    /// Enable circuit breaker
    pub enable_circuit_breaker: bool,
}

impl Default for ErrorConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            backoff_factor: 2.0,
            max_retry_delay: Duration::from_secs(30),
            enable_circuit_breaker: true,
        }
    }
}

/// Stream processor for handling content streams
pub struct StreamProcessor {
    /// Stream identifier
    stream_id: String,
    /// Stream configuration
    config: StreamConfig,
    /// Running state
    is_running: AtomicBool,
    /// Current stream state
    state: StreamState,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(stream_id: String, config: StreamConfig) -> Result<Self> {
        let state = StreamState {
            stream_id: stream_id.clone(),
            offset: 0,
            last_processed: std::time::SystemTime::now(),
            status: StreamStatus::Initializing,
            error_count: 0,
            last_error: None,
        };

        Ok(Self {
            stream_id,
            config,
            is_running: AtomicBool::new(false),
            state,
        })
    }

    /// Start the stream processor
    pub async fn start(&self) -> Result<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Err(anyhow::anyhow!("Stream processor is already running"));
        }

        self.is_running.store(true, Ordering::Release);

        // Initialize stream processing
        self.initialize_stream().await?;

        Ok(())
    }

    /// Stop the stream processor
    pub async fn stop(&self) -> Result<()> {
        self.is_running.store(false, Ordering::Release);

        // Cleanup resources
        self.cleanup_stream().await?;

        Ok(())
    }

    /// Process a content item
    pub async fn process_item(&self, item: ContentItem) -> Result<()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(anyhow::anyhow!("Stream processor is not running"));
        }

        // Process the content item
        self.handle_content_item(item).await?;

        Ok(())
    }

    /// Get stream state
    pub fn get_state(&self) -> &StreamState {
        &self.state
    }

    /// Get stream configuration
    pub fn get_config(&self) -> &StreamConfig {
        &self.config
    }

    /// Check stream health
    pub async fn health_check(&self) -> Result<HealthStatus> {
        if !self.is_running.load(Ordering::Acquire) {
            return Ok(HealthStatus::Unhealthy {
                message: "Stream processor is not running".to_string(),
            });
        }

        // Check various health metrics
        if self.state.error_count > 10 {
            return Ok(HealthStatus::Warning {
                message: format!("High error count: {}", self.state.error_count),
            });
        }

        Ok(HealthStatus::Healthy)
    }

    /// Check if the processor is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    // Private helper methods

    async fn initialize_stream(&self) -> Result<()> {
        // Initialize stream resources
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    async fn cleanup_stream(&self) -> Result<()> {
        // Cleanup stream resources
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    async fn handle_content_item(&self, _item: ContentItem) -> Result<()> {
        // Process content item
        // This would typically involve:
        // 1. Validation
        // 2. Embedding generation
        // 3. Index updates
        // 4. Quality checks
        tokio::time::sleep(Duration::from_millis(1)).await;
        Ok(())
    }
}

/// Stream multiplexer for handling multiple streams
pub struct StreamMultiplexer {
    /// Active stream processors
    processors: std::sync::RwLock<std::collections::HashMap<String, StreamProcessor>>,
    /// Configuration
    config: StreamProcessorConfig,
}

impl StreamMultiplexer {
    /// Create a new stream multiplexer
    pub fn new(config: StreamProcessorConfig) -> Self {
        Self {
            processors: std::sync::RwLock::new(std::collections::HashMap::new()),
            config,
        }
    }

    /// Add a stream processor
    pub fn add_processor(&self, processor: StreamProcessor) -> Result<()> {
        let mut processors = self
            .processors
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire processors lock"))?;

        let stream_id = processor.stream_id.clone();
        processors.insert(stream_id, processor);
        Ok(())
    }

    /// Remove a stream processor
    pub async fn remove_processor(&self, stream_id: &str) -> Result<()> {
        let mut processors = self
            .processors
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire processors lock"))?;

        if let Some(processor) = processors.remove(stream_id) {
            processor.stop().await?;
        }

        Ok(())
    }

    /// Get processor count
    pub fn processor_count(&self) -> usize {
        self.processors.read().map(|p| p.len()).unwrap_or(0)
    }

    /// Check multiplexer health
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let processors = self
            .processors
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire processors lock"))?;

        let unhealthy_count = {
            let mut count = 0;
            for processor in processors.values() {
                match processor.health_check().await? {
                    HealthStatus::Unhealthy { .. } => count += 1,
                    _ => {}
                }
            }
            count
        };

        if unhealthy_count == 0 {
            Ok(HealthStatus::Healthy)
        } else if unhealthy_count < processors.len() {
            Ok(HealthStatus::Warning {
                message: format!("{} processors are unhealthy", unhealthy_count),
            })
        } else {
            Ok(HealthStatus::Unhealthy {
                message: "All processors are unhealthy".to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.buffer_size, 1000);
        assert_eq!(config.max_retries, 3);
    }

    #[tokio::test]
    async fn test_stream_processor_creation() {
        let config = StreamConfig::default();
        let processor = StreamProcessor::new("test_stream".to_string(), config);
        assert!(processor.is_ok());
    }

    #[tokio::test]
    async fn test_stream_processor_start_stop() {
        let config = StreamConfig::default();
        let processor = StreamProcessor::new("test_stream".to_string(), config).unwrap();

        assert!(!processor.is_running());

        let start_result = processor.start().await;
        assert!(start_result.is_ok());
        assert!(processor.is_running());

        let stop_result = processor.stop().await;
        assert!(stop_result.is_ok());
    }

    #[test]
    fn test_stream_multiplexer() {
        let config = StreamProcessorConfig::default();
        let multiplexer = StreamMultiplexer::new(config);
        assert_eq!(multiplexer.processor_count(), 0);
    }
}
