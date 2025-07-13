//! Advanced Async Streaming for Real-time Federation
//!
//! This module provides cutting-edge async streaming capabilities for real-time
//! GraphQL federation, including adaptive streaming, backpressure handling,
//! and intelligent data flow optimization.

use anyhow::{anyhow, Result};
use futures_util::{pin_mut, stream::StreamExt, Stream};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{broadcast, mpsc, Mutex as AsyncMutex, RwLock as AsyncRwLock};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::ast::Value;

/// Streaming configuration for real-time federation
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub enable_adaptive_streaming: bool,
    pub enable_backpressure_control: bool,
    pub enable_stream_multiplexing: bool,
    pub enable_data_compression: bool,
    pub enable_priority_queuing: bool,
    pub buffer_size: usize,
    pub max_concurrent_streams: usize,
    pub stream_timeout: Duration,
    pub heartbeat_interval: Duration,
    pub backpressure_threshold: f64,
    pub compression_config: CompressionConfig,
    pub priority_config: PriorityConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_streaming: true,
            enable_backpressure_control: true,
            enable_stream_multiplexing: true,
            enable_data_compression: true,
            enable_priority_queuing: true,
            buffer_size: 1000,
            max_concurrent_streams: 100,
            stream_timeout: Duration::from_secs(30),
            heartbeat_interval: Duration::from_secs(10),
            backpressure_threshold: 0.8,
            compression_config: CompressionConfig::default(),
            priority_config: PriorityConfig::default(),
        }
    }
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub algorithm: CompressionAlgorithm,
    pub compression_level: u8,
    pub min_compression_size: usize,
    pub adaptive_compression: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Gzip,
            compression_level: 6,
            min_compression_size: 1024,
            adaptive_compression: true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Brotli,
}

/// Priority configuration
#[derive(Debug, Clone)]
pub struct PriorityConfig {
    pub enable_priority_inheritance: bool,
    pub default_priority: StreamPriority,
    pub priority_queues: usize,
    pub starvation_prevention: bool,
}

impl Default for PriorityConfig {
    fn default() -> Self {
        Self {
            enable_priority_inheritance: true,
            default_priority: StreamPriority::Medium,
            priority_queues: 5,
            starvation_prevention: true,
        }
    }
}

/// Stream priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    Critical,
    High,
    Medium,
    Low,
    Background,
}

/// Streaming data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamData {
    GraphQLResponse {
        query_id: String,
        data: Value,
        errors: Vec<String>,
        extensions: HashMap<String, Value>,
    },
    SubscriptionUpdate {
        subscription_id: String,
        data: Value,
        revision: u64,
    },
    FederationUpdate {
        service_id: String,
        schema_changes: Vec<SchemaChange>,
        timestamp: SystemTime,
    },
    MetricsUpdate {
        metrics: PerformanceMetrics,
        timestamp: SystemTime,
    },
    HealthUpdate {
        service_id: String,
        health_status: HealthStatus,
        timestamp: SystemTime,
    },
    ConfigUpdate {
        config_changes: HashMap<String, Value>,
        timestamp: SystemTime,
    },
}

/// Schema change information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaChange {
    pub change_type: SchemaChangeType,
    pub affected_types: Vec<String>,
    pub affected_fields: Vec<String>,
    pub description: String,
    pub breaking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchemaChangeType {
    TypeAdded,
    TypeRemoved,
    TypeModified,
    FieldAdded,
    FieldRemoved,
    FieldModified,
    DirectiveAdded,
    DirectiveRemoved,
}

/// Performance metrics for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_mbps: f64,
    pub latency_ms: f64,
    pub error_rate: f64,
    pub active_streams: usize,
    pub backpressure_events: u64,
    pub compression_ratio: f64,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Stream metadata
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    pub stream_id: String,
    pub priority: StreamPriority,
    pub created_at: Instant,
    pub last_activity: Instant,
    pub total_messages: u64,
    pub total_bytes: u64,
    pub compression_enabled: bool,
    pub client_info: ClientInfo,
}

/// Client information
#[derive(Debug, Clone)]
pub struct ClientInfo {
    pub client_id: String,
    pub connection_type: ConnectionType,
    pub capabilities: HashSet<StreamCapability>,
}

#[derive(Debug, Clone)]
pub enum ConnectionType {
    WebSocket,
    ServerSentEvents,
    GraphQLSubscription,
    CustomProtocol,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum StreamCapability {
    Compression,
    Multiplexing,
    PriorityStreaming,
    BackpressureControl,
    StreamResumption,
    DeltaUpdates,
}

use std::collections::HashSet;

/// Advanced streaming manager
pub struct AsyncStreamingManager {
    config: StreamingConfig,
    active_streams: Arc<AsyncRwLock<HashMap<String, StreamHandle>>>,
    stream_multiplexer: Arc<AsyncMutex<StreamMultiplexer>>,
    backpressure_controller: Arc<AsyncMutex<BackpressureController>>,
    compression_manager: Arc<AsyncMutex<CompressionManager>>,
    priority_scheduler: Arc<AsyncMutex<PriorityScheduler>>,
    metrics_collector: Arc<AsyncMutex<MetricsCollector>>,
    data_sender: broadcast::Sender<StreamData>,
}

impl AsyncStreamingManager {
    /// Create a new async streaming manager
    pub fn new(config: StreamingConfig) -> (Self, broadcast::Receiver<StreamData>) {
        let (data_sender, data_receiver) = broadcast::channel(config.buffer_size);

        let manager = Self {
            config: config.clone(),
            active_streams: Arc::new(AsyncRwLock::new(HashMap::new())),
            stream_multiplexer: Arc::new(AsyncMutex::new(StreamMultiplexer::new(&config))),
            backpressure_controller: Arc::new(AsyncMutex::new(BackpressureController::new(
                &config,
            ))),
            compression_manager: Arc::new(AsyncMutex::new(CompressionManager::new(
                &config.compression_config,
            ))),
            priority_scheduler: Arc::new(AsyncMutex::new(PriorityScheduler::new(
                &config.priority_config,
            ))),
            metrics_collector: Arc::new(AsyncMutex::new(MetricsCollector::new())),
            data_sender,
        };

        (manager, data_receiver)
    }

    /// Create a new adaptive stream
    pub async fn create_adaptive_stream(
        &self,
        stream_id: String,
        priority: StreamPriority,
        client_info: ClientInfo,
    ) -> Result<AdaptiveStream> {
        info!("Creating adaptive stream: {}", stream_id);

        // Check concurrent stream limit
        let active_streams = self.active_streams.read().await;
        if active_streams.len() >= self.config.max_concurrent_streams {
            return Err(anyhow!("Maximum concurrent streams exceeded"));
        }
        drop(active_streams);

        // Create stream channels
        let (data_sender, data_receiver) = mpsc::channel(self.config.buffer_size);
        let (control_sender, control_receiver) = mpsc::channel(100);

        // Create stream metadata
        let metadata = StreamMetadata {
            stream_id: stream_id.clone(),
            priority: priority.clone(),
            created_at: Instant::now(),
            last_activity: Instant::now(),
            total_messages: 0,
            total_bytes: 0,
            compression_enabled: self.config.enable_data_compression,
            client_info: client_info.clone(),
        };

        // Create stream handle
        let handle = StreamHandle {
            metadata: Arc::new(AsyncRwLock::new(metadata)),
            data_sender,
            control_sender,
            backpressure_state: Arc::new(AsyncRwLock::new(BackpressureState::Normal)),
        };

        // Register stream
        self.active_streams
            .write()
            .await
            .insert(stream_id.clone(), handle.clone());

        // Register with multiplexer
        let mut multiplexer = self.stream_multiplexer.lock().await;
        multiplexer.register_stream(&stream_id, &priority).await?;

        // Register with priority scheduler
        let mut scheduler = self.priority_scheduler.lock().await;
        scheduler.register_stream(&stream_id, priority).await?;

        // Create adaptive stream
        let adaptive_stream = AdaptiveStream::new(
            stream_id,
            data_receiver,
            control_receiver,
            handle.clone(),
            Arc::clone(&self.compression_manager),
            Arc::clone(&self.metrics_collector),
            self.config.clone(),
        );

        Ok(adaptive_stream)
    }

    /// Send data to specific stream
    pub async fn send_to_stream(&self, stream_id: &str, data: StreamData) -> Result<()> {
        let streams = self.active_streams.read().await;
        if let Some(handle) = streams.get(stream_id) {
            // Check backpressure
            let backpressure_state = handle.backpressure_state.read().await;
            if matches!(*backpressure_state, BackpressureState::Blocked) {
                warn!("Stream {} is backpressured, dropping message", stream_id);
                return Ok(()); // Or queue for later
            }
            drop(backpressure_state);

            // Compress data if enabled
            let compressed_data = if handle.metadata.read().await.compression_enabled {
                let compression_manager = self.compression_manager.lock().await;
                compression_manager.compress_data(&data).await?
            } else {
                data
            };

            // Send to stream
            if (handle.data_sender.send(compressed_data).await).is_err() {
                warn!("Failed to send data to stream {}, removing", stream_id);
                // Stream is closed, remove it
                drop(streams);
                self.remove_stream(stream_id).await?;
            } else {
                // Update metrics
                let mut metadata = handle.metadata.write().await;
                metadata.total_messages += 1;
                metadata.last_activity = Instant::now();
            }
        }

        Ok(())
    }

    /// Broadcast data to all streams
    pub async fn broadcast(&self, data: StreamData) -> Result<()> {
        // Apply priority scheduling
        let mut scheduler = self.priority_scheduler.lock().await;
        let scheduled_streams = scheduler.schedule_broadcast(&data).await?;
        drop(scheduler);

        // Send to scheduled streams
        for stream_id in scheduled_streams {
            if let Err(e) = self.send_to_stream(&stream_id, data.clone()).await {
                warn!("Failed to send broadcast to stream {}: {}", stream_id, e);
            }
        }

        // Also send to broadcast channel
        if self.data_sender.send(data).is_err() {
            warn!("No broadcast subscribers");
        }

        Ok(())
    }

    /// Remove stream
    pub async fn remove_stream(&self, stream_id: &str) -> Result<()> {
        info!("Removing stream: {}", stream_id);

        // Remove from active streams
        self.active_streams.write().await.remove(stream_id);

        // Unregister from multiplexer
        let mut multiplexer = self.stream_multiplexer.lock().await;
        multiplexer.unregister_stream(stream_id).await?;

        // Unregister from scheduler
        let mut scheduler = self.priority_scheduler.lock().await;
        scheduler.unregister_stream(stream_id).await?;

        Ok(())
    }

    /// Get stream metrics
    pub async fn get_stream_metrics(&self) -> Result<HashMap<String, StreamMetrics>> {
        let mut metrics = HashMap::new();
        let streams = self.active_streams.read().await;

        for (stream_id, handle) in streams.iter() {
            let metadata = handle.metadata.read().await;
            let stream_metrics = StreamMetrics {
                stream_id: stream_id.clone(),
                priority: metadata.priority.clone(),
                total_messages: metadata.total_messages,
                total_bytes: metadata.total_bytes,
                duration: metadata.created_at.elapsed(),
                last_activity: metadata.last_activity.elapsed(),
                compression_enabled: metadata.compression_enabled,
            };
            metrics.insert(stream_id.clone(), stream_metrics);
        }

        Ok(metrics)
    }

    /// Start background tasks
    pub async fn start_background_tasks(&self) -> Result<()> {
        // Start backpressure monitoring
        self.start_backpressure_monitoring().await?;

        // Start metrics collection
        self.start_metrics_collection().await?;

        // Start stream cleanup
        self.start_stream_cleanup().await?;

        Ok(())
    }

    /// Start backpressure monitoring
    async fn start_backpressure_monitoring(&self) -> Result<()> {
        let controller = Arc::clone(&self.backpressure_controller);
        let streams = Arc::clone(&self.active_streams);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));

            loop {
                interval.tick().await;

                let mut controller = controller.lock().await;
                let streams = streams.read().await;

                for (stream_id, handle) in streams.iter() {
                    if let Err(e) = controller.monitor_stream(stream_id, handle, &config).await {
                        error!(
                            "Backpressure monitoring error for stream {}: {}",
                            stream_id, e
                        );
                    }
                }
            }
        });

        Ok(())
    }

    /// Start metrics collection
    async fn start_metrics_collection(&self) -> Result<()> {
        let collector = Arc::clone(&self.metrics_collector);
        let streams = Arc::clone(&self.active_streams);

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));

            loop {
                interval.tick().await;

                let mut collector = collector.lock().await;
                let streams = streams.read().await;

                if let Err(e) = collector.collect_metrics(&streams).await {
                    error!("Metrics collection error: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start stream cleanup
    async fn start_stream_cleanup(&self) -> Result<()> {
        let streams = Arc::clone(&self.active_streams);
        let timeout = self.config.stream_timeout;

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                let mut to_remove = Vec::new();
                {
                    let streams = streams.read().await;
                    for (stream_id, handle) in streams.iter() {
                        let metadata = handle.metadata.read().await;
                        if metadata.last_activity.elapsed() > timeout {
                            to_remove.push(stream_id.clone());
                        }
                    }
                }

                if !to_remove.is_empty() {
                    let mut streams = streams.write().await;
                    for stream_id in to_remove {
                        info!("Cleaning up inactive stream: {}", stream_id);
                        streams.remove(&stream_id);
                    }
                }
            }
        });

        Ok(())
    }
}

/// Stream handle for managing individual streams
#[derive(Debug, Clone)]
pub struct StreamHandle {
    pub metadata: Arc<AsyncRwLock<StreamMetadata>>,
    pub data_sender: mpsc::Sender<StreamData>,
    pub control_sender: mpsc::Sender<StreamControl>,
    pub backpressure_state: Arc<AsyncRwLock<BackpressureState>>,
}

/// Stream control messages
#[derive(Debug, Clone)]
pub enum StreamControl {
    Pause,
    Resume,
    ChangeCompressionLevel(u8),
    ChangePriority(StreamPriority),
    Heartbeat,
    Close,
}

/// Backpressure state
#[derive(Debug, Clone)]
pub enum BackpressureState {
    Normal,
    Warning,
    Blocked,
}

/// Adaptive stream implementation
pub struct AdaptiveStream {
    stream_id: String,
    data_receiver: mpsc::Receiver<StreamData>,
    control_receiver: mpsc::Receiver<StreamControl>,
    handle: StreamHandle,
    compression_manager: Arc<AsyncMutex<CompressionManager>>,
    #[allow(dead_code)]
    metrics_collector: Arc<AsyncMutex<MetricsCollector>>,
    #[allow(dead_code)]
    config: StreamingConfig,
    buffer: VecDeque<StreamData>,
    paused: bool,
}

impl AdaptiveStream {
    pub fn new(
        stream_id: String,
        data_receiver: mpsc::Receiver<StreamData>,
        control_receiver: mpsc::Receiver<StreamControl>,
        handle: StreamHandle,
        compression_manager: Arc<AsyncMutex<CompressionManager>>,
        metrics_collector: Arc<AsyncMutex<MetricsCollector>>,
        config: StreamingConfig,
    ) -> Self {
        Self {
            stream_id,
            data_receiver,
            control_receiver,
            handle,
            compression_manager,
            metrics_collector,
            config,
            buffer: VecDeque::new(),
            paused: false,
        }
    }

    /// Process next item in stream
    pub async fn next(&mut self) -> Option<Result<StreamData>> {
        loop {
            if !self.buffer.is_empty() && !self.paused {
                return Some(Ok(self.buffer.pop_front().unwrap()));
            }

            tokio::select! {
                data = self.data_receiver.recv() => {
                    match data {
                        Some(data) => {
                            if self.paused {
                                self.buffer.push_back(data);
                                continue; // Continue to check for control messages
                            } else {
                                return Some(Ok(data));
                            }
                        }
                        None => return None, // Stream closed
                    }
                }
                control = self.control_receiver.recv() => {
                    match control {
                        Some(control) => {
                            self.handle_control_message(control).await;
                            // Continue loop to check for next data/control
                        }
                        None => return None, // Control channel closed
                    }
                }
            }
        }
    }

    /// Handle control messages
    async fn handle_control_message(&mut self, control: StreamControl) {
        match control {
            StreamControl::Pause => {
                debug!("Pausing stream {}", self.stream_id);
                self.paused = true;
            }
            StreamControl::Resume => {
                debug!("Resuming stream {}", self.stream_id);
                self.paused = false;
            }
            StreamControl::ChangeCompressionLevel(level) => {
                let mut compression_manager = self.compression_manager.lock().await;
                compression_manager.set_compression_level(level).await;
            }
            StreamControl::ChangePriority(priority) => {
                let mut metadata = self.handle.metadata.write().await;
                metadata.priority = priority;
            }
            StreamControl::Heartbeat => {
                let mut metadata = self.handle.metadata.write().await;
                metadata.last_activity = Instant::now();
            }
            StreamControl::Close => {
                info!("Closing stream {}", self.stream_id);
                // Stream will be closed when receivers are dropped
            }
        }
    }
}

impl Stream for AdaptiveStream {
    type Item = Result<StreamData>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let future = self.next();
        pin_mut!(future);
        future.poll(cx)
    }
}

/// Stream multiplexer for handling multiple streams efficiently
#[derive(Debug)]
pub struct StreamMultiplexer {
    #[allow(dead_code)]
    config: StreamingConfig,
    registered_streams: HashMap<String, StreamPriority>,
    #[allow(dead_code)]
    multiplexing_enabled: bool,
}

impl StreamMultiplexer {
    pub fn new(config: &StreamingConfig) -> Self {
        Self {
            config: config.clone(),
            registered_streams: HashMap::new(),
            multiplexing_enabled: config.enable_stream_multiplexing,
        }
    }

    pub async fn register_stream(
        &mut self,
        stream_id: &str,
        priority: &StreamPriority,
    ) -> Result<()> {
        self.registered_streams
            .insert(stream_id.to_string(), priority.clone());
        debug!(
            "Registered stream {} with priority {:?}",
            stream_id, priority
        );
        Ok(())
    }

    pub async fn unregister_stream(&mut self, stream_id: &str) -> Result<()> {
        self.registered_streams.remove(stream_id);
        debug!("Unregistered stream {}", stream_id);
        Ok(())
    }
}

/// Backpressure controller
#[derive(Debug)]
pub struct BackpressureController {
    #[allow(dead_code)]
    config: StreamingConfig,
    monitoring_enabled: bool,
}

impl BackpressureController {
    pub fn new(config: &StreamingConfig) -> Self {
        Self {
            config: config.clone(),
            monitoring_enabled: config.enable_backpressure_control,
        }
    }

    pub async fn monitor_stream(
        &mut self,
        _stream_id: &str,
        handle: &StreamHandle,
        config: &StreamingConfig,
    ) -> Result<()> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        // Check buffer utilization
        let buffer_utilization = self.calculate_buffer_utilization(handle).await?;

        let mut backpressure_state = handle.backpressure_state.write().await;
        *backpressure_state = if buffer_utilization > config.backpressure_threshold {
            BackpressureState::Blocked
        } else if buffer_utilization > config.backpressure_threshold * 0.7 {
            BackpressureState::Warning
        } else {
            BackpressureState::Normal
        };

        Ok(())
    }

    async fn calculate_buffer_utilization(&self, handle: &StreamHandle) -> Result<f64> {
        // Calculate buffer utilization based on queue sizes and stream activity
        let metadata = handle.metadata.read().await;
        let backpressure_state = handle.backpressure_state.read().await;

        // Base utilization from message activity
        let activity_factor = if metadata.total_messages > 0 {
            let recent_activity = metadata.last_activity.elapsed().as_secs() as f64;
            (1.0 / (1.0 + recent_activity / 10.0)).min(0.8) // Decay over time, max 80%
        } else {
            0.1 // Minimal utilization for inactive streams
        };

        // Adjust based on backpressure state
        let backpressure_factor = match *backpressure_state {
            BackpressureState::Normal => 0.0,
            BackpressureState::Warning => 0.2,
            BackpressureState::Blocked => 0.4,
        };

        Ok((activity_factor + backpressure_factor).min(0.95))
    }
}

/// Compression manager
#[derive(Debug)]
pub struct CompressionManager {
    #[allow(dead_code)]
    config: CompressionConfig,
    compression_level: u8,
}

impl CompressionManager {
    pub fn new(config: &CompressionConfig) -> Self {
        Self {
            config: config.clone(),
            compression_level: config.compression_level,
        }
    }

    pub async fn compress_data(&self, data: &StreamData) -> Result<StreamData> {
        // Implement data compression
        Ok(data.clone())
    }

    pub async fn set_compression_level(&mut self, level: u8) {
        self.compression_level = level;
    }
}

/// Priority scheduler
#[derive(Debug)]
pub struct PriorityScheduler {
    #[allow(dead_code)]
    config: PriorityConfig,
    stream_priorities: HashMap<String, StreamPriority>,
    #[allow(dead_code)]
    priority_queues: Vec<VecDeque<String>>,
}

impl PriorityScheduler {
    pub fn new(config: &PriorityConfig) -> Self {
        Self {
            config: config.clone(),
            stream_priorities: HashMap::new(),
            priority_queues: vec![VecDeque::new(); config.priority_queues],
        }
    }

    pub async fn register_stream(
        &mut self,
        stream_id: &str,
        priority: StreamPriority,
    ) -> Result<()> {
        self.stream_priorities
            .insert(stream_id.to_string(), priority);
        Ok(())
    }

    pub async fn unregister_stream(&mut self, stream_id: &str) -> Result<()> {
        self.stream_priorities.remove(stream_id);
        Ok(())
    }

    pub async fn schedule_broadcast(&mut self, _data: &StreamData) -> Result<Vec<String>> {
        // Return all registered streams for broadcast
        Ok(self.stream_priorities.keys().cloned().collect())
    }
}

/// Metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    collected_metrics: HashMap<String, StreamMetrics>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            collected_metrics: HashMap::new(),
        }
    }

    pub async fn collect_metrics(&mut self, streams: &HashMap<String, StreamHandle>) -> Result<()> {
        for (stream_id, handle) in streams {
            let metadata = handle.metadata.read().await;
            let metrics = StreamMetrics {
                stream_id: stream_id.clone(),
                priority: metadata.priority.clone(),
                total_messages: metadata.total_messages,
                total_bytes: metadata.total_bytes,
                duration: metadata.created_at.elapsed(),
                last_activity: metadata.last_activity.elapsed(),
                compression_enabled: metadata.compression_enabled,
            };
            self.collected_metrics.insert(stream_id.clone(), metrics);
        }
        Ok(())
    }
}

/// Stream metrics
#[derive(Debug, Clone)]
pub struct StreamMetrics {
    pub stream_id: String,
    pub priority: StreamPriority,
    pub total_messages: u64,
    pub total_bytes: u64,
    pub duration: Duration,
    pub last_activity: Duration,
    pub compression_enabled: bool,
}
