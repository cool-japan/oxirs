//! Backpressure and Flow Control
//!
//! This module provides sophisticated backpressure handling for stream processing:
//! - Dynamic rate limiting based on system load
//! - Buffer management with overflow strategies
//! - Flow control signals
//! - Adaptive throttling
//! - Queue depth monitoring
//!
//! Uses SciRS2 for adaptive algorithm tuning

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, warn};

// Use scirs2-core for adaptive algorithms (reserved for future use)

/// Backpressure strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackpressureStrategy {
    /// Drop oldest events when buffer is full
    DropOldest,
    /// Drop newest events when buffer is full
    DropNewest,
    /// Block until space is available
    Block,
    /// Exponential backoff with retries
    ExponentialBackoff {
        initial_delay_ms: u64,
        max_delay_ms: u64,
        multiplier: f64,
    },
    /// Adaptive throttling based on throughput
    Adaptive {
        target_throughput: f64,
        adjustment_factor: f64,
    },
}

/// Flow control signal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowControlSignal {
    /// System is healthy, proceed normally
    Proceed,
    /// System is under pressure, slow down
    SlowDown,
    /// System is overloaded, stop sending
    Stop,
}

/// Backpressure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureConfig {
    /// Maximum buffer size
    pub max_buffer_size: usize,
    /// Backpressure strategy
    pub strategy: BackpressureStrategy,
    /// High water mark (percentage of buffer)
    pub high_water_mark: f64,
    /// Low water mark (percentage of buffer)
    pub low_water_mark: f64,
    /// Enable adaptive throttling
    pub enable_adaptive: bool,
    /// Measurement window for throughput
    pub measurement_window: ChronoDuration,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10000,
            strategy: BackpressureStrategy::Block,
            high_water_mark: 0.8,
            low_water_mark: 0.2,
            enable_adaptive: true,
            measurement_window: ChronoDuration::seconds(10),
        }
    }
}

/// Backpressure controller statistics
#[derive(Debug, Clone, Default)]
pub struct BackpressureStats {
    pub events_received: u64,
    pub events_processed: u64,
    pub events_dropped: u64,
    pub events_blocked: u64,
    pub buffer_size: usize,
    pub buffer_utilization: f64,
    pub current_throughput: f64,
    pub backpressure_events: u64,
    pub avg_latency_ms: f64,
}

/// Type alias for timestamped buffer elements
type TimestampedBuffer<T> = Arc<Mutex<VecDeque<(T, DateTime<Utc>)>>>;

/// Type alias for throughput history
type ThroughputHistory = Arc<Mutex<VecDeque<(DateTime<Utc>, u64)>>>;

/// Backpressure controller
pub struct BackpressureController<T> {
    config: BackpressureConfig,
    buffer: TimestampedBuffer<T>,
    stats: Arc<Mutex<BackpressureStats>>,
    flow_control: Arc<Mutex<FlowControlSignal>>,
    semaphore: Arc<Semaphore>,
    throughput_history: ThroughputHistory,
}

impl<T: Clone + Send> BackpressureController<T> {
    /// Create a new backpressure controller
    pub fn new(config: BackpressureConfig) -> Self {
        let max_permits = config.max_buffer_size;

        Self {
            config,
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(BackpressureStats::default())),
            flow_control: Arc::new(Mutex::new(FlowControlSignal::Proceed)),
            semaphore: Arc::new(Semaphore::new(max_permits)),
            throughput_history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Offer an event to the controller
    pub async fn offer(&self, event: T) -> Result<()> {
        let mut stats = self.stats.lock().await;
        stats.events_received += 1;
        drop(stats);

        match &self.config.strategy {
            BackpressureStrategy::DropOldest => self.offer_drop_oldest(event).await,
            BackpressureStrategy::DropNewest => self.offer_drop_newest(event).await,
            BackpressureStrategy::Block => self.offer_blocking(event).await,
            BackpressureStrategy::ExponentialBackoff {
                initial_delay_ms,
                max_delay_ms,
                multiplier,
            } => {
                self.offer_with_backoff(event, *initial_delay_ms, *max_delay_ms, *multiplier)
                    .await
            }
            BackpressureStrategy::Adaptive {
                target_throughput,
                adjustment_factor,
            } => {
                self.offer_adaptive(event, *target_throughput, *adjustment_factor)
                    .await
            }
        }
    }

    /// Offer with drop oldest strategy
    async fn offer_drop_oldest(&self, event: T) -> Result<()> {
        let mut buffer = self.buffer.lock().await;

        if buffer.len() >= self.config.max_buffer_size {
            // Drop oldest
            buffer.pop_front();

            let mut stats = self.stats.lock().await;
            stats.events_dropped += 1;
            drop(stats);

            warn!("Buffer full, dropped oldest event");
        }

        buffer.push_back((event, Utc::now()));
        self.update_flow_control(buffer.len()).await;

        Ok(())
    }

    /// Offer with drop newest strategy
    async fn offer_drop_newest(&self, event: T) -> Result<()> {
        let mut buffer = self.buffer.lock().await;

        if buffer.len() >= self.config.max_buffer_size {
            let mut stats = self.stats.lock().await;
            stats.events_dropped += 1;
            drop(stats);

            warn!("Buffer full, dropped newest event");
            return Ok(());
        }

        buffer.push_back((event, Utc::now()));
        self.update_flow_control(buffer.len()).await;

        Ok(())
    }

    /// Offer with blocking strategy
    async fn offer_blocking(&self, event: T) -> Result<()> {
        // Acquire semaphore permit
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| anyhow!("Failed to acquire semaphore: {}", e))?;

        let mut buffer = self.buffer.lock().await;
        buffer.push_back((event, Utc::now()));

        let buffer_size = buffer.len();
        drop(buffer);

        self.update_flow_control(buffer_size).await;

        Ok(())
    }

    /// Offer with exponential backoff
    async fn offer_with_backoff(
        &self,
        event: T,
        initial_delay_ms: u64,
        max_delay_ms: u64,
        multiplier: f64,
    ) -> Result<()> {
        let mut delay_ms = initial_delay_ms;
        let mut retries = 0;
        const MAX_RETRIES: u32 = 10;

        loop {
            let buffer = self.buffer.lock().await;
            let buffer_size = buffer.len();
            drop(buffer);

            if buffer_size < self.config.max_buffer_size {
                let mut buffer = self.buffer.lock().await;
                buffer.push_back((event, Utc::now()));
                drop(buffer);

                self.update_flow_control(buffer_size + 1).await;
                return Ok(());
            }

            if retries >= MAX_RETRIES {
                let mut stats = self.stats.lock().await;
                stats.events_dropped += 1;
                return Err(anyhow!("Max retries exceeded, dropping event"));
            }

            // Exponential backoff
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;

            delay_ms = ((delay_ms as f64 * multiplier) as u64).min(max_delay_ms);
            retries += 1;

            let mut stats = self.stats.lock().await;
            stats.events_blocked += 1;
            drop(stats);
        }
    }

    /// Offer with adaptive throttling using SciRS2
    async fn offer_adaptive(
        &self,
        event: T,
        target_throughput: f64,
        adjustment_factor: f64,
    ) -> Result<()> {
        // Measure current throughput
        let current_throughput = self.measure_throughput().await;

        // Adaptive delay based on throughput
        if current_throughput > target_throughput {
            let delay_ms =
                ((current_throughput / target_throughput - 1.0) * adjustment_factor) as u64;
            tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
        }

        // Add to buffer
        let mut buffer = self.buffer.lock().await;

        if buffer.len() >= self.config.max_buffer_size {
            let mut stats = self.stats.lock().await;
            stats.events_dropped += 1;
            drop(stats);

            return Err(anyhow!("Buffer full even with adaptive throttling"));
        }

        buffer.push_back((event, Utc::now()));
        let buffer_size = buffer.len();
        drop(buffer);

        self.update_flow_control(buffer_size).await;

        Ok(())
    }

    /// Poll an event from the controller
    pub async fn poll(&self) -> Result<Option<T>> {
        let mut buffer = self.buffer.lock().await;

        if let Some((event, timestamp)) = buffer.pop_front() {
            let buffer_size = buffer.len();
            drop(buffer);

            // Release semaphore permit
            self.semaphore.add_permits(1);

            // Update stats
            let mut stats = self.stats.lock().await;
            stats.events_processed += 1;

            let latency = (Utc::now() - timestamp).num_milliseconds() as f64;
            let alpha = 0.1;
            stats.avg_latency_ms = alpha * latency + (1.0 - alpha) * stats.avg_latency_ms;

            drop(stats);

            self.update_flow_control(buffer_size).await;
            self.record_throughput().await;

            Ok(Some(event))
        } else {
            Ok(None)
        }
    }

    /// Update flow control signal
    async fn update_flow_control(&self, buffer_size: usize) {
        let utilization = buffer_size as f64 / self.config.max_buffer_size as f64;

        let signal = if utilization >= self.config.high_water_mark {
            FlowControlSignal::Stop
        } else if utilization >= self.config.low_water_mark {
            FlowControlSignal::SlowDown
        } else {
            FlowControlSignal::Proceed
        };

        let mut flow_control = self.flow_control.lock().await;
        if *flow_control != signal {
            debug!(
                "Flow control signal changed: {:?} -> {:?}",
                *flow_control, signal
            );

            if signal != FlowControlSignal::Proceed {
                let mut stats = self.stats.lock().await;
                stats.backpressure_events += 1;
            }
        }
        *flow_control = signal;

        // Update stats
        let mut stats = self.stats.lock().await;
        stats.buffer_size = buffer_size;
        stats.buffer_utilization = utilization;
    }

    /// Record throughput measurement
    async fn record_throughput(&self) {
        let now = Utc::now();
        let mut history = self.throughput_history.lock().await;

        history.push_back((now, 1));

        // Clean old measurements
        let window_start = now - self.config.measurement_window;
        while let Some((timestamp, _)) = history.front() {
            if *timestamp < window_start {
                history.pop_front();
            } else {
                break;
            }
        }
    }

    /// Measure current throughput
    async fn measure_throughput(&self) -> f64 {
        let now = Utc::now();
        let history = self.throughput_history.lock().await;

        if history.is_empty() {
            return 0.0;
        }

        let window_start = now - self.config.measurement_window;
        let count: u64 = history
            .iter()
            .filter(|(timestamp, _)| *timestamp >= window_start)
            .map(|(_, count)| count)
            .sum();

        let elapsed_seconds = self.config.measurement_window.num_seconds() as f64;
        count as f64 / elapsed_seconds
    }

    /// Get current flow control signal
    pub async fn flow_control_signal(&self) -> FlowControlSignal {
        *self.flow_control.lock().await
    }

    /// Get statistics
    pub async fn stats(&self) -> BackpressureStats {
        let stats = self.stats.lock().await;
        let mut result = stats.clone();

        // Update current throughput
        drop(stats);
        result.current_throughput = self.measure_throughput().await;

        result
    }

    /// Get buffer size
    pub async fn buffer_size(&self) -> usize {
        self.buffer.lock().await.len()
    }

    /// Clear buffer
    pub async fn clear(&self) {
        let mut buffer = self.buffer.lock().await;
        let cleared_count = buffer.len();
        buffer.clear();

        // Release all permits
        self.semaphore.add_permits(cleared_count);

        let mut stats = self.stats.lock().await;
        stats.buffer_size = 0;
        stats.buffer_utilization = 0.0;
    }
}

/// Rate limiter with token bucket algorithm
pub struct RateLimiter {
    tokens: Arc<Mutex<f64>>,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Arc<Mutex<DateTime<Utc>>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: Arc::new(Mutex::new(max_tokens)),
            max_tokens,
            refill_rate,
            last_refill: Arc::new(Mutex::new(Utc::now())),
        }
    }

    /// Try to acquire a token
    pub async fn try_acquire(&self) -> bool {
        self.refill_tokens().await;

        let mut tokens = self.tokens.lock().await;
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Acquire a token (blocking)
    pub async fn acquire(&self) -> Result<()> {
        loop {
            if self.try_acquire().await {
                return Ok(());
            }

            // Wait for refill
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
    }

    /// Refill tokens based on elapsed time
    async fn refill_tokens(&self) {
        let now = Utc::now();
        let mut last_refill = self.last_refill.lock().await;

        let elapsed = (now - *last_refill).num_milliseconds() as f64 / 1000.0;
        let new_tokens = elapsed * self.refill_rate;

        if new_tokens > 0.0 {
            let mut tokens = self.tokens.lock().await;
            *tokens = (*tokens + new_tokens).min(self.max_tokens);
            *last_refill = now;
        }
    }

    /// Get current token count
    pub async fn available_tokens(&self) -> f64 {
        self.refill_tokens().await;
        *self.tokens.lock().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backpressure_drop_oldest() {
        let config = BackpressureConfig {
            max_buffer_size: 3,
            strategy: BackpressureStrategy::DropOldest,
            ..Default::default()
        };

        let controller = BackpressureController::new(config);

        // Fill buffer
        for i in 0..5 {
            controller.offer(i).await.unwrap();
        }

        // Should have dropped 2 oldest (0, 1)
        assert_eq!(controller.buffer_size().await, 3);

        // Poll should return 2 (oldest after drops)
        let event = controller.poll().await.unwrap().unwrap();
        assert_eq!(event, 2);
    }

    #[tokio::test]
    async fn test_backpressure_drop_newest() {
        let config = BackpressureConfig {
            max_buffer_size: 3,
            strategy: BackpressureStrategy::DropNewest,
            ..Default::default()
        };

        let controller = BackpressureController::new(config);

        // Fill buffer
        for i in 0..5 {
            controller.offer(i).await.unwrap();
        }

        // Should have kept first 3, dropped 3 and 4
        assert_eq!(controller.buffer_size().await, 3);

        let event = controller.poll().await.unwrap().unwrap();
        assert_eq!(event, 0);
    }

    #[tokio::test]
    async fn test_flow_control_signals() {
        let config = BackpressureConfig {
            max_buffer_size: 100,
            high_water_mark: 0.8,
            low_water_mark: 0.2,
            ..Default::default()
        };

        let controller = BackpressureController::new(config);

        // Low utilization
        assert_eq!(
            controller.flow_control_signal().await,
            FlowControlSignal::Proceed
        );

        // Fill to medium utilization
        for i in 0..30 {
            controller.offer(i).await.unwrap();
        }

        assert_eq!(
            controller.flow_control_signal().await,
            FlowControlSignal::SlowDown
        );

        // Fill to high utilization
        for i in 30..85 {
            controller.offer(i).await.unwrap();
        }

        assert_eq!(
            controller.flow_control_signal().await,
            FlowControlSignal::Stop
        );
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(10.0, 10.0); // 10 tokens, 10/second refill

        // Should be able to acquire 10 tokens
        for _ in 0..10 {
            assert!(limiter.try_acquire().await);
        }

        // 11th should fail
        assert!(!limiter.try_acquire().await);

        // Wait for refill
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Should be able to acquire again
        assert!(limiter.try_acquire().await);
    }
}
