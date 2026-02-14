//! Backpressure and Flow Control
//!
//! This module provides sophisticated backpressure handling for stream processing:
//! - Dynamic rate limiting based on system load
//! - Buffer management with overflow strategies
//! - Flow control signals
//! - Adaptive throttling
//! - Queue depth monitoring
//! - Circuit breaker pattern for fault tolerance
//! - Graceful degradation strategies
//!
//! Uses SciRS2 metrics for comprehensive monitoring

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use scirs2_core::metrics::{Counter, Gauge, Histogram};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use tracing::{debug, info, warn};

type BoxFuture<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Circuit breaker state for backpressure control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CircuitState {
    /// Circuit is closed, normal operation
    #[default]
    Closed,
    /// Circuit is open, rejecting requests
    Open,
    /// Circuit is half-open, testing recovery
    HalfOpen,
}

/// Graceful degradation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DegradationStrategy {
    /// Reduce throughput by percentage
    ReduceThroughput { reduction_percent: f64 },
    /// Skip non-critical operations
    SkipNonCritical,
    /// Increase buffer size temporarily
    ExpandBuffer { factor: f64 },
    /// Sample events (keep every Nth event)
    Sampling { sample_rate: f64 },
    /// Combined strategies
    Combined(Vec<DegradationStrategy>),
}

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

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Enable circuit breaker
    pub enabled: bool,
    /// Failure threshold to open circuit
    pub failure_threshold: u32,
    /// Success threshold to close circuit
    pub success_threshold: u32,
    /// Timeout before transitioning to half-open
    pub timeout: Duration,
    /// Maximum calls in half-open state
    pub half_open_max_calls: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            half_open_max_calls: 3,
        }
    }
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
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Degradation strategy when under pressure
    pub degradation: DegradationStrategy,
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
            circuit_breaker: CircuitBreakerConfig::default(),
            degradation: DegradationStrategy::ReduceThroughput {
                reduction_percent: 50.0,
            },
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
    pub circuit_state: CircuitState,
    pub circuit_failures: u32,
    pub circuit_successes: u32,
    pub degradation_active: bool,
}

/// Type alias for timestamped buffer elements
type TimestampedBuffer<T> = Arc<Mutex<VecDeque<(T, DateTime<Utc>)>>>;

/// Type alias for throughput history
type ThroughputHistory = Arc<Mutex<VecDeque<(DateTime<Utc>, u64)>>>;

/// Exported metrics snapshot
#[derive(Debug, Clone)]
pub struct BackpressureMetrics {
    pub events_received: u64,
    pub events_processed: u64,
    pub events_dropped: u64,
    pub queue_depth: f64,
    pub latency_stats: scirs2_core::metrics::HistogramStats,
    pub backpressure_events: u64,
    pub circuit_state_changes: u64,
}

/// Backpressure controller
pub struct BackpressureController<T> {
    config: BackpressureConfig,
    buffer: TimestampedBuffer<T>,
    stats: Arc<Mutex<BackpressureStats>>,
    flow_control: Arc<Mutex<FlowControlSignal>>,
    semaphore: Arc<Semaphore>,
    throughput_history: ThroughputHistory,
    // Circuit breaker state
    circuit_state: Arc<Mutex<CircuitState>>,
    circuit_failures: Arc<Mutex<u32>>,
    circuit_successes: Arc<Mutex<u32>>,
    circuit_last_failure: Arc<Mutex<Option<Instant>>>,
    circuit_half_open_calls: Arc<Mutex<u32>>,
    // SciRS2 metrics
    metrics_events_received: Arc<Counter>,
    metrics_events_processed: Arc<Counter>,
    metrics_events_dropped: Arc<Counter>,
    metrics_queue_depth: Arc<Gauge>,
    metrics_latency: Arc<Histogram>,
    metrics_backpressure_events: Arc<Counter>,
    metrics_circuit_state_changes: Arc<Counter>,
}

impl<T: Clone + Send> BackpressureController<T> {
    /// Create a new backpressure controller
    pub fn new(config: BackpressureConfig) -> Self {
        let max_permits = config.max_buffer_size;

        // Initialize SciRS2 metrics
        let metrics_events_received =
            Arc::new(Counter::new("backpressure_events_received".to_string()));
        let metrics_events_processed =
            Arc::new(Counter::new("backpressure_events_processed".to_string()));
        let metrics_events_dropped =
            Arc::new(Counter::new("backpressure_events_dropped".to_string()));
        let metrics_queue_depth = Arc::new(Gauge::new("backpressure_queue_depth".to_string()));
        let metrics_latency = Arc::new(Histogram::new("backpressure_latency_seconds".to_string()));
        let metrics_backpressure_events =
            Arc::new(Counter::new("backpressure_events_total".to_string()));
        let metrics_circuit_state_changes = Arc::new(Counter::new(
            "backpressure_circuit_state_changes".to_string(),
        ));

        Self {
            config,
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(BackpressureStats::default())),
            flow_control: Arc::new(Mutex::new(FlowControlSignal::Proceed)),
            semaphore: Arc::new(Semaphore::new(max_permits)),
            throughput_history: Arc::new(Mutex::new(VecDeque::new())),
            circuit_state: Arc::new(Mutex::new(CircuitState::Closed)),
            circuit_failures: Arc::new(Mutex::new(0)),
            circuit_successes: Arc::new(Mutex::new(0)),
            circuit_last_failure: Arc::new(Mutex::new(None)),
            circuit_half_open_calls: Arc::new(Mutex::new(0)),
            metrics_events_received,
            metrics_events_processed,
            metrics_events_dropped,
            metrics_queue_depth,
            metrics_latency,
            metrics_backpressure_events,
            metrics_circuit_state_changes,
        }
    }

    /// Get metrics for export
    pub fn get_metrics(&self) -> BackpressureMetrics {
        BackpressureMetrics {
            events_received: self.metrics_events_received.get(),
            events_processed: self.metrics_events_processed.get(),
            events_dropped: self.metrics_events_dropped.get(),
            queue_depth: self.metrics_queue_depth.get(),
            latency_stats: self.metrics_latency.get_stats(),
            backpressure_events: self.metrics_backpressure_events.get(),
            circuit_state_changes: self.metrics_circuit_state_changes.get(),
        }
    }

    /// Check circuit breaker state and handle transitions
    async fn check_circuit_state(&self) -> Result<bool> {
        if !self.config.circuit_breaker.enabled {
            return Ok(true); // Circuit breaker disabled, always allow
        }

        let mut state = self.circuit_state.lock().await;
        let circuit_config = &self.config.circuit_breaker;

        match *state {
            CircuitState::Closed => Ok(true),
            CircuitState::Open => {
                let last_failure = self.circuit_last_failure.lock().await;
                if let Some(last_fail_time) = *last_failure {
                    if last_fail_time.elapsed() >= circuit_config.timeout {
                        // Transition to HalfOpen
                        *state = CircuitState::HalfOpen;
                        *self.circuit_half_open_calls.lock().await = 0;
                        self.metrics_circuit_state_changes.inc();
                        info!("Circuit breaker transitioned to HalfOpen");
                        Ok(true)
                    } else {
                        Ok(false) // Still open, reject request
                    }
                } else {
                    Ok(false)
                }
            }
            CircuitState::HalfOpen => {
                let mut half_open_calls = self.circuit_half_open_calls.lock().await;
                if *half_open_calls < circuit_config.half_open_max_calls {
                    *half_open_calls += 1;
                    Ok(true)
                } else {
                    Ok(false) // Too many calls in half-open state
                }
            }
        }
    }

    /// Record success for circuit breaker
    async fn record_circuit_success(&self) {
        if !self.config.circuit_breaker.enabled {
            return;
        }

        let mut state = self.circuit_state.lock().await;
        let circuit_config = &self.config.circuit_breaker;

        match *state {
            CircuitState::HalfOpen => {
                let mut successes = self.circuit_successes.lock().await;
                *successes += 1;
                if *successes >= circuit_config.success_threshold {
                    // Transition to Closed
                    *state = CircuitState::Closed;
                    *self.circuit_failures.lock().await = 0;
                    *self.circuit_successes.lock().await = 0;
                    self.metrics_circuit_state_changes.inc();
                    info!("Circuit breaker transitioned to Closed");
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                *self.circuit_failures.lock().await = 0;
            }
            CircuitState::Open => {
                // Should not happen, but reset if it does
                *state = CircuitState::Closed;
                *self.circuit_failures.lock().await = 0;
                self.metrics_circuit_state_changes.inc();
            }
        }
    }

    /// Record failure for circuit breaker
    async fn record_circuit_failure(&self) {
        if !self.config.circuit_breaker.enabled {
            return;
        }

        let mut state = self.circuit_state.lock().await;
        let circuit_config = &self.config.circuit_breaker;
        let mut failures = self.circuit_failures.lock().await;

        *failures += 1;
        *self.circuit_last_failure.lock().await = Some(Instant::now());

        if *failures >= circuit_config.failure_threshold && *state != CircuitState::Open {
            // Transition to Open
            *state = CircuitState::Open;
            *self.circuit_successes.lock().await = 0;
            self.metrics_circuit_state_changes.inc();
            warn!(
                "Circuit breaker transitioned to Open after {} failures",
                failures
            );
        }
    }

    /// Apply graceful degradation strategy
    async fn apply_degradation(&self, _event: &T) -> Result<bool> {
        let stats = self.stats.lock().await;
        let utilization = stats.buffer_utilization;
        drop(stats);

        // Only apply degradation when utilization is high
        if utilization < self.config.high_water_mark {
            return Ok(true); // No degradation needed
        }

        self.apply_degradation_strategy(&self.config.degradation)
            .await
    }

    /// Helper method to apply a specific degradation strategy
    fn apply_degradation_strategy<'a>(
        &'a self,
        strategy: &'a DegradationStrategy,
    ) -> BoxFuture<'a, Result<bool>> {
        Box::pin(async move {
            match strategy {
                DegradationStrategy::ReduceThroughput { reduction_percent } => {
                    // Randomly drop events based on reduction percentage
                    let threshold = 1.0 - (reduction_percent / 100.0);
                    Ok(fastrand::f64() < threshold)
                }
                DegradationStrategy::SkipNonCritical => {
                    // For now, accept all events (would need priority info)
                    Ok(true)
                }
                DegradationStrategy::ExpandBuffer { factor } => {
                    // Temporarily allow buffer to grow (check against expanded size)
                    let expanded_size = (self.config.max_buffer_size as f64 * factor) as usize;
                    let buffer = self.buffer.lock().await;
                    Ok(buffer.len() < expanded_size)
                }
                DegradationStrategy::Sampling { sample_rate } => {
                    // Keep events based on sample rate
                    Ok(fastrand::f64() < *sample_rate)
                }
                DegradationStrategy::Combined(strategies) => {
                    // Apply all strategies and accept only if all pass
                    for strat in strategies {
                        if !self.apply_degradation_strategy(strat).await? {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                }
            }
        })
    }

    /// Offer an event to the controller
    pub async fn offer(&self, event: T) -> Result<()> {
        // Update metrics
        self.metrics_events_received.inc();
        let mut stats = self.stats.lock().await;
        stats.events_received += 1;
        drop(stats);

        // Check circuit breaker
        if !self.check_circuit_state().await? {
            self.metrics_events_dropped.inc();
            return Err(anyhow!("Circuit breaker is open"));
        }

        // Apply graceful degradation
        if !self.apply_degradation(&event).await? {
            self.metrics_events_dropped.inc();
            let mut stats = self.stats.lock().await;
            stats.events_dropped += 1;
            stats.degradation_active = true;
            return Err(anyhow!("Event dropped due to graceful degradation"));
        }

        // Process event based on strategy
        let result = match &self.config.strategy {
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
        };

        // Update circuit breaker state based on result
        match &result {
            Ok(_) => self.record_circuit_success().await,
            Err(_) => self.record_circuit_failure().await,
        }

        result
    }

    /// Offer with drop oldest strategy
    async fn offer_drop_oldest(&self, event: T) -> Result<()> {
        let mut buffer = self.buffer.lock().await;

        if buffer.len() >= self.config.max_buffer_size {
            // Drop oldest
            buffer.pop_front();

            self.metrics_events_dropped.inc();
            let mut stats = self.stats.lock().await;
            stats.events_dropped += 1;
            drop(stats);

            warn!("Buffer full, dropped oldest event");
        }

        buffer.push_back((event, Utc::now()));
        let buffer_len = buffer.len();
        self.metrics_queue_depth.set(buffer_len as f64);
        drop(buffer);

        self.update_flow_control(buffer_len).await;

        Ok(())
    }

    /// Offer with drop newest strategy
    async fn offer_drop_newest(&self, event: T) -> Result<()> {
        let mut buffer = self.buffer.lock().await;

        if buffer.len() >= self.config.max_buffer_size {
            self.metrics_events_dropped.inc();
            let mut stats = self.stats.lock().await;
            stats.events_dropped += 1;
            drop(stats);

            warn!("Buffer full, dropped newest event");
            return Ok(());
        }

        buffer.push_back((event, Utc::now()));
        let buffer_len = buffer.len();
        self.metrics_queue_depth.set(buffer_len as f64);
        drop(buffer);

        self.update_flow_control(buffer_len).await;

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
            self.metrics_queue_depth.set(buffer_size as f64);
            drop(buffer);

            // Release semaphore permit
            self.semaphore.add_permits(1);

            // Calculate and record latency
            let latency = (Utc::now() - timestamp).num_milliseconds() as f64;
            self.metrics_latency.observe(latency / 1000.0); // Convert to seconds

            // Update metrics
            self.metrics_events_processed.inc();

            // Update stats
            let mut stats = self.stats.lock().await;
            stats.events_processed += 1;

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
                self.metrics_backpressure_events.inc();
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

        // Update circuit breaker info
        result.circuit_state = *self.circuit_state.lock().await;
        result.circuit_failures = *self.circuit_failures.lock().await;
        result.circuit_successes = *self.circuit_successes.lock().await;

        result
    }

    /// Get circuit breaker state
    pub async fn circuit_state(&self) -> CircuitState {
        *self.circuit_state.lock().await
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

    // Circuit Breaker Tests
    #[tokio::test]
    async fn test_circuit_breaker_closed_to_open() {
        let config = BackpressureConfig {
            max_buffer_size: 1, // Force failures
            strategy: BackpressureStrategy::DropNewest,
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 3,
                success_threshold: 2,
                timeout: Duration::from_millis(100),
                half_open_max_calls: 2,
            },
            ..Default::default()
        };

        let controller = BackpressureController::<i32>::new(config);

        // Initial state should be Closed
        assert_eq!(controller.circuit_state().await, CircuitState::Closed);

        // Fill buffer to cause failures
        for i in 0..10 {
            let _ = controller.offer(i).await;
        }

        // After enough failures, circuit should open
        assert_eq!(controller.circuit_state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_open_to_half_open() {
        let config = BackpressureConfig {
            max_buffer_size: 1,
            strategy: BackpressureStrategy::DropNewest,
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 3,
                success_threshold: 2,
                timeout: Duration::from_millis(50),
                half_open_max_calls: 2,
            },
            ..Default::default()
        };

        let controller = BackpressureController::<i32>::new(config);

        // Cause circuit to open
        for i in 0..10 {
            let _ = controller.offer(i).await;
        }

        assert_eq!(controller.circuit_state().await, CircuitState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Try to offer - this should transition to HalfOpen
        let _ = controller.check_circuit_state().await;
        assert_eq!(controller.circuit_state().await, CircuitState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_to_closed() {
        let config = BackpressureConfig {
            max_buffer_size: 100, // Large enough to allow successes
            strategy: BackpressureStrategy::Block,
            circuit_breaker: CircuitBreakerConfig {
                enabled: true,
                failure_threshold: 2,
                success_threshold: 2,
                timeout: Duration::from_millis(50),
                half_open_max_calls: 5,
            },
            ..Default::default()
        };

        let controller = BackpressureController::<i32>::new(config);

        // Manually set state to HalfOpen for testing
        *controller.circuit_state.lock().await = CircuitState::HalfOpen;

        // Record successes
        for _ in 0..2 {
            controller.record_circuit_success().await;
        }

        // Should transition to Closed
        assert_eq!(controller.circuit_state().await, CircuitState::Closed);
    }

    // Stress Tests
    #[tokio::test]
    async fn test_stress_high_load() {
        let config = BackpressureConfig {
            max_buffer_size: 1000,
            strategy: BackpressureStrategy::DropOldest,
            ..Default::default()
        };

        let controller = Arc::new(BackpressureController::new(config));

        // Spawn multiple producers
        let mut handles = vec![];
        for producer_id in 0..10 {
            let controller_clone = controller.clone();
            let handle = tokio::spawn(async move {
                for i in 0..1000 {
                    let value = producer_id * 1000 + i;
                    let _ = controller_clone.offer(value).await;
                }
            });
            handles.push(handle);
        }

        // Wait for all producers
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify stats
        let stats = controller.stats().await;
        assert_eq!(stats.events_received, 10000);
        assert!(stats.buffer_size <= 1000);
    }

    #[tokio::test]
    async fn test_stress_concurrent_offer_and_poll() {
        let config = BackpressureConfig {
            max_buffer_size: 500,
            strategy: BackpressureStrategy::Block,
            ..Default::default()
        };

        let controller = Arc::new(BackpressureController::new(config));

        // Spawn producer
        let producer_controller = controller.clone();
        let producer = tokio::spawn(async move {
            for i in 0..5000 {
                let _ = producer_controller.offer(i).await;
            }
        });

        // Spawn consumer
        let consumer_controller = controller.clone();
        let consumer = tokio::spawn(async move {
            let mut count = 0;
            loop {
                if let Ok(Some(_)) = consumer_controller.poll().await {
                    count += 1;
                    if count >= 5000 {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_micros(10)).await;
            }
            count
        });

        // Wait for both
        producer.await.unwrap();
        let consumed = consumer.await.unwrap();

        assert_eq!(consumed, 5000);

        // Verify stats
        let stats = controller.stats().await;
        assert_eq!(stats.events_received, 5000);
        assert_eq!(stats.events_processed, 5000);
    }

    // Degradation Strategy Tests
    #[tokio::test]
    async fn test_degradation_reduce_throughput() {
        let config = BackpressureConfig {
            max_buffer_size: 10,
            strategy: BackpressureStrategy::DropOldest,
            high_water_mark: 0.5, // Trigger degradation early
            degradation: DegradationStrategy::ReduceThroughput {
                reduction_percent: 50.0,
            },
            ..Default::default()
        };

        let controller = BackpressureController::new(config);

        // Fill buffer to trigger degradation
        for i in 0..20 {
            let _ = controller.offer(i).await;
        }

        let stats = controller.stats().await;
        // Some events should be dropped due to degradation
        assert!(stats.events_dropped > 0);
    }

    #[tokio::test]
    async fn test_degradation_sampling() {
        let config = BackpressureConfig {
            max_buffer_size: 10,
            strategy: BackpressureStrategy::DropOldest,
            high_water_mark: 0.5,
            degradation: DegradationStrategy::Sampling { sample_rate: 0.5 },
            ..Default::default()
        };

        let controller = BackpressureController::new(config);

        // Fill buffer
        for i in 0..20 {
            let _ = controller.offer(i).await;
        }

        let stats = controller.stats().await;
        // Roughly half should be sampled
        assert!(stats.events_received < 20);
    }

    // Metrics Tests
    #[tokio::test]
    async fn test_metrics_collection() {
        let config = BackpressureConfig {
            max_buffer_size: 100,
            strategy: BackpressureStrategy::Block,
            ..Default::default()
        };

        let controller = BackpressureController::new(config);

        // Verify initial metrics
        assert_eq!(controller.metrics_events_received.get(), 0);
        assert_eq!(controller.metrics_events_processed.get(), 0);

        // Offer and poll events
        for i in 0..10 {
            controller.offer(i).await.unwrap();
        }

        assert_eq!(controller.metrics_events_received.get(), 10);

        for _ in 0..5 {
            controller.poll().await.unwrap();
        }

        assert_eq!(controller.metrics_events_processed.get(), 5);
        assert_eq!(controller.metrics_queue_depth.get(), 5.0);
    }

    #[tokio::test]
    async fn test_metrics_latency() {
        let config = BackpressureConfig {
            max_buffer_size: 100,
            strategy: BackpressureStrategy::Block,
            ..Default::default()
        };

        let controller = BackpressureController::new(config);

        // Offer events
        for i in 0..10 {
            controller.offer(i).await.unwrap();
        }

        // Wait a bit to create measurable latency
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Poll events
        for _ in 0..10 {
            controller.poll().await.unwrap();
        }

        // Check latency histogram
        let stats = controller.metrics_latency.get_stats();
        assert!(stats.count == 10);
        assert!(stats.mean > 0.0);
    }

    #[tokio::test]
    async fn test_metrics_backpressure_events() {
        let config = BackpressureConfig {
            max_buffer_size: 100,
            strategy: BackpressureStrategy::DropOldest,
            high_water_mark: 0.5,
            ..Default::default()
        };

        let controller = BackpressureController::new(config);

        // Fill buffer to trigger backpressure
        for i in 0..60 {
            controller.offer(i).await.unwrap();
        }

        // Should have triggered backpressure events
        assert!(controller.metrics_backpressure_events.get() > 0);
    }
}
