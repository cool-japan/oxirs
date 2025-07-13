//! Main event processor implementation
//!
//! This module provides the core event processing functionality including:
//! - Event processor management
//! - Window lifecycle management
//! - Watermark handling
//! - Late event processing

use super::{
    aggregation::AggregationManager,
    window::{EventWindow, Watermark, WindowConfig, WindowResult},
};
use crate::StreamEvent;
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tracing::{debug, info, warn};

/// Processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Maximum number of windows to maintain
    pub max_windows: usize,
    /// Maximum late event buffer size
    pub max_late_events: usize,
    /// Watermark advancement interval
    pub watermark_interval: ChronoDuration,
    /// Enable statistics collection
    pub enable_stats: bool,
    /// Memory limit for event storage (bytes)
    pub memory_limit: Option<usize>,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            max_windows: 1000,
            max_late_events: 10000,
            watermark_interval: ChronoDuration::seconds(1),
            enable_stats: true,
            memory_limit: Some(1024 * 1024 * 100), // 100MB
        }
    }
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorStats {
    /// Total events processed
    pub events_processed: u64,
    /// Total windows created
    pub windows_created: u64,
    /// Total windows triggered
    pub windows_triggered: u64,
    /// Late events received
    pub late_events: u64,
    /// Dropped events (due to memory limits)
    pub dropped_events: u64,
    /// Processing start time
    pub start_time: DateTime<Utc>,
    /// Last processing time
    pub last_processing_time: DateTime<Utc>,
    /// Average processing latency (milliseconds)
    pub avg_latency_ms: f64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
}

impl Default for ProcessorStats {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            events_processed: 0,
            windows_created: 0,
            windows_triggered: 0,
            late_events: 0,
            dropped_events: 0,
            start_time: now,
            last_processing_time: now,
            avg_latency_ms: 0.0,
            peak_memory_usage: 0,
        }
    }
}

/// Advanced event processor with windowing and aggregations
pub struct EventProcessor {
    windows: HashMap<String, EventWindow>,
    watermark: DateTime<Utc>,
    late_events: VecDeque<(StreamEvent, DateTime<Utc>)>,
    stats: ProcessorStats,
    config: ProcessorConfig,
    watermark_manager: Watermark,
    aggregation_manager: AggregationManager,
}

impl EventProcessor {
    /// Create a new event processor
    pub fn new(config: ProcessorConfig) -> Self {
        Self {
            windows: HashMap::new(),
            watermark: Utc::now(),
            late_events: VecDeque::new(),
            stats: ProcessorStats::default(),
            config,
            watermark_manager: Watermark::default(),
            aggregation_manager: AggregationManager::default(),
        }
    }

    /// Create a new window with the given configuration
    pub fn create_window(&mut self, config: WindowConfig) -> Result<String> {
        let window = EventWindow::new(config);
        let window_id = window.id().to_string();

        // Check memory limits
        if let Some(limit) = self.config.memory_limit {
            if self.estimate_memory_usage() > limit {
                return Err(anyhow!("Memory limit exceeded, cannot create new window"));
            }
        }

        // Check window count limits
        if self.windows.len() >= self.config.max_windows {
            warn!("Maximum number of windows reached, removing oldest window");
            self.remove_oldest_window();
        }

        self.windows.insert(window_id.clone(), window);
        self.stats.windows_created += 1;

        info!("Created new window: {}", window_id);
        Ok(window_id)
    }

    /// Process an event through all windows
    pub fn process_event(&mut self, event: StreamEvent) -> Result<Vec<WindowResult>> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();

        // Update watermark
        self.update_watermark(&event)?;

        // Check if event is late
        if self.is_late_event(&event) {
            self.handle_late_event(event)?;
            return Ok(results);
        }

        // Process event through all windows
        let mut windows_to_trigger = Vec::new();

        for (window_id, window) in &mut self.windows {
            if let Err(e) = window.add_event(event.clone()) {
                warn!("Failed to add event to window {}: {}", window_id, e);
                continue;
            }

            // Check if window should trigger
            if window.should_trigger(self.watermark) {
                windows_to_trigger.push(window_id.clone());
            }
        }

        // Trigger windows that need to be triggered
        for window_id in windows_to_trigger {
            let result = self.trigger_window(&window_id)?;
            results.push(result);
        }

        // Update statistics
        self.update_stats(start_time);

        Ok(results)
    }

    /// Trigger a window and produce results
    fn trigger_window(&mut self, window_id: &str) -> Result<WindowResult> {
        let window = self
            .windows
            .get(window_id)
            .ok_or_else(|| anyhow!("Window not found: {}", window_id))?;

        // Calculate aggregations
        let aggregations = self.aggregation_manager.results()?;

        let result = WindowResult {
            window_id: window_id.to_string(),
            window_start: window
                .config()
                .window_type
                .start_time()
                .unwrap_or(Utc::now()),
            window_end: Utc::now(),
            event_count: window.event_count(),
            aggregations,
            trigger_reason: "Window trigger condition met".to_string(),
            processing_time: Utc::now(),
        };

        self.stats.windows_triggered += 1;
        info!("Triggered window: {}", window_id);

        Ok(result)
    }

    /// Update watermark based on event
    fn update_watermark(&mut self, event: &StreamEvent) -> Result<()> {
        let event_time = event.timestamp();
        self.watermark_manager.update(event_time);
        self.watermark = self.watermark_manager.current();
        Ok(())
    }

    /// Check if event is late
    fn is_late_event(&self, event: &StreamEvent) -> bool {
        let event_time = event.timestamp();
        let allowed_lateness = self.watermark_manager.allowed_lateness;
        event_time < self.watermark - allowed_lateness
    }

    /// Handle late events
    fn handle_late_event(&mut self, event: StreamEvent) -> Result<()> {
        if self.late_events.len() >= self.config.max_late_events {
            self.late_events.pop_front();
            self.stats.dropped_events += 1;
        }

        self.late_events.push_back((event, Utc::now()));
        self.stats.late_events += 1;

        Ok(())
    }

    /// Remove oldest window to make room for new ones
    fn remove_oldest_window(&mut self) {
        if let Some((oldest_id, _)) = self.windows.iter().min_by_key(|(_, window)| {
            window
                .config()
                .window_type
                .start_time()
                .unwrap_or(Utc::now())
        }) {
            let oldest_id = oldest_id.clone();
            self.windows.remove(&oldest_id);
            debug!("Removed oldest window: {}", oldest_id);
        }
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        // Rough estimation of memory usage
        let window_size = std::mem::size_of::<EventWindow>();
        let late_event_size = std::mem::size_of::<(StreamEvent, DateTime<Utc>)>();

        self.windows.len() * window_size + self.late_events.len() * late_event_size
    }

    /// Update processing statistics
    fn update_stats(&mut self, start_time: std::time::Instant) {
        self.stats.events_processed += 1;
        self.stats.last_processing_time = Utc::now();

        let elapsed = start_time.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;

        // Update average latency (exponential moving average)
        let alpha = 0.1;
        self.stats.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * self.stats.avg_latency_ms;

        // Update peak memory usage
        let current_memory = self.estimate_memory_usage();
        if current_memory > self.stats.peak_memory_usage {
            self.stats.peak_memory_usage = current_memory;
        }
    }

    /// Get processing statistics
    pub fn stats(&self) -> &ProcessorStats {
        &self.stats
    }

    /// Get active windows
    pub fn active_windows(&self) -> Vec<String> {
        self.windows.keys().cloned().collect()
    }

    /// Get window by ID
    pub fn get_window(&self, window_id: &str) -> Option<&EventWindow> {
        self.windows.get(window_id)
    }

    /// Remove window by ID
    pub fn remove_window(&mut self, window_id: &str) -> Result<()> {
        if self.windows.remove(window_id).is_some() {
            info!("Removed window: {}", window_id);
            Ok(())
        } else {
            Err(anyhow!("Window not found: {}", window_id))
        }
    }

    /// Clear all windows
    pub fn clear_windows(&mut self) {
        self.windows.clear();
        info!("Cleared all windows");
    }

    /// Get current watermark
    pub fn current_watermark(&self) -> DateTime<Utc> {
        self.watermark
    }

    /// Get late events
    pub fn late_events(&self) -> &VecDeque<(StreamEvent, DateTime<Utc>)> {
        &self.late_events
    }
}

impl Default for EventProcessor {
    fn default() -> Self {
        Self::new(ProcessorConfig::default())
    }
}

// Helper trait for window types
trait WindowTypeExt {
    fn start_time(&self) -> Option<DateTime<Utc>>;
}

impl WindowTypeExt for super::window::WindowType {
    fn start_time(&self) -> Option<DateTime<Utc>> {
        // This would need to be implemented based on the actual window type
        Some(Utc::now())
    }
}
