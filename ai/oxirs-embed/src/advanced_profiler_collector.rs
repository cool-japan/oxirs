//! Advanced Profiler Collector
//!
//! Event collection: instrumentation points, sampling strategies,
//! buffer management, and performance tracker lifecycle.

use std::time::Instant;

use super::advanced_profiler_types::{
    CollectionStats, MetricDataPoint, PerformanceCollector, PerformanceTracker, TrackerState,
};

impl Default for PerformanceCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceCollector {
    /// Create a new performance collector
    pub fn new() -> Self {
        Self {
            buffer: std::collections::VecDeque::new(),
            stats: CollectionStats::default(),
            trackers: std::collections::HashMap::new(),
        }
    }

    /// Add a metric to the ring buffer (drops oldest entry when full)
    pub fn add_metric(&mut self, metric: MetricDataPoint) {
        if self.buffer.len() >= 100_000 {
            self.buffer.pop_front();
            self.stats.drop_rate += 1.0;
        }

        self.buffer.push_back(metric);
        self.stats.total_points += 1;
        self.stats.memory_usage_bytes =
            (self.buffer.len() * std::mem::size_of::<MetricDataPoint>()) as u64;
    }

    /// Start a named performance tracker
    pub fn start_tracker(&mut self, name: String) -> String {
        let tracker = PerformanceTracker {
            name: name.clone(),
            start_time: Instant::now(),
            measurements: Vec::new(),
            state: TrackerState::Active,
        };

        self.trackers.insert(name.clone(), tracker);
        name
    }

    /// Stop a named performance tracker and return the completed tracker
    pub fn stop_tracker(&mut self, name: &str) -> Option<PerformanceTracker> {
        if let Some(mut tracker) = self.trackers.remove(name) {
            tracker.state = TrackerState::Stopped;
            Some(tracker)
        } else {
            None
        }
    }
}
