//! Metrics collection for compaction system

use super::types::{CompactionResult, CompactionState, CompactionStatistics};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Compaction metrics collector
#[derive(Debug, Clone)]
pub struct CompactionMetrics {
    /// Current state
    state: Arc<Mutex<CompactionState>>,
    /// Statistics
    statistics: Arc<Mutex<CompactionStatistics>>,
    /// Recent compaction history
    history: Arc<Mutex<VecDeque<CompactionResult>>>,
    /// Maximum history size
    max_history_size: usize,
}

impl Default for CompactionMetrics {
    fn default() -> Self {
        Self::new(100)
    }
}

impl CompactionMetrics {
    /// Create a new metrics collector
    pub fn new(max_history_size: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(CompactionState::Idle)),
            statistics: Arc::new(Mutex::new(CompactionStatistics::default())),
            history: Arc::new(Mutex::new(VecDeque::new())),
            max_history_size,
        }
    }

    /// Update state
    pub fn update_state(&self, state: CompactionState) {
        let mut s = self.state.lock().unwrap();
        *s = state;
    }

    /// Get current state
    pub fn get_state(&self) -> CompactionState {
        *self.state.lock().unwrap()
    }

    /// Record compaction result
    pub fn record_compaction(&self, result: CompactionResult) {
        let mut stats = self.statistics.lock().unwrap();
        let mut history = self.history.lock().unwrap();

        // Update statistics
        stats.total_compactions += 1;
        if result.success {
            stats.successful_compactions += 1;
        } else {
            stats.failed_compactions += 1;
        }

        stats.total_vectors_processed += result.vectors_processed;
        stats.total_vectors_removed += result.vectors_removed;
        stats.total_bytes_reclaimed += result.bytes_reclaimed;
        stats.current_fragmentation = result.fragmentation_after;
        stats.last_compaction_time = Some(result.end_time);
        stats.last_compaction_result = Some(result.clone());

        // Update average duration
        if stats.total_compactions > 0 {
            let total_duration = stats.avg_compaction_duration.as_secs_f64()
                * (stats.total_compactions - 1) as f64
                + result.duration.as_secs_f64();
            stats.avg_compaction_duration =
                Duration::from_secs_f64(total_duration / stats.total_compactions as f64);
        } else {
            stats.avg_compaction_duration = result.duration;
        }

        // Add to history
        history.push_back(result);
        while history.len() > self.max_history_size {
            history.pop_front();
        }
    }

    /// Update fragmentation
    pub fn update_fragmentation(&self, fragmentation: f64) {
        let mut stats = self.statistics.lock().unwrap();
        stats.current_fragmentation = fragmentation;
    }

    /// Get statistics
    pub fn get_statistics(&self) -> CompactionStatistics {
        self.statistics.lock().unwrap().clone()
    }

    /// Get compaction history
    pub fn get_history(&self, limit: Option<usize>) -> Vec<CompactionResult> {
        let history = self.history.lock().unwrap();
        if let Some(lim) = limit {
            history.iter().rev().take(lim).cloned().collect()
        } else {
            history.iter().cloned().collect()
        }
    }

    /// Calculate compaction efficiency
    pub fn calculate_efficiency(&self) -> CompactionEfficiency {
        let stats = self.statistics.lock().unwrap();

        let success_rate = if stats.total_compactions > 0 {
            stats.successful_compactions as f64 / stats.total_compactions as f64
        } else {
            0.0
        };

        let avg_space_reclaimed = if stats.successful_compactions > 0 {
            stats.total_bytes_reclaimed as f64 / stats.successful_compactions as f64
        } else {
            0.0
        };

        let avg_vectors_removed = if stats.successful_compactions > 0 {
            stats.total_vectors_removed as f64 / stats.successful_compactions as f64
        } else {
            0.0
        };

        CompactionEfficiency {
            success_rate,
            avg_space_reclaimed_bytes: avg_space_reclaimed as u64,
            avg_vectors_removed: avg_vectors_removed as usize,
            avg_duration: stats.avg_compaction_duration,
            current_fragmentation: stats.current_fragmentation,
        }
    }

    /// Reset metrics
    pub fn reset(&self) {
        let mut stats = self.statistics.lock().unwrap();
        *stats = CompactionStatistics::default();

        let mut history = self.history.lock().unwrap();
        history.clear();

        let mut state = self.state.lock().unwrap();
        *state = CompactionState::Idle;
    }
}

/// Compaction efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionEfficiency {
    /// Success rate (0.0 - 1.0)
    pub success_rate: f64,
    /// Average space reclaimed per compaction
    pub avg_space_reclaimed_bytes: u64,
    /// Average vectors removed per compaction
    pub avg_vectors_removed: usize,
    /// Average duration
    pub avg_duration: Duration,
    /// Current fragmentation
    pub current_fragmentation: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compaction::types::CompactionResult;
    use std::time::SystemTime;

    fn create_test_result(success: bool, bytes_reclaimed: u64) -> CompactionResult {
        CompactionResult {
            start_time: SystemTime::now(),
            end_time: SystemTime::now(),
            duration: Duration::from_secs(10),
            vectors_processed: 1000,
            vectors_removed: 100,
            bytes_reclaimed,
            fragmentation_before: 0.4,
            fragmentation_after: 0.1,
            success,
            error: None,
        }
    }

    #[test]
    fn test_metrics_recording() {
        let metrics = CompactionMetrics::new(10);

        let result = create_test_result(true, 1_000_000);
        metrics.record_compaction(result);

        let stats = metrics.get_statistics();
        assert_eq!(stats.total_compactions, 1);
        assert_eq!(stats.successful_compactions, 1);
        assert_eq!(stats.total_bytes_reclaimed, 1_000_000);
    }

    #[test]
    fn test_efficiency_calculation() {
        let metrics = CompactionMetrics::new(10);

        metrics.record_compaction(create_test_result(true, 1_000_000));
        metrics.record_compaction(create_test_result(true, 2_000_000));
        metrics.record_compaction(create_test_result(false, 0));

        let efficiency = metrics.calculate_efficiency();
        assert!((efficiency.success_rate - 0.666).abs() < 0.01);
        assert_eq!(efficiency.avg_space_reclaimed_bytes, 1_500_000);
    }

    #[test]
    fn test_history_limit() {
        let metrics = CompactionMetrics::new(5);

        for i in 0..10 {
            metrics.record_compaction(create_test_result(true, i * 1000));
        }

        let history = metrics.get_history(None);
        assert_eq!(history.len(), 5);
    }

    #[test]
    fn test_state_updates() {
        let metrics = CompactionMetrics::new(10);

        assert_eq!(metrics.get_state(), CompactionState::Idle);

        metrics.update_state(CompactionState::Running);
        assert_eq!(metrics.get_state(), CompactionState::Running);

        metrics.update_state(CompactionState::Completed);
        assert_eq!(metrics.get_state(), CompactionState::Completed);
    }
}
