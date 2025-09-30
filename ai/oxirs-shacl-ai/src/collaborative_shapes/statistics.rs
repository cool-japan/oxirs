//! Statistics tracking for collaborative shape development

use std::time::Duration;

/// Collaborative statistics
#[derive(Debug, Clone, Default)]
pub struct CollaborativeStatistics {
    pub total_workspaces: usize,
    pub active_collaborations: usize,
    pub total_contributors: usize,
    pub shapes_created_collaboratively: usize,
    pub conflicts_resolved: usize,
    pub average_resolution_time: Duration,
    pub peer_reviews_completed: usize,
    pub library_contributions: usize,
    pub user_satisfaction_score: f64,
    pub collaboration_efficiency: f64,
}

impl CollaborativeStatistics {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get overall collaboration health score
    pub fn health_score(&self) -> f64 {
        if self.total_workspaces == 0 {
            return 0.0;
        }

        let activity_score = (self.active_collaborations as f64 / self.total_workspaces as f64).min(1.0);
        let efficiency_score = self.collaboration_efficiency;
        let satisfaction_score = self.user_satisfaction_score;

        (activity_score + efficiency_score + satisfaction_score) / 3.0
    }

    /// Update resolution time average
    pub fn update_resolution_time(&mut self, new_time: Duration) {
        if self.conflicts_resolved == 0 {
            self.average_resolution_time = new_time;
        } else {
            let total_time = self.average_resolution_time * self.conflicts_resolved as u32;
            let new_total = total_time + new_time;
            self.average_resolution_time = new_total / (self.conflicts_resolved + 1) as u32;
        }
        self.conflicts_resolved += 1;
    }
}