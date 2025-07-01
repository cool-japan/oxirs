//! Search history tracking and statistics

use crate::neural_architecture_search::architecture::Architecture;
use std::collections::HashMap;
use uuid::Uuid;

/// Search history tracking
#[derive(Debug, Clone, Default)]
pub struct SearchHistory {
    /// All evaluated architectures
    pub architectures: HashMap<Uuid, Architecture>,
    /// Performance timeline
    pub performance_timeline: Vec<(usize, f64)>, // (generation, best_performance)
    /// Search statistics
    pub search_stats: SearchStatistics,
    /// Best architecture found so far
    pub best_architecture: Option<Uuid>,
    /// Pareto front for multi-objective optimization
    pub pareto_front: Vec<Uuid>,
}

/// Search statistics
#[derive(Debug, Clone, Default)]
pub struct SearchStatistics {
    /// Total architectures evaluated
    pub total_evaluations: usize,
    /// Total search time
    pub total_search_time_minutes: f64,
    /// Average evaluation time
    pub avg_evaluation_time_minutes: f64,
    /// Convergence information
    pub convergence_generation: Option<usize>,
    /// Diversity metrics
    pub diversity_metrics: DiversityMetrics,
    /// Resource usage statistics
    pub resource_usage: ResourceUsageStats,
}

/// Diversity metrics for population
#[derive(Debug, Clone, Default)]
pub struct DiversityMetrics {
    /// Architectural diversity score
    pub architectural_diversity: f64,
    /// Performance diversity score
    pub performance_diversity: f64,
    /// Hyperparameter diversity score
    pub hyperparameter_diversity: f64,
    /// Novelty score
    pub novelty_score: f64,
}

/// Resource usage statistics
#[derive(Debug, Clone, Default)]
pub struct ResourceUsageStats {
    /// Total CPU hours used
    pub total_cpu_hours: f64,
    /// Total GPU hours used
    pub total_gpu_hours: f64,
    /// Peak memory usage in GB
    pub peak_memory_gb: f64,
    /// Total disk I/O in GB
    pub total_disk_io_gb: f64,
}

impl SearchHistory {
    /// Create a new search history
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an architecture to the history
    pub fn add_architecture(&mut self, architecture: Architecture) {
        self.architectures.insert(architecture.id, architecture);
        self.search_stats.total_evaluations += 1;
    }

    /// Update the best architecture
    pub fn update_best(&mut self, architecture_id: Uuid, performance: f64, generation: usize) {
        self.best_architecture = Some(architecture_id);
        self.performance_timeline.push((generation, performance));
    }

    /// Get the best architecture
    pub fn get_best_architecture(&self) -> Option<&Architecture> {
        self.best_architecture
            .and_then(|id| self.architectures.get(&id))
    }

    /// Update search statistics
    pub fn update_statistics(&mut self, generation: usize, evaluation_time: f64) {
        self.search_stats.avg_evaluation_time_minutes = 
            (self.search_stats.avg_evaluation_time_minutes * (self.search_stats.total_evaluations - 1) as f64 + evaluation_time) 
            / self.search_stats.total_evaluations as f64;
    }

    /// Check if search has converged
    pub fn has_converged(&self, patience: usize, tolerance: f64) -> bool {
        if self.performance_timeline.len() < patience {
            return false;
        }

        let recent_performances: Vec<f64> = self.performance_timeline
            .iter()
            .rev()
            .take(patience)
            .map(|(_, perf)| *perf)
            .collect();

        let max_perf = recent_performances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_perf = recent_performances.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        (max_perf - min_perf) < tolerance
    }
}

impl SearchStatistics {
    /// Update diversity metrics
    pub fn calculate_diversity(&mut self, architectures: &[Architecture]) {
        // Placeholder implementation for diversity calculation
        self.diversity_metrics.architectural_diversity = self.calculate_architectural_diversity(architectures);
        self.diversity_metrics.performance_diversity = self.calculate_performance_diversity(architectures);
        self.diversity_metrics.hyperparameter_diversity = self.calculate_hyperparameter_diversity(architectures);
        self.diversity_metrics.novelty_score = self.calculate_novelty_score(architectures);
    }

    fn calculate_architectural_diversity(&self, _architectures: &[Architecture]) -> f64 {
        // Placeholder: calculate diversity based on layer types, depths, etc.
        0.5
    }

    fn calculate_performance_diversity(&self, architectures: &[Architecture]) -> f64 {
        if architectures.len() < 2 {
            return 0.0;
        }

        let performances: Vec<f64> = architectures
            .iter()
            .filter_map(|arch| arch.performance.as_ref().map(|p| p.composite_score))
            .collect();

        if performances.len() < 2 {
            return 0.0;
        }

        let mean = performances.iter().sum::<f64>() / performances.len() as f64;
        let variance = performances.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / performances.len() as f64;

        variance.sqrt()
    }

    fn calculate_hyperparameter_diversity(&self, _architectures: &[Architecture]) -> f64 {
        // Placeholder: calculate diversity based on hyperparameter values
        0.3
    }

    fn calculate_novelty_score(&self, _architectures: &[Architecture]) -> f64 {
        // Placeholder: calculate novelty based on architecture uniqueness
        0.4
    }
}