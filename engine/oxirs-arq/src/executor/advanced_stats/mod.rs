//! Advanced query statistics and performance analytics

pub mod anomaly;
pub mod collector;
pub mod monitoring;
pub mod optimization_feedback;
pub mod pattern_analysis;
pub mod prediction;
pub mod resource_tracking;
pub mod streaming;
pub mod workload;

// Re-export main types
pub use collector::AdvancedStatisticsCollector;
pub use pattern_analysis::{
    AntiPatternDetector, CorrelationMatrix, PatternAnalyzer, PatternFrequency, ResourceImpact,
    SeasonalPatternDetector, TemporalCorrelation,
};
pub use prediction::{EnsemblePredictor, NeuralNetworkPredictor, PerformancePredictor};
pub use workload::{OptimizationStrategy, WorkloadClassifier};
pub use monitoring::{AlertSystem, RealTimeMonitor};
