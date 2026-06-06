//! Performance Monitoring for SHACL-AI
//!
//! Thin facade re-exporting from sibling modules.

pub use crate::performance_monitoring_advanced::{
    AdaptiveLearningOptimizer, AnnealingParameters, ConfidenceInterval, ConstraintType,
    ForecastPoint, MultiObjectiveAlgorithm, MultiObjectiveOptimizer, MultiObjectiveSolution,
    ObjectivePriority, OptimizationConstraint, OptimizationExperience, OptimizationObjective,
    OptimizationPattern, OptimizationState, ParetoSolution, PatternCondition, PerformanceContext,
    PredictionAccuracy, PredictiveAutoScaler, PredictorModelType, PreferenceType, QuantumGate,
    QuantumGateType, QuantumInspiredOptimizer, QuantumMeasurement, QuantumState, ResourceForecast,
    ResourceType, ScalingAction, ScalingEvent, ScalingOutcome, ScalingPolicy, TradeOffPreferences,
    WorkloadDataPoint, WorkloadPredictor,
};

pub use crate::performance_monitoring_types::{
    AcknowledgmentStatus, ActiveAlert, AggregationMethod, AlertCondition, AlertRule, AlertSeverity,
    AnomalyAlgorithmType, AnomalyDetectionAlgorithm, AnomalyEvent, AnomalySeverity, BaselineModel,
    BottleneckAnalysis, CapacityForecast, ComparisonOperator, ConcurrencyIssue, ConstraintHotspot,
    CurrentMetrics, CustomChart, DashboardWidget, EscalationPolicy, EscalationStep, ExportFormat,
    ExportMetadata, GcActivity, ImpactAssessment, ImplementationEffort, IoBottleneck,
    MonitoringConfig, MonitoringStatistics, NotificationChannel, NotificationChannelType,
    OptimizationAction, OptimizationOpportunity, OptimizationRecommendation, OptimizationRule,
    OptimizationRuleType, PerformanceAnomalyType, PerformanceDataExport, PerformanceMetric,
    PerformanceSnapshot, PerformanceTrend, ResolutionStatus, ResourceConsumption, ResourceUsage,
    RiskLevel, ShapePerformanceMetric, SystemMetric, TimeRange, TrendDirection, TrendModel,
    TrendModelType, TrendSummary, TriggerCondition, ValidationMetric, VisualizationConfig,
    VisualizationThreshold, WidgetType,
};

pub use crate::performance_monitoring_collector::MetricsCollector;

pub use crate::performance_monitoring_reporter::{
    export_to_csv, export_to_xml, AlertSystem, AnalysisModels, AnomalyDetector,
    EffectivenessTracker, ForecastModel, ForecastingEngine, MonitoringDashboard,
    PerformanceOptimizationEngine, PerformancePredictors, RealTimeAnalyzer, ReportGenerator,
    ReportSection, ReportTemplate, ReportType, ScheduledReport, SeasonalAnalyzer,
    SeasonalPatternType, SuccessMetrics, TrendAnalyzer,
};

use crate::{Result, ShaclAiError};
use std::time::Duration;

/// Real-time performance monitoring system
#[derive(Debug)]
pub struct PerformanceMonitor {
    pub config: MonitoringConfig,
    pub metrics_collector: MetricsCollector,
    pub real_time_analyzer: RealTimeAnalyzer,
    pub optimization_engine: PerformanceOptimizationEngine,
    pub alert_system: AlertSystem,
    pub monitoring_dashboard: MonitoringDashboard,
    pub statistics: MonitoringStatistics,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new() -> Self {
        Self::with_config(MonitoringConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics_collector: MetricsCollector::new(),
            real_time_analyzer: RealTimeAnalyzer::new(),
            optimization_engine: PerformanceOptimizationEngine::new(),
            alert_system: AlertSystem::new(),
            monitoring_dashboard: MonitoringDashboard::new(),
            statistics: MonitoringStatistics::default(),
        }
    }

    /// Start real-time monitoring
    pub fn start_monitoring(&mut self) -> Result<()> {
        if !self.config.enable_realtime_monitoring {
            return Err(ShaclAiError::Configuration(
                "Real-time monitoring is disabled".to_string(),
            ));
        }
        tracing::info!("Starting real-time performance monitoring");
        self.metrics_collector
            .start_collection(self.config.monitoring_interval_ms)?;
        self.real_time_analyzer.initialize_models()?;
        self.alert_system.start_monitoring()?;
        tracing::info!("Performance monitoring started successfully");
        Ok(())
    }

    /// Stop monitoring
    pub fn stop_monitoring(&mut self) -> Result<()> {
        tracing::info!("Stopping performance monitoring");
        self.metrics_collector.stop_collection()?;
        self.alert_system.stop_monitoring()?;
        tracing::info!("Performance monitoring stopped");
        Ok(())
    }

    /// Record validation performance
    pub fn record_validation_performance(
        &mut self,
        shape_id: String,
        duration: Duration,
        success: bool,
        resource_usage: ResourceUsage,
    ) -> Result<()> {
        let metric = ValidationMetric {
            timestamp: chrono::Utc::now(),
            shape_id: shape_id.clone(),
            validation_duration_ms: duration.as_millis() as f64,
            constraint_evaluation_count: 0,
            data_size_triples: 0,
            success,
            error_type: if success {
                None
            } else {
                Some("ValidationError".to_string())
            },
            optimization_applied: false,
            cache_utilized: false,
            parallel_execution: false,
            resource_usage,
        };

        self.metrics_collector.add_validation_metric(metric)?;
        self.update_shape_metrics(&shape_id, duration, success)?;

        if self.config.enable_predictive_analytics {
            self.real_time_analyzer
                .detect_anomalies(&shape_id, duration.as_millis() as f64)?;
        }

        Ok(())
    }

    /// Generate optimization recommendations
    pub fn generate_optimization_recommendations(
        &mut self,
    ) -> Result<Vec<OptimizationRecommendation>> {
        if !self.config.enable_optimization_recommendations {
            return Ok(vec![]);
        }
        tracing::info!("Generating performance optimization recommendations");
        let current_metrics = self.metrics_collector.get_current_metrics()?;
        let recommendations = self
            .optimization_engine
            .analyze_and_recommend(&current_metrics)?;
        self.statistics.total_optimizations_recommended += recommendations.len();
        tracing::info!(
            "Generated {} optimization recommendations",
            recommendations.len()
        );
        Ok(recommendations)
    }

    /// Get current performance snapshot
    pub fn get_performance_snapshot(&self) -> Result<PerformanceSnapshot> {
        let current_metrics = self.metrics_collector.get_current_metrics()?;
        let active_alerts = self.alert_system.get_active_alerts();
        let anomalies = self
            .real_time_analyzer
            .get_recent_anomalies(Duration::from_secs(3600))?;

        Ok(PerformanceSnapshot {
            timestamp: chrono::Utc::now(),
            overall_health_score: self.calculate_overall_health_score(),
            current_metrics,
            active_alerts,
            recent_anomalies: anomalies,
            trend_analysis: self.real_time_analyzer.get_trend_summary()?,
            optimization_opportunities: self.optimization_engine.get_current_opportunities()?,
        })
    }

    /// Get monitoring statistics
    pub fn get_statistics(&self) -> &MonitoringStatistics {
        &self.statistics
    }

    /// Export performance data
    pub fn export_performance_data(
        &self,
        time_range: TimeRange,
        metrics: Vec<String>,
        format: ExportFormat,
    ) -> Result<String> {
        tracing::info!("Exporting performance data for time range {:?}", time_range);
        let data = self
            .metrics_collector
            .get_metrics_for_range(&time_range, &metrics)?;
        let exported_data = match format {
            ExportFormat::JSON => serde_json::to_string_pretty(&data)?,
            ExportFormat::CSV => export_to_csv(&data)?,
            ExportFormat::XML => export_to_xml(&data)?,
        };
        Ok(exported_data)
    }

    fn update_shape_metrics(
        &mut self,
        _shape_id: &str,
        _duration: Duration,
        _success: bool,
    ) -> Result<()> {
        Ok(())
    }

    fn calculate_overall_health_score(&self) -> f64 {
        0.85
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}
