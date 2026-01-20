//! Streaming analytics

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct ActiveAlert {
    pub rule_name: String,
    pub triggered_at: SystemTime,
    pub current_value: f64,
    pub threshold: f64,
    pub status: AlertStatus,
}

/// Alert status
#[derive(Debug, Clone)]
pub enum AlertStatus {
    Active,
    Resolved,
    Suppressed,
}

/// Escalation policy
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    pub name: String,
    pub escalation_levels: Vec<EscalationLevel>,
    pub auto_resolution: bool,
}

/// Escalation level
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    pub delay: Duration,
    pub actions: Vec<AlertAction>,
    pub conditions: Vec<EscalationCondition>,
}

/// Escalation conditions
#[derive(Debug, Clone)]
pub enum EscalationCondition {
    TimeElapsed(Duration),
    ValueStillAbove(f64),
    MultipleFailures(usize),
}

/// Dashboard metrics for visualization
#[derive(Debug, Clone)]
pub struct DashboardMetrics {
    pub time_series: HashMap<String, TimeSeries>,
    pub histograms: HashMap<String, Histogram>,
    pub counters: HashMap<String, Counter>,
    pub gauges: HashMap<String, Gauge>,
}

/// Time series data
#[derive(Debug, Clone)]
pub struct TimeSeries {
    pub points: VecDeque<TimeSeriesPoint>,
    pub retention_period: Duration,
    pub aggregation_interval: Duration,
}

/// Time series data point
#[derive(Debug, Clone)]
pub struct TimeSeriesPoint {
    pub timestamp: SystemTime,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

/// Histogram for distribution tracking
#[derive(Debug, Clone)]
pub struct Histogram {
    pub buckets: Vec<f64>,
    pub counts: Vec<usize>,
    pub sum: f64,
    pub count: usize,
}

/// Counter metric
#[derive(Debug, Clone)]
pub struct Counter {
    pub value: usize,
    pub rate: f64,
    pub labels: HashMap<String, String>,
}

/// Gauge metric
#[derive(Debug, Clone)]
pub struct Gauge {
    pub value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub trend: Trend,
}

/// Trend direction
#[derive(Debug, Clone)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
}

/// Streaming analytics engine
#[derive(Debug, Clone)]
pub struct StreamingAnalytics {
    pub sliding_windows: HashMap<String, SlidingWindow>,
    pub tumbling_windows: HashMap<String, TumblingWindow>,
    pub session_windows: HashMap<String, SessionWindow>,
}

/// Sliding window analytics
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    pub window_size: Duration,
    pub slide_interval: Duration,
    pub aggregation_function: AggregationFunction,
    pub current_value: f64,
}

/// Tumbling window analytics
#[derive(Debug, Clone)]
pub struct TumblingWindow {
    pub window_size: Duration,
    pub aggregation_function: AggregationFunction,
    pub windows: VecDeque<WindowResult>,
}

/// Session window analytics
#[derive(Debug, Clone)]
pub struct SessionWindow {
    pub session_timeout: Duration,
    pub aggregation_function: AggregationFunction,
    pub active_sessions: HashMap<String, SessionData>,
}

/// Session data tracking
#[derive(Debug, Clone)]
pub struct SessionData {
    pub session_id: String,
    pub start_time: SystemTime,
    pub last_activity: SystemTime,
    pub events: Vec<SessionEvent>,
}

/// Session event
#[derive(Debug, Clone)]
pub struct SessionEvent {
    pub timestamp: SystemTime,
    pub event_type: String,
    pub value: f64,
}

/// Window aggregation result
#[derive(Debug, Clone)]
pub struct WindowResult {
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub value: f64,
    pub count: usize,
}

/// Aggregation functions
#[derive(Debug, Clone)]
pub enum AggregationFunction {
    Sum,
    Average,
    Min,
    Max,
    Count,
    Median,
    Percentile(f64),
    StandardDeviation,
    Variance,
}

/// Anomaly detection system
#[derive(Debug, Clone)]
pub struct AnomalyDetector {
    /// Statistical anomaly detection
    statistical_detector: StatisticalAnomalyDetector,
    /// Machine learning anomaly detection
    ml_detector: MlAnomalyDetector,
    /// Rule-based anomaly detection
    rule_based_detector: RuleBasedAnomalyDetector,
    /// Ensemble anomaly detection
    ensemble_detector: EnsembleAnomalyDetector,
}

/// Statistical anomaly detection methods
#[derive(Debug, Clone)]
pub struct StatisticalAnomalyDetector {
    pub z_score_threshold: f64,
    pub iqr_multiplier: f64,
    pub moving_average_window: usize,
    pub seasonal_decomposition: bool,
}

/// Machine learning anomaly detection
#[derive(Debug, Clone)]
pub struct MlAnomalyDetector {
    pub isolation_forest: IsolationForest,
    pub one_class_svm: OneClassSvm,
    pub autoencoder: Autoencoder,
    pub lstm_detector: LstmDetector,
}

/// Isolation forest model
#[derive(Debug, Clone)]
pub struct IsolationForest {
    pub num_trees: usize,
    pub contamination_rate: f64,
    pub trees: Vec<IsolationTree>,
}

/// Isolation tree
#[derive(Debug, Clone)]
pub struct IsolationTree {
    pub depth: usize,
    pub splits: Vec<TreeSplit>,
}

/// Tree split node
#[derive(Debug, Clone)]
pub struct TreeSplit {
    pub feature_index: usize,
    pub split_value: f64,
    pub left_child: Option<Box<TreeSplit>>,
    pub right_child: Option<Box<TreeSplit>>,
}

/// One-class SVM model
#[derive(Debug, Clone)]
pub struct OneClassSvm {
    pub nu: f64,
    pub gamma: f64,
    pub support_vectors: Vec<Vec<f64>>,
    pub decision_function: Vec<f64>,
}

/// Autoencoder for anomaly detection
#[derive(Debug, Clone)]
pub struct Autoencoder {
    pub encoder_layers: Vec<NeuralLayer>,
    pub decoder_layers: Vec<NeuralLayer>,
    pub reconstruction_threshold: f64,
}

/// LSTM-based anomaly detector
#[derive(Debug, Clone)]
pub struct LstmDetector {
    pub lstm_layers: Vec<LstmLayer>,
    pub sequence_length: usize,
    pub prediction_threshold: f64,
}

/// LSTM layer
#[derive(Debug, Clone)]
pub struct LstmLayer {
    pub hidden_size: usize,
    pub cell_weights: Vec<Vec<f64>>,
    pub hidden_weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

/// Rule-based anomaly detection
#[derive(Debug, Clone)]
pub struct RuleBasedAnomalyDetector {
    pub rules: Vec<AnomalyRule>,
    pub rule_priorities: HashMap<String, i32>,
}

/// Anomaly detection rule
#[derive(Debug, Clone)]
pub struct AnomalyRule {
    pub name: String,
    pub condition: AnomalyCondition,
    pub severity: AnomalySeverity,
    pub description: String,
}

/// Anomaly conditions
#[derive(Debug, Clone)]
pub enum AnomalyCondition {
    ThresholdExceeded { metric: String, threshold: f64 },
    PatternDeviation { pattern: String, deviation: f64 },
    SequentialFailures { count: usize, window: Duration },
    ResourceExhaustion { resource: ResourceType, threshold: f64 },
}

/// Anomaly severity
#[derive(Debug, Clone)]
pub enum AnomalySeverity {
    Low,
    Medium,

impl DashboardMetrics {
    pub fn new() -> Self {
        Self {
            time_series: HashMap::new(),
            histograms: HashMap::new(),
            counters: HashMap::new(),
            gauges: HashMap::new(),
        }
    }
}

impl StreamingAnalytics {
    pub fn new() -> Self {
        Self {
            sliding_windows: HashMap::new(),
            tumbling_windows: HashMap::new(),
            session_windows: HashMap::new(),
        }
    }
}

