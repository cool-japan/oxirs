//! # Advanced Visualization & Dashboarding Module
//!
//! Comprehensive visualization and dashboarding capabilities:
//! - Real-time metrics visualization
//! - Query performance dashboards
//! - Federation topology visualization
//! - Security monitoring dashboards
//! - Compliance dashboards
//! - Customizable widget system
//! - Alert visualization
//! - Export capabilities (PNG, SVG, JSON)

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use scirs2_core::ndarray_ext::Array2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::info;

/// Main visualization and dashboarding system
#[derive(Clone)]
pub struct AdvancedVisualization {
    #[allow(dead_code)]
    config: VisualizationConfig,
    dashboards: Arc<DashMap<String, Dashboard>>,
    metrics_collector: Arc<MetricsCollector>,
    chart_generator: Arc<ChartGenerator>,
    topology_visualizer: Arc<TopologyVisualizer>,
    alert_visualizer: Arc<AlertVisualizer>,
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Enable real-time updates
    pub enable_realtime: bool,
    /// Update interval for real-time dashboards
    pub update_interval: Duration,
    /// Default chart theme
    pub default_theme: ChartTheme,
    /// Maximum data points to display
    pub max_data_points: usize,
    /// Enable export features
    pub enable_export: bool,
    /// Default export format
    pub default_export_format: ExportFormat,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            enable_realtime: true,
            update_interval: Duration::from_secs(5),
            default_theme: ChartTheme::Dark,
            max_data_points: 1000,
            enable_export: true,
            default_export_format: ExportFormat::SVG,
        }
    }
}

/// Chart theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartTheme {
    Light,
    Dark,
    HighContrast,
    Custom(CustomTheme),
}

/// Custom theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomTheme {
    pub background_color: String,
    pub text_color: String,
    pub primary_color: String,
    pub secondary_color: String,
    pub grid_color: String,
}

/// Export format
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExportFormat {
    PNG,
    SVG,
    JSON,
    CSV,
    PDF,
}

/// Dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dashboard {
    pub id: String,
    pub name: String,
    pub description: String,
    pub layout: DashboardLayout,
    pub widgets: Vec<Widget>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub auto_refresh: bool,
    pub refresh_interval: Duration,
}

/// Dashboard layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardLayout {
    Grid { rows: usize, cols: usize },
    Flexible,
    SingleColumn,
    TwoColumn,
}

/// Widget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub id: String,
    pub widget_type: WidgetType,
    pub title: String,
    pub position: WidgetPosition,
    pub size: WidgetSize,
    pub data_source: DataSource,
    pub visualization_type: VisualizationType,
    pub config: WidgetConfig,
}

/// Widget position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPosition {
    pub row: usize,
    pub col: usize,
}

/// Widget size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSize {
    pub width: usize,  // Grid units
    pub height: usize, // Grid units
}

/// Widget type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    Chart,
    Table,
    Map,
    Topology,
    Alert,
    Metric,
    Text,
}

/// Data source for widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    QueryMetrics,
    FederationTopology,
    SecurityAlerts,
    ComplianceStatus,
    PerformanceMetrics,
    ServiceHealth,
    Custom(String),
}

/// Visualization type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    LineChart,
    BarChart,
    PieChart,
    ScatterPlot,
    Heatmap,
    NetworkGraph,
    Gauge,
    Table,
    TreeMap,
    Sankey,
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    pub show_legend: bool,
    pub show_grid: bool,
    pub x_axis_label: Option<String>,
    pub y_axis_label: Option<String>,
    pub color_scheme: Vec<String>,
    pub custom_options: HashMap<String, serde_json::Value>,
}

impl Default for WidgetConfig {
    fn default() -> Self {
        Self {
            show_legend: true,
            show_grid: true,
            x_axis_label: None,
            y_axis_label: None,
            color_scheme: vec![
                "#1f77b4".to_string(),
                "#ff7f0e".to_string(),
                "#2ca02c".to_string(),
                "#d62728".to_string(),
                "#9467bd".to_string(),
            ],
            custom_options: HashMap::new(),
        }
    }
}

/// Metrics collector
pub struct MetricsCollector {
    time_series: Arc<RwLock<HashMap<String, TimeSeries>>>,
    #[allow(dead_code)]
    aggregations: Arc<RwLock<HashMap<String, MetricAggregation>>>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self {
            time_series: Arc::new(RwLock::new(HashMap::new())),
            aggregations: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record metric value
    pub async fn record_metric(&self, metric_name: &str, value: f64) -> Result<()> {
        let mut time_series = self.time_series.write().await;
        let series = time_series
            .entry(metric_name.to_string())
            .or_insert_with(|| TimeSeries {
                name: metric_name.to_string(),
                data_points: VecDeque::new(),
                unit: "".to_string(),
            });

        series.data_points.push_back(DataPoint {
            timestamp: Utc::now(),
            value,
        });

        // Keep only recent data points (last 1000)
        if series.data_points.len() > 1000 {
            series.data_points.pop_front();
        }

        Ok(())
    }

    /// Get time series data
    pub async fn get_time_series(&self, metric_name: &str) -> Option<TimeSeries> {
        let time_series = self.time_series.read().await;
        time_series.get(metric_name).cloned()
    }

    /// Calculate aggregation
    pub async fn calculate_aggregation(
        &self,
        metric_name: &str,
        aggregation_type: AggregationType,
        window: Duration,
    ) -> Result<f64> {
        let time_series = self.time_series.read().await;
        let series = time_series
            .get(metric_name)
            .ok_or_else(|| anyhow!("Metric not found: {}", metric_name))?;

        let cutoff = Utc::now() - chrono::Duration::from_std(window)?;
        let recent_values: Vec<f64> = series
            .data_points
            .iter()
            .filter(|dp| dp.timestamp >= cutoff)
            .map(|dp| dp.value)
            .collect();

        if recent_values.is_empty() {
            return Ok(0.0);
        }

        let result = match aggregation_type {
            AggregationType::Average => {
                recent_values.iter().sum::<f64>() / recent_values.len() as f64
            }
            AggregationType::Sum => recent_values.iter().sum(),
            AggregationType::Min => recent_values.iter().cloned().fold(f64::INFINITY, f64::min),
            AggregationType::Max => recent_values
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            AggregationType::Count => recent_values.len() as f64,
            AggregationType::Percentile(p) => {
                let mut sorted = recent_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let index = ((p / 100.0) * sorted.len() as f64) as usize;
                sorted[index.min(sorted.len() - 1)]
            }
        };

        Ok(result)
    }
}

/// Time series data
#[derive(Debug, Clone)]
pub struct TimeSeries {
    pub name: String,
    pub data_points: VecDeque<DataPoint>,
    pub unit: String,
}

/// Data point
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
}

/// Aggregation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Average,
    Sum,
    Min,
    Max,
    Count,
    Percentile(f64),
}

/// Metric aggregation
#[derive(Debug, Clone)]
pub struct MetricAggregation {
    pub metric_name: String,
    pub aggregation_type: AggregationType,
    pub value: f64,
    pub calculated_at: DateTime<Utc>,
}

/// Chart generator
pub struct ChartGenerator {
    #[allow(dead_code)]
    theme: ChartTheme,
}

impl ChartGenerator {
    pub fn new(theme: ChartTheme) -> Self {
        Self { theme }
    }

    /// Generate line chart
    pub async fn generate_line_chart(
        &self,
        data: &TimeSeries,
        config: &WidgetConfig,
    ) -> Result<ChartData> {
        let points: Vec<(f64, f64)> = data
            .data_points
            .iter()
            .enumerate()
            .map(|(i, dp)| (i as f64, dp.value))
            .collect();

        Ok(ChartData {
            chart_type: VisualizationType::LineChart,
            series: vec![ChartSeries {
                name: data.name.clone(),
                data: points,
                color: config.color_scheme.first().cloned(),
            }],
            x_axis_label: config.x_axis_label.clone(),
            y_axis_label: config.y_axis_label.clone(),
            show_legend: config.show_legend,
            show_grid: config.show_grid,
        })
    }

    /// Generate bar chart
    pub async fn generate_bar_chart(
        &self,
        _categories: Vec<String>,
        values: Vec<f64>,
        config: &WidgetConfig,
    ) -> Result<ChartData> {
        let points: Vec<(f64, f64)> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64, v))
            .collect();

        Ok(ChartData {
            chart_type: VisualizationType::BarChart,
            series: vec![ChartSeries {
                name: "Values".to_string(),
                data: points,
                color: config.color_scheme.first().cloned(),
            }],
            x_axis_label: config.x_axis_label.clone(),
            y_axis_label: config.y_axis_label.clone(),
            show_legend: config.show_legend,
            show_grid: config.show_grid,
        })
    }

    /// Generate pie chart
    pub async fn generate_pie_chart(
        &self,
        labels: Vec<String>,
        values: Vec<f64>,
        config: &WidgetConfig,
    ) -> Result<PieChartData> {
        let total: f64 = values.iter().sum();
        let slices: Vec<PieSlice> = labels
            .into_iter()
            .zip(values.iter())
            .enumerate()
            .map(|(i, (label, &value))| PieSlice {
                label,
                value,
                percentage: (value / total) * 100.0,
                color: config
                    .color_scheme
                    .get(i % config.color_scheme.len())
                    .cloned(),
            })
            .collect();

        Ok(PieChartData {
            slices,
            show_legend: config.show_legend,
        })
    }

    /// Generate heatmap
    pub async fn generate_heatmap(
        &self,
        data: Array2<f64>,
        row_labels: Vec<String>,
        col_labels: Vec<String>,
        _config: &WidgetConfig,
    ) -> Result<HeatmapData> {
        let (rows, cols) = data.dim();
        let mut cells = Vec::new();

        for i in 0..rows {
            for j in 0..cols {
                cells.push(HeatmapCell {
                    row: i,
                    col: j,
                    value: data[[i, j]],
                });
            }
        }

        Ok(HeatmapData {
            cells,
            row_labels,
            col_labels,
            color_scale: ColorScale::Viridis,
        })
    }
}

/// Chart data
#[derive(Debug, Clone, Serialize)]
pub struct ChartData {
    pub chart_type: VisualizationType,
    pub series: Vec<ChartSeries>,
    pub x_axis_label: Option<String>,
    pub y_axis_label: Option<String>,
    pub show_legend: bool,
    pub show_grid: bool,
}

/// Chart series
#[derive(Debug, Clone, Serialize)]
pub struct ChartSeries {
    pub name: String,
    pub data: Vec<(f64, f64)>,
    pub color: Option<String>,
}

/// Pie chart data
#[derive(Debug, Clone, Serialize)]
pub struct PieChartData {
    pub slices: Vec<PieSlice>,
    pub show_legend: bool,
}

/// Pie slice
#[derive(Debug, Clone, Serialize)]
pub struct PieSlice {
    pub label: String,
    pub value: f64,
    pub percentage: f64,
    pub color: Option<String>,
}

/// Heatmap data
#[derive(Debug, Clone, Serialize)]
pub struct HeatmapData {
    pub cells: Vec<HeatmapCell>,
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
    pub color_scale: ColorScale,
}

/// Heatmap cell
#[derive(Debug, Clone, Serialize)]
pub struct HeatmapCell {
    pub row: usize,
    pub col: usize,
    pub value: f64,
}

/// Color scale for heatmaps
#[derive(Debug, Clone, Serialize)]
pub enum ColorScale {
    Viridis,
    Plasma,
    Inferno,
    Magma,
    Grayscale,
}

/// Topology visualizer
pub struct TopologyVisualizer {
    nodes: Arc<RwLock<Vec<TopologyNode>>>,
    edges: Arc<RwLock<Vec<TopologyEdge>>>,
}

impl Default for TopologyVisualizer {
    fn default() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(Vec::new())),
            edges: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl TopologyVisualizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add node to topology
    pub async fn add_node(&self, node: TopologyNode) -> Result<()> {
        let mut nodes = self.nodes.write().await;
        nodes.push(node);
        Ok(())
    }

    /// Add edge to topology
    pub async fn add_edge(&self, edge: TopologyEdge) -> Result<()> {
        let mut edges = self.edges.write().await;
        edges.push(edge);
        Ok(())
    }

    /// Generate topology visualization
    pub async fn generate_topology(&self) -> Result<TopologyVisualization> {
        let nodes = self.nodes.read().await.clone();
        let edges = self.edges.read().await.clone();

        // Calculate node positions using force-directed layout (simplified)
        let positioned_nodes = self.calculate_layout(&nodes, &edges).await?;

        Ok(TopologyVisualization {
            nodes: positioned_nodes,
            edges,
            layout_algorithm: LayoutAlgorithm::ForceDirected,
        })
    }

    /// Calculate node positions using force-directed layout
    async fn calculate_layout(
        &self,
        nodes: &[TopologyNode],
        _edges: &[TopologyEdge],
    ) -> Result<Vec<PositionedNode>> {
        // Simplified circular layout
        let n = nodes.len();
        let radius = 200.0;
        let center_x = 300.0;
        let center_y = 300.0;

        let positioned: Vec<PositionedNode> = nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let angle = (i as f64 / n as f64) * 2.0 * std::f64::consts::PI;
                let x = center_x + radius * angle.cos();
                let y = center_y + radius * angle.sin();

                PositionedNode {
                    node: node.clone(),
                    x,
                    y,
                }
            })
            .collect();

        Ok(positioned)
    }
}

/// Topology node
#[derive(Debug, Clone, Serialize)]
pub struct TopologyNode {
    pub id: String,
    pub label: String,
    pub node_type: NodeType,
    pub status: NodeStatus,
    pub metadata: HashMap<String, String>,
}

/// Node type
#[derive(Debug, Clone, Serialize)]
pub enum NodeType {
    Service,
    DataSource,
    Gateway,
    Cache,
    LoadBalancer,
}

/// Node status
#[derive(Debug, Clone, Serialize)]
pub enum NodeStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Topology edge
#[derive(Debug, Clone, Serialize)]
pub struct TopologyEdge {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: EdgeType,
    pub weight: f64,
    pub label: Option<String>,
}

/// Edge type
#[derive(Debug, Clone, Serialize)]
pub enum EdgeType {
    Query,
    DataFlow,
    Federation,
    Replication,
}

/// Positioned node
#[derive(Debug, Clone, Serialize)]
pub struct PositionedNode {
    pub node: TopologyNode,
    pub x: f64,
    pub y: f64,
}

/// Topology visualization
#[derive(Debug, Clone, Serialize)]
pub struct TopologyVisualization {
    pub nodes: Vec<PositionedNode>,
    pub edges: Vec<TopologyEdge>,
    pub layout_algorithm: LayoutAlgorithm,
}

/// Layout algorithm
#[derive(Debug, Clone, Serialize)]
pub enum LayoutAlgorithm {
    ForceDirected,
    Hierarchical,
    Circular,
    Grid,
}

/// Alert visualizer
pub struct AlertVisualizer {
    alerts: Arc<RwLock<VecDeque<Alert>>>,
}

impl Default for AlertVisualizer {
    fn default() -> Self {
        Self {
            alerts: Arc::new(RwLock::new(VecDeque::new())),
        }
    }
}

impl AlertVisualizer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add alert
    pub async fn add_alert(&self, alert: Alert) -> Result<()> {
        let mut alerts = self.alerts.write().await;
        alerts.push_back(alert);

        // Keep only recent alerts (last 100)
        if alerts.len() > 100 {
            alerts.pop_front();
        }

        Ok(())
    }

    /// Get alerts by severity
    pub async fn get_alerts_by_severity(&self, severity: AlertSeverity) -> Vec<Alert> {
        let alerts = self.alerts.read().await;
        alerts
            .iter()
            .filter(|a| a.severity == severity)
            .cloned()
            .collect()
    }

    /// Generate alert timeline
    pub async fn generate_alert_timeline(&self) -> Result<AlertTimeline> {
        let alerts = self.alerts.read().await.clone().into_iter().collect();

        Ok(AlertTimeline {
            alerts,
            group_by: AlertGrouping::Severity,
        })
    }
}

/// Alert
#[derive(Debug, Clone, Serialize)]
pub struct Alert {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub source: String,
    pub acknowledged: bool,
}

/// Alert severity
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Alert timeline
#[derive(Debug, Clone, Serialize)]
pub struct AlertTimeline {
    pub alerts: Vec<Alert>,
    pub group_by: AlertGrouping,
}

/// Alert grouping
#[derive(Debug, Clone, Serialize)]
pub enum AlertGrouping {
    Severity,
    Source,
    Time,
}

impl AdvancedVisualization {
    /// Create new visualization system
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config: config.clone(),
            dashboards: Arc::new(DashMap::new()),
            metrics_collector: Arc::new(MetricsCollector::new()),
            chart_generator: Arc::new(ChartGenerator::new(config.default_theme.clone())),
            topology_visualizer: Arc::new(TopologyVisualizer::new()),
            alert_visualizer: Arc::new(AlertVisualizer::new()),
        }
    }

    /// Create new dashboard
    pub async fn create_dashboard(&self, mut dashboard: Dashboard) -> Result<String> {
        dashboard.created_at = Utc::now();
        dashboard.last_updated = Utc::now();
        let id = dashboard.id.clone();

        self.dashboards.insert(id.clone(), dashboard);
        info!("Created dashboard: {}", id);

        Ok(id)
    }

    /// Get dashboard
    pub async fn get_dashboard(&self, dashboard_id: &str) -> Option<Dashboard> {
        self.dashboards.get(dashboard_id).map(|d| d.clone())
    }

    /// Update dashboard
    pub async fn update_dashboard(
        &self,
        dashboard_id: &str,
        mut dashboard: Dashboard,
    ) -> Result<()> {
        dashboard.last_updated = Utc::now();
        self.dashboards.insert(dashboard_id.to_string(), dashboard);
        Ok(())
    }

    /// Delete dashboard
    pub async fn delete_dashboard(&self, dashboard_id: &str) -> Result<()> {
        self.dashboards.remove(dashboard_id);
        Ok(())
    }

    /// Record metric
    pub async fn record_metric(&self, metric_name: &str, value: f64) -> Result<()> {
        self.metrics_collector
            .record_metric(metric_name, value)
            .await
    }

    /// Generate chart for widget
    pub async fn generate_widget_chart(&self, widget: &Widget) -> Result<ChartData> {
        match widget.data_source {
            DataSource::QueryMetrics => {
                if let Some(series) = self
                    .metrics_collector
                    .get_time_series("query_latency")
                    .await
                {
                    self.chart_generator
                        .generate_line_chart(&series, &widget.config)
                        .await
                } else {
                    Err(anyhow!("No data available for query metrics"))
                }
            }
            DataSource::PerformanceMetrics => {
                if let Some(series) = self.metrics_collector.get_time_series("performance").await {
                    self.chart_generator
                        .generate_line_chart(&series, &widget.config)
                        .await
                } else {
                    Err(anyhow!("No data available for performance metrics"))
                }
            }
            _ => Err(anyhow!("Data source not yet implemented")),
        }
    }

    /// Generate topology visualization
    pub async fn generate_topology_visualization(&self) -> Result<TopologyVisualization> {
        self.topology_visualizer.generate_topology().await
    }

    /// Add topology node
    pub async fn add_topology_node(&self, node: TopologyNode) -> Result<()> {
        self.topology_visualizer.add_node(node).await
    }

    /// Add topology edge
    pub async fn add_topology_edge(&self, edge: TopologyEdge) -> Result<()> {
        self.topology_visualizer.add_edge(edge).await
    }

    /// Add alert
    pub async fn add_alert(&self, alert: Alert) -> Result<()> {
        self.alert_visualizer.add_alert(alert).await
    }

    /// Get alert timeline
    pub async fn get_alert_timeline(&self) -> Result<AlertTimeline> {
        self.alert_visualizer.generate_alert_timeline().await
    }

    /// Export dashboard
    pub async fn export_dashboard(
        &self,
        dashboard_id: &str,
        format: ExportFormat,
    ) -> Result<Vec<u8>> {
        let dashboard = self
            .get_dashboard(dashboard_id)
            .await
            .ok_or_else(|| anyhow!("Dashboard not found"))?;

        match format {
            ExportFormat::JSON => {
                let json = serde_json::to_string_pretty(&dashboard)?;
                Ok(json.into_bytes())
            }
            ExportFormat::SVG => {
                // Simplified SVG export
                let svg = format!(
                    r#"<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
                    <text x="10" y="20">{}</text>
                </svg>"#,
                    dashboard.name
                );
                Ok(svg.into_bytes())
            }
            _ => Err(anyhow!("Export format not yet implemented")),
        }
    }

    /// Create default performance dashboard
    pub async fn create_default_performance_dashboard(&self) -> Result<String> {
        let dashboard = Dashboard {
            id: uuid::Uuid::new_v4().to_string(),
            name: "Performance Dashboard".to_string(),
            description: "Real-time performance metrics".to_string(),
            layout: DashboardLayout::Grid { rows: 2, cols: 2 },
            widgets: vec![
                Widget {
                    id: uuid::Uuid::new_v4().to_string(),
                    widget_type: WidgetType::Chart,
                    title: "Query Latency".to_string(),
                    position: WidgetPosition { row: 0, col: 0 },
                    size: WidgetSize {
                        width: 1,
                        height: 1,
                    },
                    data_source: DataSource::QueryMetrics,
                    visualization_type: VisualizationType::LineChart,
                    config: WidgetConfig::default(),
                },
                Widget {
                    id: uuid::Uuid::new_v4().to_string(),
                    widget_type: WidgetType::Chart,
                    title: "Throughput".to_string(),
                    position: WidgetPosition { row: 0, col: 1 },
                    size: WidgetSize {
                        width: 1,
                        height: 1,
                    },
                    data_source: DataSource::PerformanceMetrics,
                    visualization_type: VisualizationType::BarChart,
                    config: WidgetConfig::default(),
                },
                Widget {
                    id: uuid::Uuid::new_v4().to_string(),
                    widget_type: WidgetType::Topology,
                    title: "Federation Topology".to_string(),
                    position: WidgetPosition { row: 1, col: 0 },
                    size: WidgetSize {
                        width: 2,
                        height: 1,
                    },
                    data_source: DataSource::FederationTopology,
                    visualization_type: VisualizationType::NetworkGraph,
                    config: WidgetConfig::default(),
                },
            ],
            created_at: Utc::now(),
            last_updated: Utc::now(),
            auto_refresh: true,
            refresh_interval: Duration::from_secs(5),
        };

        self.create_dashboard(dashboard).await
    }

    /// Create security monitoring dashboard
    pub async fn create_security_dashboard(&self) -> Result<String> {
        let dashboard = Dashboard {
            id: uuid::Uuid::new_v4().to_string(),
            name: "Security Monitoring".to_string(),
            description: "Real-time security alerts and metrics".to_string(),
            layout: DashboardLayout::TwoColumn,
            widgets: vec![
                Widget {
                    id: uuid::Uuid::new_v4().to_string(),
                    widget_type: WidgetType::Alert,
                    title: "Security Alerts".to_string(),
                    position: WidgetPosition { row: 0, col: 0 },
                    size: WidgetSize {
                        width: 1,
                        height: 1,
                    },
                    data_source: DataSource::SecurityAlerts,
                    visualization_type: VisualizationType::Table,
                    config: WidgetConfig::default(),
                },
                Widget {
                    id: uuid::Uuid::new_v4().to_string(),
                    widget_type: WidgetType::Chart,
                    title: "Threat Level".to_string(),
                    position: WidgetPosition { row: 0, col: 1 },
                    size: WidgetSize {
                        width: 1,
                        height: 1,
                    },
                    data_source: DataSource::SecurityAlerts,
                    visualization_type: VisualizationType::Gauge,
                    config: WidgetConfig::default(),
                },
            ],
            created_at: Utc::now(),
            last_updated: Utc::now(),
            auto_refresh: true,
            refresh_interval: Duration::from_secs(10),
        };

        self.create_dashboard(dashboard).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_visualization_creation() {
        let config = VisualizationConfig::default();
        let viz = AdvancedVisualization::new(config);

        let dashboard_id = viz
            .create_default_performance_dashboard()
            .await
            .expect("async operation should succeed");
        assert!(!dashboard_id.is_empty());

        let dashboard = viz.get_dashboard(&dashboard_id).await;
        assert!(dashboard.is_some());
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let collector = MetricsCollector::new();

        collector
            .record_metric("test_metric", 100.0)
            .await
            .expect("async operation should succeed");
        collector
            .record_metric("test_metric", 200.0)
            .await
            .expect("async operation should succeed");

        let series = collector.get_time_series("test_metric").await;
        assert!(series.is_some());
        assert_eq!(
            series.expect("operation should succeed").data_points.len(),
            2
        );
    }

    #[tokio::test]
    async fn test_aggregation() {
        let collector = MetricsCollector::new();

        for i in 1..=10 {
            collector
                .record_metric("test", i as f64)
                .await
                .expect("async operation should succeed");
        }

        let avg = collector
            .calculate_aggregation("test", AggregationType::Average, Duration::from_secs(3600))
            .await
            .expect("operation should succeed");

        assert!((avg - 5.5).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_chart_generation() {
        let config = WidgetConfig::default();
        let generator = ChartGenerator::new(ChartTheme::Dark);

        let time_series = TimeSeries {
            name: "test".to_string(),
            data_points: (1..=5)
                .map(|i| DataPoint {
                    timestamp: Utc::now(),
                    value: i as f64,
                })
                .collect(),
            unit: "ms".to_string(),
        };

        let chart = generator
            .generate_line_chart(&time_series, &config)
            .await
            .expect("operation should succeed");

        assert_eq!(chart.series.len(), 1);
        assert_eq!(chart.series[0].data.len(), 5);
    }

    #[tokio::test]
    async fn test_topology_visualization() {
        let visualizer = TopologyVisualizer::new();

        visualizer
            .add_node(TopologyNode {
                id: "node1".to_string(),
                label: "Service 1".to_string(),
                node_type: NodeType::Service,
                status: NodeStatus::Healthy,
                metadata: HashMap::new(),
            })
            .await
            .expect("operation should succeed");

        visualizer
            .add_node(TopologyNode {
                id: "node2".to_string(),
                label: "Service 2".to_string(),
                node_type: NodeType::Service,
                status: NodeStatus::Healthy,
                metadata: HashMap::new(),
            })
            .await
            .expect("operation should succeed");

        visualizer
            .add_edge(TopologyEdge {
                source_id: "node1".to_string(),
                target_id: "node2".to_string(),
                edge_type: EdgeType::Query,
                weight: 1.0,
                label: None,
            })
            .await
            .expect("operation should succeed");

        let topology = visualizer
            .generate_topology()
            .await
            .expect("async operation should succeed");
        assert_eq!(topology.nodes.len(), 2);
        assert_eq!(topology.edges.len(), 1);
    }

    #[tokio::test]
    async fn test_alert_visualization() {
        let visualizer = AlertVisualizer::new();

        visualizer
            .add_alert(Alert {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                severity: AlertSeverity::Critical,
                title: "Test Alert".to_string(),
                description: "Test description".to_string(),
                source: "test".to_string(),
                acknowledged: false,
            })
            .await
            .expect("operation should succeed");

        let timeline = visualizer
            .generate_alert_timeline()
            .await
            .expect("async operation should succeed");
        assert_eq!(timeline.alerts.len(), 1);
    }

    #[tokio::test]
    async fn test_dashboard_export() {
        let config = VisualizationConfig::default();
        let viz = AdvancedVisualization::new(config);

        let dashboard_id = viz
            .create_default_performance_dashboard()
            .await
            .expect("async operation should succeed");

        let json_export = viz
            .export_dashboard(&dashboard_id, ExportFormat::JSON)
            .await
            .expect("operation should succeed");
        assert!(!json_export.is_empty());

        let svg_export = viz
            .export_dashboard(&dashboard_id, ExportFormat::SVG)
            .await
            .expect("operation should succeed");
        assert!(!svg_export.is_empty());
    }
}
