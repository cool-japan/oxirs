//! Core visualization engine and configuration types

use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::{Result, ShaclAiError};

use super::{collectors::DataCollector, renderers::VisualizationRenderer};

/// Advanced visualization engine for SHACL-AI components
#[derive(Debug)]
pub struct AdvancedVisualizationEngine {
    /// Visualization configurations
    config: VisualizationConfig,
    /// Real-time data collectors
    data_collectors: Arc<RwLock<HashMap<String, Box<dyn DataCollector>>>>,
    /// Visualization renderers
    renderers: Arc<RwLock<HashMap<String, Box<dyn VisualizationRenderer>>>>,
    /// Active visualizations
    active_visualizations: Arc<RwLock<HashMap<String, ActiveVisualization>>>,
}

impl AdvancedVisualizationEngine {
    /// Create a new advanced visualization engine
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config,
            data_collectors: Arc::new(RwLock::new(HashMap::new())),
            renderers: Arc::new(RwLock::new(HashMap::new())),
            active_visualizations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a data collector
    pub async fn register_data_collector(
        &self,
        name: String,
        collector: Box<dyn DataCollector>,
    ) -> Result<()> {
        let mut collectors = self.data_collectors.write().await;
        collectors.insert(name, collector);
        Ok(())
    }

    /// Register a visualization renderer
    pub async fn register_renderer(
        &self,
        name: String,
        renderer: Box<dyn VisualizationRenderer>,
    ) -> Result<()> {
        let mut renderers = self.renderers.write().await;
        renderers.insert(name, renderer);
        Ok(())
    }

    /// Create a new visualization
    pub async fn create_visualization(
        &self,
        visualization_id: String,
        collector_name: String,
        renderer_name: String,
        options: RenderOptions,
    ) -> Result<VisualizationOutput> {
        // Get collector and renderer
        let collectors = self.data_collectors.read().await;
        let renderers = self.renderers.read().await;

        let collector = collectors.get(&collector_name).ok_or_else(|| {
            ShaclAiError::Visualization(format!("Collector '{collector_name}' not found"))
        })?;

        let renderer = renderers.get(&renderer_name).ok_or_else(|| {
            ShaclAiError::Visualization(format!("Renderer '{renderer_name}' not found"))
        })?;

        // Collect data
        let data = collector.collect_data().await?;

        // Render visualization
        let output = renderer.render(&data, &options).await?;

        // Store active visualization
        let active_viz = ActiveVisualization {
            output: output.clone(),
            last_updated: SystemTime::now(),
            update_count: 1,
        };

        let mut active = self.active_visualizations.write().await;
        active.insert(visualization_id, active_viz);

        Ok(output)
    }

    /// Update an existing visualization
    pub async fn update_visualization(
        &self,
        visualization_id: &str,
    ) -> Result<VisualizationOutput> {
        let mut active = self.active_visualizations.write().await;

        if let Some(active_viz) = active.get_mut(visualization_id) {
            // Update the visualization (simplified - would need to re-collect and render)
            active_viz.last_updated = SystemTime::now();
            active_viz.update_count += 1;

            Ok(active_viz.output.clone())
        } else {
            Err(ShaclAiError::Visualization(format!(
                "Visualization '{visualization_id}' not found"
            )))
        }
    }

    /// Get active visualization
    pub async fn get_visualization(
        &self,
        visualization_id: &str,
    ) -> Result<Option<VisualizationOutput>> {
        let active = self.active_visualizations.read().await;
        Ok(active.get(visualization_id).map(|v| v.output.clone()))
    }

    /// List all active visualizations
    pub async fn list_visualizations(&self) -> Result<Vec<String>> {
        let active = self.active_visualizations.read().await;
        Ok(active.keys().cloned().collect())
    }

    /// Get configuration
    pub fn config(&self) -> &VisualizationConfig {
        &self.config
    }
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub default_color_scheme: ColorScheme,
    pub enable_animations: bool,
    pub max_data_points: usize,
    pub update_interval_ms: u64,
    pub enable_real_time: bool,
    pub export_quality: ExportQuality,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            default_color_scheme: ColorScheme::Viridis,
            enable_animations: true,
            max_data_points: 10000,
            update_interval_ms: 1000,
            enable_real_time: true,
            export_quality: ExportQuality::High,
        }
    }
}

/// Visualization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationType {
    HierarchicalGraph,
    FlowDiagram,
    ScatterPlot3D,
    ComplexPlane,
    NetworkGraph,
    Heatmap,
    Timeline,
    Dashboard,
    InterpretabilityMap,
    AttentionFlow,
    LayerPerformance,
    BottleneckAnalysis,
    MemoryProfile,
    PerformanceDebugger,
    InteractiveExplorer,
    QuantumClassicalHybrid,
}

/// Visualization data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisualizationData {
    Graph {
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
    },
    TimeSeries {
        series: Vec<TimeSeriesData>,
    },
    Heatmap {
        matrix: Vec<Vec<f64>>,
        labels: HeatmapLabels,
    },
    Quantum {
        states: Vec<QuantumStateVisualization>,
        entanglements: Vec<EntanglementVisualization>,
    },
    Dashboard {
        components: Vec<VisualizationOutput>,
        layout: DashboardLayout,
    },
    Interpretability {
        explanations: Vec<InterpretabilityExplanation>,
    },
    AttentionFlow {
        flows: Vec<AttentionFlow>,
    },
    PerformanceDebug {
        debug_info: Vec<PerformanceDebugInfo>,
    },
    InteractiveExplorer {
        nodes: Vec<ExplorerNode>,
    },
    QuantumClassicalHybrid {
        connections: Vec<QuantumClassicalConnection>,
    },
}

/// Visualization output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationOutput {
    pub id: String,
    pub visualization_type: VisualizationType,
    pub data: VisualizationData,
    pub metadata: VisualizationMetadata,
    pub export_formats: Vec<ExportFormat>,
}

/// Visualization metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationMetadata {
    pub created_at: SystemTime,
    pub title: String,
    pub description: String,
    pub interactive: bool,
    pub real_time: bool,
}

/// Active visualization tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveVisualization {
    pub output: VisualizationOutput,
    pub last_updated: SystemTime,
    pub update_count: u64,
}

/// Render options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderOptions {
    pub visualization_type: VisualizationType,
    pub color_scheme: ColorScheme,
    pub interactive: bool,
    pub animation: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            visualization_type: VisualizationType::NetworkGraph,
            color_scheme: ColorScheme::Viridis,
            interactive: true,
            animation: true,
        }
    }
}

/// Color schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorScheme {
    Viridis,
    Plasma,
    Inferno,
    Magma,
    Blues,
    Reds,
    Greens,
    Custom(Vec<String>),
}

/// Export quality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportQuality {
    Low,
    Medium,
    High,
    Ultra,
}

/// Export formats
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum ExportFormat {
    SVG,
    PNG,
    HTML,
    PDF,
    JSON,
}

/// Graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub position: (f64, f64, f64),
    pub size: f64,
    pub color: String,
    pub metadata: HashMap<String, String>,
}

/// Graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub weight: f64,
    pub color: String,
    pub metadata: HashMap<String, String>,
}

/// Time series data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub name: String,
    pub data_points: Vec<(f64, f64)>, // (timestamp, value)
    pub color: String,
}

/// Heatmap labels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapLabels {
    pub x_labels: Vec<String>,
    pub y_labels: Vec<String>,
}

/// Quantum state visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumStateVisualization {
    pub state_id: String,
    pub amplitude: f64,
    pub phase: f64,
    pub position: (f64, f64),
}

/// Entanglement visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementVisualization {
    pub qubit1: String,
    pub qubit2: String,
    pub strength: f64,
}

/// Dashboard layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayout {
    pub grid_columns: usize,
    pub grid_rows: usize,
    pub component_positions: HashMap<String, (usize, usize)>,
}

/// Interpretability explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityExplanation {
    pub feature: String,
    pub importance: f64,
    pub explanation: String,
}

/// Attention flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlow {
    pub from_layer: String,
    pub to_layer: String,
    pub attention_weights: Vec<f64>,
}

/// Performance debug info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDebugInfo {
    pub component: String,
    pub metric: String,
    pub value: f64,
    pub threshold: Option<f64>,
}

/// Explorer node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorerNode {
    pub id: String,
    pub name: String,
    pub node_type: String,
    pub children: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Quantum-classical connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumClassicalConnection {
    pub quantum_component: String,
    pub classical_component: String,
    pub connection_strength: f64,
    pub data_flow: String,
}

/// Architecture visualization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchitectureVisualizationType {
    LayerHierarchy,
    AttentionFlow,
    EmbeddingSpace,
}

/// Quantum visualization modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumVisualizationMode {
    StateVector,
    Entanglement,
    Coherence,
}

/// Result of an export operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportResult {
    /// Export format used
    pub format: ExportFormat,
    /// Path to exported file
    pub file_path: String,
    /// Size of exported file in bytes
    pub file_size: u64,
    /// Export timestamp
    pub timestamp: SystemTime,
    /// Export success status
    pub success: bool,
    /// Optional error message
    pub error_message: Option<String>,
}

/// Interactive controls for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveControls {
    /// Enable zoom functionality
    pub enable_zoom: bool,
    /// Enable pan functionality
    pub enable_pan: bool,
    /// Enable node selection
    pub enable_selection: bool,
    /// Enable hover tooltips
    pub enable_tooltips: bool,
    /// Enable animation controls
    pub enable_animation_controls: bool,
    /// Enable filter controls
    pub enable_filters: bool,
    /// Enable export controls
    pub enable_export: bool,
    /// Custom control configurations
    pub custom_controls: HashMap<String, String>,
}
