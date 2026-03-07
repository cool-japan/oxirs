//! Result Visualization Helpers
//!
//! Provides utilities for generating visualizations and visual representations
//! of query results, knowledge graph structures, and analytics data.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Visualization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationType {
    /// Table view
    Table,
    /// Bar chart
    BarChart,
    /// Line chart
    LineChart,
    /// Pie chart
    PieChart,
    /// Network/graph visualization
    Network,
    /// Tree visualization
    Tree,
    /// Timeline visualization
    Timeline,
    /// Heatmap
    Heatmap,
    /// Geographic map
    GeoMap,
    /// Custom visualization
    Custom,
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Visualization type
    pub viz_type: VisualizationType,
    /// Title
    pub title: String,
    /// Width (pixels or percentage)
    pub width: String,
    /// Height (pixels or percentage)
    pub height: String,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Interactive features
    pub interactive: bool,
    /// Additional options
    pub options: HashMap<String, serde_json::Value>,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            viz_type: VisualizationType::Table,
            title: "Data Visualization".to_string(),
            width: "100%".to_string(),
            height: "400px".to_string(),
            color_scheme: ColorScheme::Default,
            interactive: true,
            options: HashMap::new(),
        }
    }
}

/// Color scheme for visualizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Default colors
    Default,
    /// Blue theme
    Blue,
    /// Green theme
    Green,
    /// Red theme
    Red,
    /// Purple theme
    Purple,
    /// Grayscale
    Grayscale,
    /// Rainbow
    Rainbow,
    /// Custom
    Custom,
}

/// Visualization data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// X-axis value
    pub x: String,
    /// Y-axis value
    pub y: f64,
    /// Label
    pub label: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Network node for graph visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkNode {
    /// Node ID
    pub id: String,
    /// Node label
    pub label: String,
    /// Node type/category
    pub node_type: String,
    /// Node size (relative)
    pub size: f32,
    /// Node color
    pub color: Option<String>,
    /// Position (x, y)
    pub position: Option<(f64, f64)>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Network edge for graph visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge label
    pub label: Option<String>,
    /// Edge weight
    pub weight: f32,
    /// Edge color
    pub color: Option<String>,
    /// Is directed?
    pub directed: bool,
}

/// Visualization specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationSpec {
    /// Configuration
    pub config: VisualizationConfig,
    /// Data (format depends on visualization type)
    pub data: VisualizationData,
    /// Rendering hints
    pub rendering_hints: HashMap<String, String>,
}

/// Visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VisualizationData {
    /// Tabular data
    Table {
        columns: Vec<String>,
        rows: Vec<Vec<String>>,
    },
    /// Chart data points
    Chart { series: Vec<ChartSeries> },
    /// Network/graph data
    Network {
        nodes: Vec<NetworkNode>,
        edges: Vec<NetworkEdge>,
    },
    /// Tree data
    Tree { root: TreeNode },
    /// Timeline events
    Timeline { events: Vec<TimelineEvent> },
    /// Heatmap data
    Heatmap {
        data: Vec<Vec<f64>>,
        x_labels: Vec<String>,
        y_labels: Vec<String>,
    },
}

/// Chart series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartSeries {
    /// Series name
    pub name: String,
    /// Data points
    pub data: Vec<DataPoint>,
    /// Series color
    pub color: Option<String>,
}

/// Tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// Node ID
    pub id: String,
    /// Node label
    pub label: String,
    /// Children
    pub children: Vec<TreeNode>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Timeline event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    /// Event ID
    pub id: String,
    /// Event title
    pub title: String,
    /// Event description
    pub description: Option<String>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Duration (if applicable)
    pub duration: Option<chrono::Duration>,
    /// Category
    pub category: Option<String>,
}

/// Visualization builder
pub struct VisualizationBuilder {
    spec: VisualizationSpec,
}

impl VisualizationBuilder {
    /// Create a new visualization builder
    pub fn new(viz_type: VisualizationType) -> Self {
        Self {
            spec: VisualizationSpec {
                config: VisualizationConfig {
                    viz_type,
                    ..Default::default()
                },
                data: VisualizationData::Table {
                    columns: vec![],
                    rows: vec![],
                },
                rendering_hints: HashMap::new(),
            },
        }
    }

    /// Set title
    pub fn title(mut self, title: String) -> Self {
        self.spec.config.title = title;
        self
    }

    /// Set dimensions
    pub fn dimensions(mut self, width: String, height: String) -> Self {
        self.spec.config.width = width;
        self.spec.config.height = height;
        self
    }

    /// Set color scheme
    pub fn color_scheme(mut self, scheme: ColorScheme) -> Self {
        self.spec.config.color_scheme = scheme;
        self
    }

    /// Set interactive mode
    pub fn interactive(mut self, interactive: bool) -> Self {
        self.spec.config.interactive = interactive;
        self
    }

    /// Add option
    pub fn option<K: Into<String>, V: Serialize>(mut self, key: K, value: V) -> Self {
        self.spec.config.options.insert(
            key.into(),
            serde_json::to_value(value).expect("serializable value should convert to JSON"),
        );
        self
    }

    /// Set table data
    pub fn table_data(mut self, columns: Vec<String>, rows: Vec<Vec<String>>) -> Self {
        self.spec.data = VisualizationData::Table { columns, rows };
        self
    }

    /// Set chart data
    pub fn chart_data(mut self, series: Vec<ChartSeries>) -> Self {
        self.spec.data = VisualizationData::Chart { series };
        self
    }

    /// Set network data
    pub fn network_data(mut self, nodes: Vec<NetworkNode>, edges: Vec<NetworkEdge>) -> Self {
        self.spec.data = VisualizationData::Network { nodes, edges };
        self
    }

    /// Set tree data
    pub fn tree_data(mut self, root: TreeNode) -> Self {
        self.spec.data = VisualizationData::Tree { root };
        self
    }

    /// Set timeline data
    pub fn timeline_data(mut self, events: Vec<TimelineEvent>) -> Self {
        self.spec.data = VisualizationData::Timeline { events };
        self
    }

    /// Set heatmap data
    pub fn heatmap_data(
        mut self,
        data: Vec<Vec<f64>>,
        x_labels: Vec<String>,
        y_labels: Vec<String>,
    ) -> Self {
        self.spec.data = VisualizationData::Heatmap {
            data,
            x_labels,
            y_labels,
        };
        self
    }

    /// Add rendering hint
    pub fn hint<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.spec.rendering_hints.insert(key.into(), value.into());
        self
    }

    /// Build the visualization specification
    pub fn build(self) -> VisualizationSpec {
        self.spec
    }
}

/// Visualization renderer (outputs various formats)
pub struct VisualizationRenderer;

impl VisualizationRenderer {
    /// Render to JSON
    pub fn to_json(spec: &VisualizationSpec) -> Result<String> {
        Ok(serde_json::to_string_pretty(spec)?)
    }

    /// Render to Vega-Lite JSON (for web rendering)
    pub fn to_vega_lite(spec: &VisualizationSpec) -> Result<String> {
        debug!("Converting to Vega-Lite specification");

        let vega_spec = match &spec.data {
            VisualizationData::Chart { series } => {
                // Convert chart to Vega-Lite spec
                let mut data_values = Vec::new();
                for s in series {
                    for point in &s.data {
                        data_values.push(serde_json::json!({
                            "x": point.x,
                            "y": point.y,
                            "series": s.name,
                        }));
                    }
                }

                serde_json::json!({
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "title": spec.config.title,
                    "width": 600,
                    "height": 400,
                    "data": {
                        "values": data_values
                    },
                    "mark": "line",
                    "encoding": {
                        "x": {"field": "x", "type": "ordinal"},
                        "y": {"field": "y", "type": "quantitative"},
                        "color": {"field": "series", "type": "nominal"}
                    }
                })
            }
            _ => {
                serde_json::json!({
                    "error": "Visualization type not yet supported for Vega-Lite"
                })
            }
        };

        Ok(serde_json::to_string_pretty(&vega_spec)?)
    }

    /// Render to PlantUML (for diagrams)
    pub fn to_plantuml(spec: &VisualizationSpec) -> Result<String> {
        debug!("Converting to PlantUML");

        match &spec.data {
            VisualizationData::Network { nodes, edges } => {
                let mut plantuml = String::from("@startuml\n");

                // Add nodes
                for node in nodes {
                    plantuml.push_str(&format!("object {} {{\n", node.id));
                    plantuml.push_str(&format!("  label = \"{}\"\n", node.label));
                    plantuml.push_str("}\n");
                }

                // Add edges
                for edge in edges {
                    let arrow = if edge.directed { "-->" } else { "--" };
                    if let Some(label) = &edge.label {
                        plantuml.push_str(&format!(
                            "{} {} {} : {}\n",
                            edge.source, arrow, edge.target, label
                        ));
                    } else {
                        plantuml.push_str(&format!("{} {} {}\n", edge.source, arrow, edge.target));
                    }
                }

                plantuml.push_str("@enduml\n");
                Ok(plantuml)
            }
            _ => Err(anyhow::anyhow!(
                "PlantUML rendering only supported for network visualizations"
            )),
        }
    }

    /// Render to HTML table
    pub fn to_html_table(spec: &VisualizationSpec) -> Result<String> {
        match &spec.data {
            VisualizationData::Table { columns, rows } => {
                let mut html = String::from("<table>\n");

                // Header
                html.push_str("  <thead>\n    <tr>\n");
                for col in columns {
                    html.push_str(&format!("      <th>{}</th>\n", col));
                }
                html.push_str("    </tr>\n  </thead>\n");

                // Body
                html.push_str("  <tbody>\n");
                for row in rows {
                    html.push_str("    <tr>\n");
                    for cell in row {
                        html.push_str(&format!("      <td>{}</td>\n", cell));
                    }
                    html.push_str("    </tr>\n");
                }
                html.push_str("  </tbody>\n");

                html.push_str("</table>");
                Ok(html)
            }
            _ => Err(anyhow::anyhow!(
                "HTML table rendering only supported for table visualizations"
            )),
        }
    }

    /// Render to Mermaid diagram
    pub fn to_mermaid(spec: &VisualizationSpec) -> Result<String> {
        match &spec.data {
            VisualizationData::Network { nodes, edges } => {
                let mut mermaid = String::from("graph TD\n");

                // Add nodes with labels
                for node in nodes {
                    mermaid.push_str(&format!("  {}[\"{}\"]\n", node.id, node.label));
                }

                // Add edges
                for edge in edges {
                    if let Some(label) = &edge.label {
                        mermaid.push_str(&format!(
                            "  {} -->|{}| {}\n",
                            edge.source, label, edge.target
                        ));
                    } else {
                        mermaid.push_str(&format!("  {} --> {}\n", edge.source, edge.target));
                    }
                }

                Ok(mermaid)
            }
            VisualizationData::Tree { root } => {
                let mut mermaid = String::from("graph TD\n");

                fn add_tree_node(mermaid: &mut String, node: &TreeNode, parent_id: Option<&str>) {
                    mermaid.push_str(&format!("  {}[\"{}\"]\n", node.id, node.label));

                    if let Some(parent) = parent_id {
                        mermaid.push_str(&format!("  {} --> {}\n", parent, node.id));
                    }

                    for child in &node.children {
                        add_tree_node(mermaid, child, Some(&node.id));
                    }
                }

                add_tree_node(&mut mermaid, root, None);

                Ok(mermaid)
            }
            _ => Err(anyhow::anyhow!(
                "Mermaid rendering supported for network and tree visualizations"
            )),
        }
    }
}

/// Helper functions for common visualizations
pub mod helpers {
    use super::*;

    /// Create a simple bar chart
    pub fn bar_chart(title: String, data: Vec<(String, f64)>) -> VisualizationSpec {
        let series = ChartSeries {
            name: "Values".to_string(),
            data: data
                .into_iter()
                .map(|(x, y)| DataPoint {
                    x,
                    y,
                    label: None,
                    metadata: HashMap::new(),
                })
                .collect(),
            color: None,
        };

        VisualizationBuilder::new(VisualizationType::BarChart)
            .title(title)
            .chart_data(vec![series])
            .build()
    }

    /// Create a simple network graph
    pub fn network_graph(
        title: String,
        nodes: Vec<(String, String)>,
        edges: Vec<(String, String, Option<String>)>,
    ) -> VisualizationSpec {
        let network_nodes: Vec<NetworkNode> = nodes
            .into_iter()
            .map(|(id, label)| NetworkNode {
                id,
                label,
                node_type: "default".to_string(),
                size: 1.0,
                color: None,
                position: None,
                metadata: HashMap::new(),
            })
            .collect();

        let network_edges: Vec<NetworkEdge> = edges
            .into_iter()
            .map(|(source, target, label)| NetworkEdge {
                source,
                target,
                label,
                weight: 1.0,
                color: None,
                directed: true,
            })
            .collect();

        VisualizationBuilder::new(VisualizationType::Network)
            .title(title)
            .network_data(network_nodes, network_edges)
            .build()
    }

    /// Create a simple timeline
    pub fn timeline(title: String, events: Vec<TimelineEvent>) -> VisualizationSpec {
        VisualizationBuilder::new(VisualizationType::Timeline)
            .title(title)
            .timeline_data(events)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualization_builder() {
        let spec = VisualizationBuilder::new(VisualizationType::Table)
            .title("Test Table".to_string())
            .table_data(
                vec!["Col1".to_string(), "Col2".to_string()],
                vec![
                    vec!["A".to_string(), "B".to_string()],
                    vec!["C".to_string(), "D".to_string()],
                ],
            )
            .build();

        assert_eq!(spec.config.title, "Test Table");
        match spec.data {
            VisualizationData::Table { columns, rows } => {
                assert_eq!(columns.len(), 2);
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected table data"),
        }
    }

    #[test]
    fn test_bar_chart_helper() {
        let spec = helpers::bar_chart(
            "Sales".to_string(),
            vec![
                ("Q1".to_string(), 100.0),
                ("Q2".to_string(), 150.0),
                ("Q3".to_string(), 120.0),
            ],
        );

        assert_eq!(spec.config.viz_type, VisualizationType::BarChart);
        assert_eq!(spec.config.title, "Sales");
    }

    #[test]
    fn test_network_graph_helper() {
        let spec = helpers::network_graph(
            "Knowledge Graph".to_string(),
            vec![("A".to_string(), "Node A".to_string())],
            vec![(
                "A".to_string(),
                "B".to_string(),
                Some("relates".to_string()),
            )],
        );

        assert_eq!(spec.config.viz_type, VisualizationType::Network);
    }

    #[test]
    fn test_html_table_rendering() {
        let spec = VisualizationBuilder::new(VisualizationType::Table)
            .table_data(
                vec!["Name".to_string(), "Value".to_string()],
                vec![vec!["Test".to_string(), "123".to_string()]],
            )
            .build();

        let html = VisualizationRenderer::to_html_table(&spec).unwrap();
        assert!(html.contains("<table>"));
        assert!(html.contains("<th>Name</th>"));
        assert!(html.contains("<td>Test</td>"));
    }
}
