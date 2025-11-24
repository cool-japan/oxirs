//! Visual Shape Editor Support
//!
//! Provides export formats and representations suitable for visual shape editing:
//!
//! - **GraphViz/DOT** - Export shapes as DOT graphs for visualization
//! - **Mermaid** - Generate Mermaid diagram syntax
//! - **JSON Schema** - Export as JSON Schema for web-based editors
//! - **SVG** - Generate SVG diagrams of shapes
//! - **Interactive Editor Protocol** - JSON protocol for real-time editors
//!
//! ## Example
//!
//! ```rust,ignore
//! use oxirs_shacl::visual_editor::{ShapeVisualizer, ExportFormat};
//!
//! let visualizer = ShapeVisualizer::new();
//! let dot = visualizer.export(&shape, ExportFormat::Dot)?;
//! let mermaid = visualizer.export(&shape, ExportFormat::Mermaid)?;
//! ```

use crate::{ConstraintComponentId, Shape, ShapeType, Target};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Visual export format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// GraphViz DOT format
    Dot,
    /// Mermaid diagram syntax
    Mermaid,
    /// JSON Schema format
    JsonSchema,
    /// SVG diagram
    Svg,
    /// PlantUML format
    PlantUml,
    /// D3.js JSON format
    D3Json,
    /// Cytoscape.js JSON format
    CytoscapeJson,
}

/// Shape visualizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizerConfig {
    /// Include constraint details
    pub show_constraints: bool,
    /// Include targets
    pub show_targets: bool,
    /// Include metadata
    pub show_metadata: bool,
    /// Color scheme
    pub color_scheme: ColorScheme,
    /// Shape node style
    pub shape_style: ShapeStyle,
    /// Property node style
    pub property_style: PropertyStyle,
    /// Layout direction
    pub layout_direction: LayoutDirection,
    /// Max label length (truncate)
    pub max_label_length: usize,
    /// Show prefixes or full IRIs
    pub use_prefixes: bool,
    /// Namespace prefixes
    pub prefixes: HashMap<String, String>,
}

impl Default for VisualizerConfig {
    fn default() -> Self {
        let mut prefixes = HashMap::new();
        prefixes.insert("sh".to_string(), "http://www.w3.org/ns/shacl#".to_string());
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        prefixes.insert("schema".to_string(), "http://schema.org/".to_string());
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());

        Self {
            show_constraints: true,
            show_targets: true,
            show_metadata: false,
            color_scheme: ColorScheme::default(),
            shape_style: ShapeStyle::default(),
            property_style: PropertyStyle::default(),
            layout_direction: LayoutDirection::TopToBottom,
            max_label_length: 50,
            use_prefixes: true,
            prefixes,
        }
    }
}

/// Color scheme for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    /// Node shape fill color
    pub shape_fill: String,
    /// Node shape border color
    pub shape_border: String,
    /// Property shape fill color
    pub property_fill: String,
    /// Property shape border color
    pub property_border: String,
    /// Target fill color
    pub target_fill: String,
    /// Constraint fill color
    pub constraint_fill: String,
    /// Edge color
    pub edge_color: String,
    /// Text color
    pub text_color: String,
    /// Required indicator color
    pub required_color: String,
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            shape_fill: "#e8f4fc".to_string(),
            shape_border: "#3498db".to_string(),
            property_fill: "#f9f9f9".to_string(),
            property_border: "#7f8c8d".to_string(),
            target_fill: "#d5f5e3".to_string(),
            constraint_fill: "#fef9e7".to_string(),
            edge_color: "#34495e".to_string(),
            text_color: "#2c3e50".to_string(),
            required_color: "#e74c3c".to_string(),
        }
    }
}

/// Shape node style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeStyle {
    /// Node shape (box, ellipse, etc.)
    pub shape: String,
    /// Font size
    pub font_size: u32,
    /// Font family
    pub font_family: String,
    /// Border width
    pub border_width: u32,
}

impl Default for ShapeStyle {
    fn default() -> Self {
        Self {
            shape: "box".to_string(),
            font_size: 12,
            font_family: "Arial".to_string(),
            border_width: 2,
        }
    }
}

/// Property node style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyStyle {
    /// Node shape
    pub shape: String,
    /// Font size
    pub font_size: u32,
    /// Font family
    pub font_family: String,
    /// Border width
    pub border_width: u32,
}

impl Default for PropertyStyle {
    fn default() -> Self {
        Self {
            shape: "record".to_string(),
            font_size: 10,
            font_family: "Arial".to_string(),
            border_width: 1,
        }
    }
}

/// Layout direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayoutDirection {
    TopToBottom,
    LeftToRight,
    BottomToTop,
    RightToLeft,
}

impl LayoutDirection {
    fn as_dot(self) -> &'static str {
        match self {
            LayoutDirection::TopToBottom => "TB",
            LayoutDirection::LeftToRight => "LR",
            LayoutDirection::BottomToTop => "BT",
            LayoutDirection::RightToLeft => "RL",
        }
    }

    fn as_mermaid(self) -> &'static str {
        match self {
            LayoutDirection::TopToBottom => "TD",
            LayoutDirection::LeftToRight => "LR",
            LayoutDirection::BottomToTop => "BT",
            LayoutDirection::RightToLeft => "RL",
        }
    }
}

/// Shape visualizer
#[derive(Debug)]
pub struct ShapeVisualizer {
    config: VisualizerConfig,
}

impl ShapeVisualizer {
    /// Create a new visualizer
    pub fn new() -> Self {
        Self {
            config: VisualizerConfig::default(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: VisualizerConfig) -> Self {
        Self { config }
    }

    /// Export a single shape
    pub fn export(&self, shape: &Shape, format: ExportFormat) -> String {
        self.export_multiple(std::slice::from_ref(shape), format)
    }

    /// Export multiple shapes
    pub fn export_multiple(&self, shapes: &[Shape], format: ExportFormat) -> String {
        match format {
            ExportFormat::Dot => self.to_dot(shapes),
            ExportFormat::Mermaid => self.to_mermaid(shapes),
            ExportFormat::JsonSchema => self.to_json_schema(shapes),
            ExportFormat::Svg => self.to_svg(shapes),
            ExportFormat::PlantUml => self.to_plantuml(shapes),
            ExportFormat::D3Json => self.to_d3_json(shapes),
            ExportFormat::CytoscapeJson => self.to_cytoscape_json(shapes),
        }
    }

    /// Export to GraphViz DOT format
    pub fn to_dot(&self, shapes: &[Shape]) -> String {
        let mut dot = String::new();

        writeln!(dot, "digraph SHACL_Shapes {{").ok();
        writeln!(dot, "  rankdir={};", self.config.layout_direction.as_dot()).ok();
        writeln!(
            dot,
            "  node [fontname=\"{}\"];",
            self.config.shape_style.font_family
        )
        .ok();
        writeln!(
            dot,
            "  edge [fontname=\"{}\"];",
            self.config.shape_style.font_family
        )
        .ok();
        writeln!(dot).ok();

        // Style definitions
        writeln!(dot, "  // Node shape styles").ok();
        writeln!(
            dot,
            "  node [shape={}, style=filled, fillcolor=\"{}\", color=\"{}\", fontsize={}];",
            self.config.shape_style.shape,
            self.config.color_scheme.shape_fill,
            self.config.color_scheme.shape_border,
            self.config.shape_style.font_size
        )
        .ok();
        writeln!(dot).ok();

        for shape in shapes {
            self.shape_to_dot(&mut dot, shape);
        }

        // Add relationships between shapes
        let shape_ids: HashSet<_> = shapes.iter().map(|s| s.id.as_str()).collect();
        for shape in shapes {
            for (_, constraint) in &shape.constraints {
                // Check for sh:node or sh:class references
                if let crate::constraints::Constraint::Node(node_constraint) = constraint {
                    let ref_id = node_constraint.shape.as_str();
                    if shape_ids.contains(ref_id) {
                        let from_id = self.sanitize_id(&shape.id.0);
                        let to_id = self.sanitize_id(ref_id);
                        writeln!(
                            dot,
                            "  {} -> {} [label=\"sh:node\", style=dashed];",
                            from_id, to_id
                        )
                        .ok();
                    }
                }
            }
        }

        writeln!(dot, "}}").ok();
        dot
    }

    fn shape_to_dot(&self, dot: &mut String, shape: &Shape) {
        let shape_id = self.sanitize_id(&shape.id.0);
        let label = self.compact_iri(&shape.id.0);

        // Create shape node
        let shape_type_label = match shape.shape_type {
            ShapeType::NodeShape => "NodeShape",
            ShapeType::PropertyShape => "PropertyShape",
        };

        writeln!(dot, "  // Shape: {}", label).ok();
        writeln!(
            dot,
            "  {} [label=\"{} ({})\\n{}\"];",
            shape_id,
            label,
            shape_type_label,
            shape.label.as_deref().unwrap_or("")
        )
        .ok();

        // Add targets
        if self.config.show_targets && !shape.targets.is_empty() {
            for (idx, target) in shape.targets.iter().enumerate() {
                let target_id = format!("{}_target_{}", shape_id, idx);
                let target_label = self.target_label(target);

                writeln!(
                    dot,
                    "  {} [shape=ellipse, style=filled, fillcolor=\"{}\", label=\"{}\"];",
                    target_id, self.config.color_scheme.target_fill, target_label
                )
                .ok();
                writeln!(dot, "  {} -> {} [label=\"target\"];", shape_id, target_id).ok();
            }
        }

        // Add constraints
        if self.config.show_constraints && !shape.constraints.is_empty() {
            let constraints_id = format!("{}_constraints", shape_id);
            let mut constraint_labels = Vec::new();

            for (comp_id, constraint) in &shape.constraints {
                let constraint_label = self.constraint_label(comp_id, constraint);
                constraint_labels.push(constraint_label);
            }

            let constraints_label = constraint_labels.join("\\n");
            writeln!(
                dot,
                "  {} [shape=note, style=filled, fillcolor=\"{}\", label=\"{}\", fontsize={}];",
                constraints_id,
                self.config.color_scheme.constraint_fill,
                constraints_label,
                self.config.property_style.font_size
            )
            .ok();
            writeln!(
                dot,
                "  {} -> {} [style=dotted, label=\"constraints\"];",
                shape_id, constraints_id
            )
            .ok();
        }

        writeln!(dot).ok();
    }

    /// Export to Mermaid diagram syntax
    pub fn to_mermaid(&self, shapes: &[Shape]) -> String {
        let mut mermaid = String::new();

        writeln!(
            mermaid,
            "flowchart {}",
            self.config.layout_direction.as_mermaid()
        )
        .ok();

        for shape in shapes {
            self.shape_to_mermaid(&mut mermaid, shape);
        }

        mermaid
    }

    fn shape_to_mermaid(&self, mermaid: &mut String, shape: &Shape) {
        let shape_id = self.sanitize_id(&shape.id.0);
        let label = self.compact_iri(&shape.id.0);

        let shape_type = match shape.shape_type {
            ShapeType::NodeShape => "NodeShape",
            ShapeType::PropertyShape => "PropertyShape",
        };

        // Shape node
        writeln!(
            mermaid,
            "    {}[\"<b>{}</b><br/><small>{}</small>\"]",
            shape_id, label, shape_type
        )
        .ok();

        // Style the node
        writeln!(
            mermaid,
            "    style {} fill:{},stroke:{},stroke-width:{}px",
            shape_id,
            self.config.color_scheme.shape_fill,
            self.config.color_scheme.shape_border,
            self.config.shape_style.border_width
        )
        .ok();

        // Targets
        if self.config.show_targets {
            for (idx, target) in shape.targets.iter().enumerate() {
                let target_id = format!("{}_t{}", shape_id, idx);
                let target_label = self.target_label(target);

                writeln!(mermaid, "    {}((\"{}\"))", target_id, target_label).ok();
                writeln!(
                    mermaid,
                    "    style {} fill:{},stroke:#27ae60",
                    target_id, self.config.color_scheme.target_fill
                )
                .ok();
                writeln!(mermaid, "    {} -->|target| {}", shape_id, target_id).ok();
            }
        }

        // Constraints
        if self.config.show_constraints && !shape.constraints.is_empty() {
            let constraints_id = format!("{}_c", shape_id);
            let mut labels = Vec::new();

            for (comp_id, constraint) in &shape.constraints {
                labels.push(self.constraint_label(comp_id, constraint));
            }

            writeln!(
                mermaid,
                "    {}[\"{}\"]",
                constraints_id,
                labels.join("<br/>")
            )
            .ok();
            writeln!(
                mermaid,
                "    style {} fill:{},stroke:#f39c12",
                constraints_id, self.config.color_scheme.constraint_fill
            )
            .ok();
            writeln!(
                mermaid,
                "    {} -.->|constraints| {}",
                shape_id, constraints_id
            )
            .ok();
        }
    }

    /// Export to JSON Schema format (for web editors)
    pub fn to_json_schema(&self, shapes: &[Shape]) -> String {
        let mut schema = serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "shacl-shapes-schema",
            "type": "object",
            "title": "SHACL Shapes",
            "definitions": {}
        });

        let definitions = schema["definitions"].as_object_mut().unwrap();

        for shape in shapes {
            let shape_schema = self.shape_to_json_schema(shape);
            let shape_name = self.compact_iri(&shape.id.0).replace(':', "_");
            definitions.insert(shape_name, shape_schema);
        }

        serde_json::to_string_pretty(&schema).unwrap_or_default()
    }

    fn shape_to_json_schema(&self, shape: &Shape) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for (comp_id, constraint) in &shape.constraints {
            if let Some((prop_name, prop_schema, is_required)) =
                self.constraint_to_json_schema(comp_id, constraint)
            {
                properties.insert(prop_name.clone(), prop_schema);
                if is_required {
                    required.push(prop_name);
                }
            }
        }

        serde_json::json!({
            "type": "object",
            "title": shape.label.as_deref().unwrap_or(&self.compact_iri(&shape.id.0)),
            "description": shape.description.clone().unwrap_or_default(),
            "properties": properties,
            "required": required
        })
    }

    fn constraint_to_json_schema(
        &self,
        comp_id: &ConstraintComponentId,
        constraint: &crate::constraints::Constraint,
    ) -> Option<(String, serde_json::Value, bool)> {
        // This is a simplified mapping - in production you'd handle all constraint types
        let comp_name = comp_id.as_str();

        match constraint {
            crate::constraints::Constraint::Datatype(dt) => {
                let dt_iri = dt.datatype_iri.as_str();
                let json_type = if dt_iri.contains("string") || dt_iri.contains("date") {
                    "string"
                } else if dt_iri.contains("integer") {
                    "integer"
                } else if dt_iri.contains("decimal") || dt_iri.contains("float") {
                    "number"
                } else if dt_iri.contains("boolean") {
                    "boolean"
                } else {
                    "string"
                };

                let mut schema = serde_json::json!({ "type": json_type });
                if dt_iri.contains("date") {
                    schema["format"] = serde_json::json!("date");
                }

                Some((comp_name.to_string(), schema, false))
            }
            crate::constraints::Constraint::MinCount(mc) => Some((
                comp_name.to_string(),
                serde_json::json!({}),
                mc.min_count >= 1,
            )),
            crate::constraints::Constraint::Pattern(pat) => Some((
                comp_name.to_string(),
                serde_json::json!({
                    "type": "string",
                    "pattern": pat.pattern
                }),
                false,
            )),
            crate::constraints::Constraint::MinLength(ml) => Some((
                comp_name.to_string(),
                serde_json::json!({
                    "type": "string",
                    "minLength": ml.min_length
                }),
                false,
            )),
            crate::constraints::Constraint::MaxLength(ml) => Some((
                comp_name.to_string(),
                serde_json::json!({
                    "type": "string",
                    "maxLength": ml.max_length
                }),
                false,
            )),
            _ => None,
        }
    }

    /// Export to SVG diagram
    pub fn to_svg(&self, shapes: &[Shape]) -> String {
        let mut svg = String::new();

        // Calculate dimensions
        let shape_height = 80;
        let shape_width = 200;
        let padding = 20;
        let total_height = shapes.len() * (shape_height + padding) + padding;
        let total_width = shape_width + padding * 2;

        writeln!(
            svg,
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {} {}"
     width="{}" height="{}">"#,
            total_width, total_height, total_width, total_height
        )
        .ok();

        // Add style
        writeln!(
            svg,
            r#"<style>
    .shape-rect {{ fill: {}; stroke: {}; stroke-width: {}; }}
    .shape-text {{ font-family: {}; font-size: {}px; fill: {}; }}
    .constraint-text {{ font-family: {}; font-size: {}px; fill: #666; }}
</style>"#,
            self.config.color_scheme.shape_fill,
            self.config.color_scheme.shape_border,
            self.config.shape_style.border_width,
            self.config.shape_style.font_family,
            self.config.shape_style.font_size,
            self.config.color_scheme.text_color,
            self.config.property_style.font_family,
            self.config.property_style.font_size
        )
        .ok();

        // Draw shapes
        for (idx, shape) in shapes.iter().enumerate() {
            let y = padding + idx * (shape_height + padding);
            self.shape_to_svg(&mut svg, shape, padding, y, shape_width, shape_height);
        }

        writeln!(svg, "</svg>").ok();
        svg
    }

    fn shape_to_svg(
        &self,
        svg: &mut String,
        shape: &Shape,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) {
        let label = self.compact_iri(&shape.id.0);
        let shape_type = match shape.shape_type {
            ShapeType::NodeShape => "NodeShape",
            ShapeType::PropertyShape => "PropertyShape",
        };

        // Rectangle
        writeln!(
            svg,
            r#"  <rect class="shape-rect" x="{}" y="{}" width="{}" height="{}" rx="5" />"#,
            x, y, width, height
        )
        .ok();

        // Title
        writeln!(
            svg,
            r#"  <text class="shape-text" x="{}" y="{}" text-anchor="middle">{}</text>"#,
            x + width / 2,
            y + 25,
            label
        )
        .ok();

        // Type
        writeln!(
            svg,
            r#"  <text class="constraint-text" x="{}" y="{}" text-anchor="middle">({})</text>"#,
            x + width / 2,
            y + 45,
            shape_type
        )
        .ok();

        // Constraint count
        if !shape.constraints.is_empty() {
            writeln!(
                svg,
                r#"  <text class="constraint-text" x="{}" y="{}" text-anchor="middle">{} constraints</text>"#,
                x + width / 2,
                y + 65,
                shape.constraints.len()
            )
            .ok();
        }
    }

    /// Export to PlantUML format
    pub fn to_plantuml(&self, shapes: &[Shape]) -> String {
        let mut uml = String::new();

        writeln!(uml, "@startuml").ok();
        writeln!(uml, "skinparam class {{").ok();
        writeln!(
            uml,
            "  BackgroundColor {}",
            self.config.color_scheme.shape_fill
        )
        .ok();
        writeln!(
            uml,
            "  BorderColor {}",
            self.config.color_scheme.shape_border
        )
        .ok();
        writeln!(uml, "}}").ok();
        writeln!(uml).ok();

        for shape in shapes {
            let label = self.compact_iri(&shape.id.0).replace(':', "_");
            let shape_type = match shape.shape_type {
                ShapeType::NodeShape => "<<NodeShape>>",
                ShapeType::PropertyShape => "<<PropertyShape>>",
            };

            writeln!(uml, "class \"{}\" {} {{", label, shape_type).ok();

            for (comp_id, constraint) in &shape.constraints {
                let constraint_label = self.constraint_label(comp_id, constraint);
                writeln!(uml, "  + {}", constraint_label).ok();
            }

            writeln!(uml, "}}").ok();
            writeln!(uml).ok();
        }

        writeln!(uml, "@enduml").ok();
        uml
    }

    /// Export to D3.js JSON format
    pub fn to_d3_json(&self, shapes: &[Shape]) -> String {
        let mut nodes = Vec::new();
        let mut links = Vec::new();

        for shape in shapes {
            let node = serde_json::json!({
                "id": shape.id.as_str(),
                "label": self.compact_iri(&shape.id.0),
                "type": match shape.shape_type {
                    ShapeType::NodeShape => "node",
                    ShapeType::PropertyShape => "property"
                },
                "constraints": shape.constraints.len(),
                "targets": shape.targets.len()
            });
            nodes.push(node);

            // Add links for shape references
            for (_, constraint) in &shape.constraints {
                if let crate::constraints::Constraint::Node(nc) = constraint {
                    links.push(serde_json::json!({
                        "source": shape.id.as_str(),
                        "target": nc.shape.as_str(),
                        "type": "sh:node"
                    }));
                }
            }
        }

        let result = serde_json::json!({
            "nodes": nodes,
            "links": links
        });

        serde_json::to_string_pretty(&result).unwrap_or_default()
    }

    /// Export to Cytoscape.js JSON format
    pub fn to_cytoscape_json(&self, shapes: &[Shape]) -> String {
        let mut elements = Vec::new();

        for shape in shapes {
            // Add node
            let node = serde_json::json!({
                "data": {
                    "id": shape.id.as_str(),
                    "label": self.compact_iri(&shape.id.0),
                    "type": match shape.shape_type {
                        ShapeType::NodeShape => "nodeShape",
                        ShapeType::PropertyShape => "propertyShape"
                    }
                },
                "classes": match shape.shape_type {
                    ShapeType::NodeShape => "nodeShape",
                    ShapeType::PropertyShape => "propertyShape"
                }
            });
            elements.push(node);

            // Add edges for references
            for (_, constraint) in &shape.constraints {
                if let crate::constraints::Constraint::Node(nc) = constraint {
                    let edge = serde_json::json!({
                        "data": {
                            "source": shape.id.as_str(),
                            "target": nc.shape.as_str(),
                            "label": "sh:node"
                        },
                        "classes": "reference"
                    });
                    elements.push(edge);
                }
            }
        }

        let result = serde_json::json!({
            "elements": elements,
            "style": [
                {
                    "selector": "node.nodeShape",
                    "style": {
                        "background-color": self.config.color_scheme.shape_fill,
                        "border-color": self.config.color_scheme.shape_border,
                        "label": "data(label)"
                    }
                },
                {
                    "selector": "node.propertyShape",
                    "style": {
                        "background-color": self.config.color_scheme.property_fill,
                        "border-color": self.config.color_scheme.property_border,
                        "shape": "rectangle",
                        "label": "data(label)"
                    }
                },
                {
                    "selector": "edge",
                    "style": {
                        "line-color": self.config.color_scheme.edge_color,
                        "target-arrow-color": self.config.color_scheme.edge_color,
                        "target-arrow-shape": "triangle",
                        "label": "data(label)"
                    }
                }
            ]
        });

        serde_json::to_string_pretty(&result).unwrap_or_default()
    }

    /// Get interactive editor protocol JSON
    pub fn to_editor_protocol(&self, shapes: &[Shape]) -> String {
        let editor_shapes: Vec<_> = shapes
            .iter()
            .map(|s| self.shape_to_editor_json(s))
            .collect();

        let result = serde_json::json!({
            "version": "1.0",
            "protocol": "shacl-editor",
            "shapes": editor_shapes,
            "config": {
                "prefixes": self.config.prefixes,
                "colorScheme": self.config.color_scheme
            }
        });

        serde_json::to_string_pretty(&result).unwrap_or_default()
    }

    fn shape_to_editor_json(&self, shape: &Shape) -> serde_json::Value {
        let targets: Vec<_> = shape
            .targets
            .iter()
            .map(|t| self.target_to_editor_json(t))
            .collect();

        let constraints: Vec<_> = shape
            .constraints
            .iter()
            .map(|(comp_id, c)| self.constraint_to_editor_json(comp_id, c))
            .collect();

        serde_json::json!({
            "id": shape.id.as_str(),
            "label": self.compact_iri(&shape.id.0),
            "humanLabel": shape.label,
            "description": shape.description,
            "type": match shape.shape_type {
                ShapeType::NodeShape => "NodeShape",
                ShapeType::PropertyShape => "PropertyShape"
            },
            "targets": targets,
            "constraints": constraints,
            "severity": format!("{:?}", shape.severity),
            "deactivated": shape.deactivated,
            "metadata": {
                "groups": shape.groups,
                "order": shape.order,
                "extends": shape.extends.iter().map(|s| s.as_str()).collect::<Vec<_>>()
            }
        })
    }

    fn target_to_editor_json(&self, target: &Target) -> serde_json::Value {
        match target {
            Target::Class(node) => serde_json::json!({
                "type": "class",
                "value": node.as_str(),
                "label": self.compact_iri(node.as_str())
            }),
            Target::Node(node) => serde_json::json!({
                "type": "node",
                "value": node.to_string(),
                "label": self.compact_iri(&node.to_string())
            }),
            Target::SubjectsOf(node) => serde_json::json!({
                "type": "subjectsOf",
                "value": node.as_str(),
                "label": self.compact_iri(node.as_str())
            }),
            Target::ObjectsOf(node) => serde_json::json!({
                "type": "objectsOf",
                "value": node.as_str(),
                "label": self.compact_iri(node.as_str())
            }),
            Target::Sparql(sparql_target) => serde_json::json!({
                "type": "sparql",
                "value": &sparql_target.query,
                "label": "SPARQL Target"
            }),
            Target::Implicit(node) => serde_json::json!({
                "type": "implicit",
                "value": node.as_str(),
                "label": self.compact_iri(node.as_str())
            }),
            Target::Union(union_target) => serde_json::json!({
                "type": "union",
                "value": format!("Union of {} targets", union_target.targets.len()),
                "label": "Union Target"
            }),
            Target::Intersection(intersection_target) => serde_json::json!({
                "type": "intersection",
                "value": format!("Intersection of {} targets", intersection_target.targets.len()),
                "label": "Intersection Target"
            }),
            Target::Difference(_diff_target) => serde_json::json!({
                "type": "difference",
                "value": "Difference Target",
                "label": format!("Difference (primary - exclusion)"),
            }),
            Target::Conditional(_cond_target) => serde_json::json!({
                "type": "conditional",
                "value": "Conditional Target",
                "label": format!("Conditional (with condition)")
            }),
            Target::Hierarchical(hier_target) => serde_json::json!({
                "type": "hierarchical",
                "value": format!("{:?}", hier_target.relationship),
                "label": format!("Hierarchical (depth: {:?})", hier_target.max_depth)
            }),
            Target::PathBased(path_target) => serde_json::json!({
                "type": "pathBased",
                "value": "Path-based Target",
                "label": format!("Path-based (direction: {:?})", path_target.direction)
            }),
        }
    }

    fn constraint_to_editor_json(
        &self,
        comp_id: &ConstraintComponentId,
        constraint: &crate::constraints::Constraint,
    ) -> serde_json::Value {
        serde_json::json!({
            "componentId": comp_id.as_str(),
            "label": self.compact_iri(comp_id.as_str()),
            "description": self.constraint_label(comp_id, constraint)
        })
    }

    // Helper methods

    fn sanitize_id(&self, id: &str) -> String {
        id.chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect()
    }

    fn compact_iri(&self, iri: &str) -> String {
        if !self.config.use_prefixes {
            return iri.to_string();
        }

        for (prefix, namespace) in &self.config.prefixes {
            if iri.starts_with(namespace) {
                let local = &iri[namespace.len()..];
                return format!("{}:{}", prefix, local);
            }
        }

        // Try to get just the local name
        if let Some(pos) = iri.rfind('#') {
            return iri[pos + 1..].to_string();
        }
        if let Some(pos) = iri.rfind('/') {
            return iri[pos + 1..].to_string();
        }

        iri.to_string()
    }

    fn target_label(&self, target: &Target) -> String {
        match target {
            Target::Class(node) => format!("class: {}", self.compact_iri(node.as_str())),
            Target::Node(node) => format!("node: {}", self.compact_iri(&node.to_string())),
            Target::SubjectsOf(node) => {
                format!("subjectsOf: {}", self.compact_iri(node.as_str()))
            }
            Target::ObjectsOf(node) => {
                format!("objectsOf: {}", self.compact_iri(node.as_str()))
            }
            Target::Sparql(_) => "SPARQL target".to_string(),
            Target::Implicit(node) => format!("implicit: {}", self.compact_iri(node.as_str())),
            Target::Union(union_target) => format!("union: {} targets", union_target.targets.len()),
            Target::Intersection(intersection_target) => {
                format!(
                    "intersection: {} targets",
                    intersection_target.targets.len()
                )
            }
            Target::Difference(_) => "difference target".to_string(),
            Target::Conditional(_) => "conditional target".to_string(),
            Target::Hierarchical(hier_target) => {
                format!("hierarchical: {:?}", hier_target.relationship)
            }
            Target::PathBased(_) => "path-based target".to_string(),
        }
    }

    fn constraint_label(
        &self,
        comp_id: &ConstraintComponentId,
        constraint: &crate::constraints::Constraint,
    ) -> String {
        let comp_name = self.compact_iri(comp_id.as_str());

        match constraint {
            crate::constraints::Constraint::MinCount(mc) => {
                format!("{} = {}", comp_name, mc.min_count)
            }
            crate::constraints::Constraint::MaxCount(mc) => {
                format!("{} = {}", comp_name, mc.max_count)
            }
            crate::constraints::Constraint::Datatype(dt) => {
                format!(
                    "{} = {}",
                    comp_name,
                    self.compact_iri(dt.datatype_iri.as_str())
                )
            }
            crate::constraints::Constraint::Pattern(pat) => {
                format!("{} = /{}/", comp_name, pat.pattern)
            }
            crate::constraints::Constraint::MinLength(ml) => {
                format!("{} = {}", comp_name, ml.min_length)
            }
            crate::constraints::Constraint::MaxLength(ml) => {
                format!("{} = {}", comp_name, ml.max_length)
            }
            _ => comp_name,
        }
    }
}

impl Default for ShapeVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::cardinality_constraints::MinCountConstraint;
    use crate::constraints::value_constraints::DatatypeConstraint;
    use crate::ShapeId;
    use oxirs_core::NamedNode;

    fn create_test_shape() -> Shape {
        let mut shape = Shape::new(
            ShapeId::new("http://example.org/PersonShape"),
            ShapeType::NodeShape,
        );
        shape.label = Some("Person Shape".to_string());
        shape.targets.push(Target::Class(
            NamedNode::new("http://xmlns.com/foaf/0.1/Person").unwrap(),
        ));
        shape.add_constraint(
            ConstraintComponentId::new("sh:minCount"),
            crate::constraints::Constraint::MinCount(MinCountConstraint { min_count: 1 }),
        );
        shape.add_constraint(
            ConstraintComponentId::new("sh:datatype"),
            crate::constraints::Constraint::Datatype(DatatypeConstraint {
                datatype_iri: NamedNode::new("http://www.w3.org/2001/XMLSchema#string").unwrap(),
            }),
        );
        shape
    }

    #[test]
    fn test_dot_export() {
        let visualizer = ShapeVisualizer::new();
        let shape = create_test_shape();

        let dot = visualizer.to_dot(&[shape]);
        assert!(dot.contains("digraph"));
        assert!(dot.contains("PersonShape"));
    }

    #[test]
    fn test_mermaid_export() {
        let visualizer = ShapeVisualizer::new();
        let shape = create_test_shape();

        let mermaid = visualizer.to_mermaid(&[shape]);
        assert!(mermaid.contains("flowchart"));
        assert!(mermaid.contains("PersonShape"));
    }

    #[test]
    fn test_json_schema_export() {
        let visualizer = ShapeVisualizer::new();
        let shape = create_test_shape();

        let json_schema = visualizer.to_json_schema(&[shape]);
        assert!(json_schema.contains("$schema"));
        assert!(json_schema.contains("definitions"));
    }

    #[test]
    fn test_svg_export() {
        let visualizer = ShapeVisualizer::new();
        let shape = create_test_shape();

        let svg = visualizer.to_svg(&[shape]);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("PersonShape"));
    }

    #[test]
    fn test_plantuml_export() {
        let visualizer = ShapeVisualizer::new();
        let shape = create_test_shape();

        let uml = visualizer.to_plantuml(&[shape]);
        assert!(uml.contains("@startuml"));
        assert!(uml.contains("@enduml"));
    }

    #[test]
    fn test_d3_json_export() {
        let visualizer = ShapeVisualizer::new();
        let shape = create_test_shape();

        let d3 = visualizer.to_d3_json(&[shape]);
        assert!(d3.contains("nodes"));
        assert!(d3.contains("links"));
    }

    #[test]
    fn test_cytoscape_export() {
        let visualizer = ShapeVisualizer::new();
        let shape = create_test_shape();

        let cytoscape = visualizer.to_cytoscape_json(&[shape]);
        assert!(cytoscape.contains("elements"));
        assert!(cytoscape.contains("style"));
    }

    #[test]
    fn test_editor_protocol() {
        let visualizer = ShapeVisualizer::new();
        let shape = create_test_shape();

        let protocol = visualizer.to_editor_protocol(&[shape]);
        assert!(protocol.contains("shacl-editor"));
        assert!(protocol.contains("shapes"));
    }

    #[test]
    fn test_compact_iri() {
        let visualizer = ShapeVisualizer::new();

        let compacted = visualizer.compact_iri("http://xmlns.com/foaf/0.1/Person");
        assert_eq!(compacted, "foaf:Person");

        let compacted = visualizer.compact_iri("http://example.org/test");
        assert_eq!(compacted, "ex:test");
    }
}
