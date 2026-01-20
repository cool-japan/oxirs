//! Visual Schema Designer
//!
//! This module provides a visual schema design framework for creating,
//! editing, and visualizing GraphQL schemas with an interactive interface.
//!
//! ## Features
//!
//! - **Schema Visualization**: Generate visual representations of schemas
//! - **Interactive Editing**: Add, modify, and remove types and fields
//! - **Relationship Mapping**: Visualize relationships between types
//! - **Validation**: Real-time schema validation
//! - **Import/Export**: Support for SDL import and export
//! - **Layout Algorithms**: Automatic layout for clean visualization
//!
//! ## Usage
//!
//! ```rust,ignore
//! use oxirs_gql::visual_schema_designer::{SchemaDesigner, SchemaNode, LayoutOptions};
//!
//! let designer = SchemaDesigner::new();
//! designer.import_sdl(sdl_string).await?;
//!
//! // Get visualization data
//! let graph = designer.get_visualization().await;
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Node type in the schema graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// Object type
    ObjectType,
    /// Interface type
    InterfaceType,
    /// Input type
    InputType,
    /// Enum type
    EnumType,
    /// Union type
    UnionType,
    /// Scalar type
    ScalarType,
    /// Directive
    Directive,
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeType::ObjectType => write!(f, "type"),
            NodeType::InterfaceType => write!(f, "interface"),
            NodeType::InputType => write!(f, "input"),
            NodeType::EnumType => write!(f, "enum"),
            NodeType::UnionType => write!(f, "union"),
            NodeType::ScalarType => write!(f, "scalar"),
            NodeType::Directive => write!(f, "directive"),
        }
    }
}

/// Edge type in the schema graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Field reference
    Field,
    /// Implements interface
    Implements,
    /// Union member
    UnionMember,
    /// Input argument
    Argument,
    /// Directive application
    DirectiveUse,
}

/// Position in the visual canvas
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Position {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl Position {
    /// Create a new position
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Calculate distance to another position
    pub fn distance_to(&self, other: &Position) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Size dimensions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Size {
    /// Width
    pub width: f64,
    /// Height
    pub height: f64,
}

impl Default for Size {
    fn default() -> Self {
        Self {
            width: 200.0,
            height: 100.0,
        }
    }
}

/// Schema node (type definition)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaNode {
    /// Unique identifier
    pub id: String,
    /// Node name
    pub name: String,
    /// Node type
    pub node_type: NodeType,
    /// Description
    pub description: Option<String>,
    /// Fields (for object, interface, input types)
    pub fields: Vec<FieldDefinition>,
    /// Enum values (for enum types)
    pub enum_values: Vec<EnumValue>,
    /// Union members (for union types)
    pub union_members: Vec<String>,
    /// Implemented interfaces
    pub implements: Vec<String>,
    /// Applied directives
    pub directives: Vec<DirectiveUse>,
    /// Visual position
    pub position: Position,
    /// Visual size
    pub size: Size,
    /// Whether node is collapsed in view
    pub collapsed: bool,
    /// Whether node is selected
    pub selected: bool,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl SchemaNode {
    /// Create a new schema node
    pub fn new(id: &str, name: &str, node_type: NodeType) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            node_type,
            description: None,
            fields: Vec::new(),
            enum_values: Vec::new(),
            union_members: Vec::new(),
            implements: Vec::new(),
            directives: Vec::new(),
            position: Position::default(),
            size: Size::default(),
            collapsed: false,
            selected: false,
            metadata: HashMap::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Add a field
    pub fn with_field(mut self, field: FieldDefinition) -> Self {
        self.fields.push(field);
        self
    }

    /// Set position
    pub fn with_position(mut self, x: f64, y: f64) -> Self {
        self.position = Position::new(x, y);
        self
    }

    /// Calculate visual height based on fields
    pub fn calculate_height(&self) -> f64 {
        let base_height = 40.0; // Header
        let field_height = 24.0;
        let padding = 16.0;

        if self.collapsed {
            return base_height;
        }

        let content_height = match self.node_type {
            NodeType::ObjectType | NodeType::InterfaceType | NodeType::InputType => {
                self.fields.len() as f64 * field_height
            }
            NodeType::EnumType => self.enum_values.len() as f64 * field_height,
            NodeType::UnionType => self.union_members.len() as f64 * field_height,
            _ => 0.0,
        };

        base_height + content_height + padding
    }
}

/// Field definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    /// Field name
    pub name: String,
    /// Field type (as string)
    pub field_type: String,
    /// Whether field is nullable
    pub nullable: bool,
    /// Whether field is a list
    pub is_list: bool,
    /// Description
    pub description: Option<String>,
    /// Arguments (for object type fields)
    pub arguments: Vec<ArgumentDefinition>,
    /// Applied directives
    pub directives: Vec<DirectiveUse>,
    /// Default value (for input fields)
    pub default_value: Option<String>,
    /// Whether field is deprecated
    pub deprecated: bool,
    /// Deprecation reason
    pub deprecation_reason: Option<String>,
}

impl FieldDefinition {
    /// Create a new field definition
    pub fn new(name: &str, field_type: &str) -> Self {
        Self {
            name: name.to_string(),
            field_type: field_type.to_string(),
            nullable: true,
            is_list: false,
            description: None,
            arguments: Vec::new(),
            directives: Vec::new(),
            default_value: None,
            deprecated: false,
            deprecation_reason: None,
        }
    }

    /// Set nullable
    pub fn non_null(mut self) -> Self {
        self.nullable = false;
        self
    }

    /// Set as list
    pub fn list(mut self) -> Self {
        self.is_list = true;
        self
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Get type string for display
    pub fn type_string(&self) -> String {
        let mut s = self.field_type.clone();
        if self.is_list {
            s = format!("[{}]", s);
        }
        if !self.nullable {
            s = format!("{}!", s);
        }
        s
    }
}

/// Argument definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentDefinition {
    /// Argument name
    pub name: String,
    /// Argument type
    pub arg_type: String,
    /// Whether nullable
    pub nullable: bool,
    /// Default value
    pub default_value: Option<String>,
    /// Description
    pub description: Option<String>,
}

impl ArgumentDefinition {
    /// Create a new argument
    pub fn new(name: &str, arg_type: &str) -> Self {
        Self {
            name: name.to_string(),
            arg_type: arg_type.to_string(),
            nullable: true,
            default_value: None,
            description: None,
        }
    }
}

/// Enum value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumValue {
    /// Value name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Whether deprecated
    pub deprecated: bool,
    /// Deprecation reason
    pub deprecation_reason: Option<String>,
}

impl EnumValue {
    /// Create a new enum value
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            deprecated: false,
            deprecation_reason: None,
        }
    }
}

/// Directive usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectiveUse {
    /// Directive name
    pub name: String,
    /// Arguments
    pub arguments: HashMap<String, String>,
}

/// Schema edge (relationship)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaEdge {
    /// Unique identifier
    pub id: String,
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Source field name (for field edges)
    pub source_field: Option<String>,
    /// Edge type
    pub edge_type: EdgeType,
    /// Label
    pub label: Option<String>,
    /// Whether edge is highlighted
    pub highlighted: bool,
}

impl SchemaEdge {
    /// Create a new edge
    pub fn new(id: &str, source: &str, target: &str, edge_type: EdgeType) -> Self {
        Self {
            id: id.to_string(),
            source: source.to_string(),
            target: target.to_string(),
            source_field: None,
            edge_type,
            label: None,
            highlighted: false,
        }
    }

    /// Set source field
    pub fn with_source_field(mut self, field: &str) -> Self {
        self.source_field = Some(field.to_string());
        self
    }

    /// Set label
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

/// Schema graph (visualization data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaGraph {
    /// Nodes
    pub nodes: Vec<SchemaNode>,
    /// Edges
    pub edges: Vec<SchemaEdge>,
    /// Graph width
    pub width: f64,
    /// Graph height
    pub height: f64,
    /// Zoom level
    pub zoom: f64,
    /// Pan offset
    pub pan: Position,
}

impl Default for SchemaGraph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            width: 1000.0,
            height: 800.0,
            zoom: 1.0,
            pan: Position::default(),
        }
    }
}

/// Layout algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LayoutAlgorithm {
    /// Force-directed layout
    #[default]
    ForceDirected,
    /// Hierarchical (top-down)
    Hierarchical,
    /// Circular layout
    Circular,
    /// Grid layout
    Grid,
    /// Manual (no auto-layout)
    Manual,
}

/// Layout options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutOptions {
    /// Layout algorithm
    pub algorithm: LayoutAlgorithm,
    /// Node spacing
    pub node_spacing: f64,
    /// Layer spacing (for hierarchical)
    pub layer_spacing: f64,
    /// Iterations (for force-directed)
    pub iterations: u32,
    /// Animation duration (ms)
    pub animation_duration: u64,
    /// Group by type
    pub group_by_type: bool,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            algorithm: LayoutAlgorithm::default(),
            node_spacing: 50.0,
            layer_spacing: 100.0,
            iterations: 100,
            animation_duration: 300,
            group_by_type: true,
        }
    }
}

/// Edit operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EditOperation {
    /// Add a node
    AddNode { node: SchemaNode },
    /// Remove a node
    RemoveNode { node_id: String },
    /// Update a node
    UpdateNode { node: SchemaNode },
    /// Add a field to a node
    AddField {
        node_id: String,
        field: FieldDefinition,
    },
    /// Remove a field from a node
    RemoveField { node_id: String, field_name: String },
    /// Update a field
    UpdateField {
        node_id: String,
        field: FieldDefinition,
    },
    /// Add an edge
    AddEdge { edge: SchemaEdge },
    /// Remove an edge
    RemoveEdge { edge_id: String },
    /// Move a node
    MoveNode { node_id: String, position: Position },
    /// Batch operation
    Batch { operations: Vec<EditOperation> },
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether schema is valid
    pub valid: bool,
    /// Errors
    pub errors: Vec<ValidationError>,
    /// Warnings
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Node ID (if applicable)
    pub node_id: Option<String>,
    /// Field name (if applicable)
    pub field_name: Option<String>,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning code
    pub code: String,
    /// Warning message
    pub message: String,
    /// Node ID (if applicable)
    pub node_id: Option<String>,
}

/// Designer state
struct DesignerState {
    /// Schema graph
    graph: SchemaGraph,
    /// Node index
    node_index: HashMap<String, usize>,
    /// Edge index
    edge_index: HashMap<String, usize>,
    /// Edit history (for undo)
    history: Vec<EditOperation>,
    /// History position
    history_position: usize,
    /// Layout options
    layout_options: LayoutOptions,
}

impl DesignerState {
    fn new() -> Self {
        Self {
            graph: SchemaGraph::default(),
            node_index: HashMap::new(),
            edge_index: HashMap::new(),
            history: Vec::new(),
            history_position: 0,
            layout_options: LayoutOptions::default(),
        }
    }

    fn rebuild_indices(&mut self) {
        self.node_index.clear();
        for (i, node) in self.graph.nodes.iter().enumerate() {
            self.node_index.insert(node.id.clone(), i);
        }

        self.edge_index.clear();
        for (i, edge) in self.graph.edges.iter().enumerate() {
            self.edge_index.insert(edge.id.clone(), i);
        }
    }
}

/// Visual Schema Designer
///
/// Provides an interface for creating and editing GraphQL schemas
/// with visual representation and validation.
pub struct SchemaDesigner {
    /// Internal state
    state: Arc<RwLock<DesignerState>>,
}

impl SchemaDesigner {
    /// Create a new schema designer
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(DesignerState::new())),
        }
    }

    /// Get the current schema graph
    pub async fn get_visualization(&self) -> SchemaGraph {
        let state = self.state.read().await;
        state.graph.clone()
    }

    /// Add a node
    pub async fn add_node(&self, mut node: SchemaNode) -> Result<String> {
        let node_id = if node.id.is_empty() {
            uuid::Uuid::new_v4().to_string()
        } else {
            node.id.clone()
        };
        node.id = node_id.clone();

        let mut state = self.state.write().await;

        // Check for duplicate name
        if state.graph.nodes.iter().any(|n| n.name == node.name) {
            return Err(anyhow!("Type '{}' already exists", node.name));
        }

        // Calculate size
        node.size.height = node.calculate_height();

        state.graph.nodes.push(node.clone());
        let node_index_value = state.graph.nodes.len() - 1;
        state.node_index.insert(node_id.clone(), node_index_value);

        // Record in history
        let history_pos = state.history_position;
        state.history.truncate(history_pos);
        state.history.push(EditOperation::AddNode { node });
        state.history_position = state.history.len();

        Ok(node_id)
    }

    /// Remove a node
    pub async fn remove_node(&self, node_id: &str) -> Result<()> {
        let mut state = self.state.write().await;

        let Some(&index) = state.node_index.get(node_id) else {
            return Err(anyhow!("Node '{}' not found", node_id));
        };

        let _node = state.graph.nodes.remove(index);

        // Remove related edges
        state
            .graph
            .edges
            .retain(|e| e.source != node_id && e.target != node_id);

        // Rebuild indices
        state.rebuild_indices();

        // Record in history
        let history_pos = state.history_position;
        state.history.truncate(history_pos);
        state.history.push(EditOperation::RemoveNode {
            node_id: node_id.to_string(),
        });
        state.history_position = state.history.len();

        Ok(())
    }

    /// Update a node
    pub async fn update_node(&self, node: SchemaNode) -> Result<()> {
        let mut state = self.state.write().await;

        let Some(&index) = state.node_index.get(&node.id) else {
            return Err(anyhow!("Node '{}' not found", node.id));
        };

        let mut updated_node = node.clone();
        updated_node.size.height = updated_node.calculate_height();

        state.graph.nodes[index] = updated_node;

        // Record in history
        let history_pos = state.history_position;
        state.history.truncate(history_pos);
        state.history.push(EditOperation::UpdateNode { node });
        state.history_position = state.history.len();

        Ok(())
    }

    /// Add a field to a node
    pub async fn add_field(&self, node_id: &str, field: FieldDefinition) -> Result<()> {
        let mut state = self.state.write().await;

        let Some(&index) = state.node_index.get(node_id) else {
            return Err(anyhow!("Node '{}' not found", node_id));
        };

        // Check for duplicate field name
        if state.graph.nodes[index]
            .fields
            .iter()
            .any(|f| f.name == field.name)
        {
            return Err(anyhow!(
                "Field '{}' already exists in '{}'",
                field.name,
                node_id
            ));
        }

        state.graph.nodes[index].fields.push(field.clone());
        state.graph.nodes[index].size.height = state.graph.nodes[index].calculate_height();

        // Create edge if field references another type
        let target_type = field
            .field_type
            .trim_matches(|c| c == '[' || c == ']' || c == '!');
        if let Some(target_node) = state.graph.nodes.iter().find(|n| n.name == target_type) {
            let edge = SchemaEdge::new(
                &uuid::Uuid::new_v4().to_string(),
                node_id,
                &target_node.id,
                EdgeType::Field,
            )
            .with_source_field(&field.name);
            state.graph.edges.push(edge);
            state.rebuild_indices();
        }

        // Record in history
        let history_pos = state.history_position;
        state.history.truncate(history_pos);
        state.history.push(EditOperation::AddField {
            node_id: node_id.to_string(),
            field,
        });
        state.history_position = state.history.len();

        Ok(())
    }

    /// Remove a field from a node
    pub async fn remove_field(&self, node_id: &str, field_name: &str) -> Result<()> {
        let mut state = self.state.write().await;

        let Some(&index) = state.node_index.get(node_id) else {
            return Err(anyhow!("Node '{}' not found", node_id));
        };

        state.graph.nodes[index]
            .fields
            .retain(|f| f.name != field_name);
        state.graph.nodes[index].size.height = state.graph.nodes[index].calculate_height();

        // Remove related edges
        state
            .graph
            .edges
            .retain(|e| e.source != node_id || e.source_field.as_deref() != Some(field_name));
        state.rebuild_indices();

        // Record in history
        let history_pos = state.history_position;
        state.history.truncate(history_pos);
        state.history.push(EditOperation::RemoveField {
            node_id: node_id.to_string(),
            field_name: field_name.to_string(),
        });
        state.history_position = state.history.len();

        Ok(())
    }

    /// Move a node
    pub async fn move_node(&self, node_id: &str, position: Position) -> Result<()> {
        let mut state = self.state.write().await;

        let Some(&index) = state.node_index.get(node_id) else {
            return Err(anyhow!("Node '{}' not found", node_id));
        };

        state.graph.nodes[index].position = position;

        Ok(())
    }

    /// Apply layout algorithm
    pub async fn apply_layout(&self) -> Result<()> {
        let mut state = self.state.write().await;

        match state.layout_options.algorithm {
            LayoutAlgorithm::ForceDirected => {
                self.apply_force_directed_layout(&mut state)?;
            }
            LayoutAlgorithm::Hierarchical => {
                self.apply_hierarchical_layout(&mut state)?;
            }
            LayoutAlgorithm::Circular => {
                self.apply_circular_layout(&mut state)?;
            }
            LayoutAlgorithm::Grid => {
                self.apply_grid_layout(&mut state)?;
            }
            LayoutAlgorithm::Manual => {
                // No layout changes
            }
        }

        Ok(())
    }

    fn apply_force_directed_layout(&self, state: &mut DesignerState) -> Result<()> {
        let node_count = state.graph.nodes.len();
        if node_count == 0 {
            return Ok(());
        }

        // Initialize positions randomly if not set
        for node in &mut state.graph.nodes {
            if node.position.x == 0.0 && node.position.y == 0.0 {
                node.position.x = fastrand::f64() * 800.0;
                node.position.y = fastrand::f64() * 600.0;
            }
        }

        let spacing = state.layout_options.node_spacing;
        let iterations = state.layout_options.iterations;

        // Simple force-directed simulation
        for _ in 0..iterations {
            let mut forces: Vec<Position> = vec![Position::default(); node_count];

            // Repulsion between all nodes
            for i in 0..node_count {
                for j in (i + 1)..node_count {
                    let dx = state.graph.nodes[j].position.x - state.graph.nodes[i].position.x;
                    let dy = state.graph.nodes[j].position.y - state.graph.nodes[i].position.y;
                    let dist = (dx * dx + dy * dy).sqrt().max(1.0);
                    let repulsion = spacing * spacing / dist;

                    let fx = repulsion * dx / dist;
                    let fy = repulsion * dy / dist;

                    forces[i].x -= fx;
                    forces[i].y -= fy;
                    forces[j].x += fx;
                    forces[j].y += fy;
                }
            }

            // Attraction along edges
            for edge in &state.graph.edges {
                let Some(&source_idx) = state.node_index.get(&edge.source) else {
                    continue;
                };
                let Some(&target_idx) = state.node_index.get(&edge.target) else {
                    continue;
                };

                let dx = state.graph.nodes[target_idx].position.x
                    - state.graph.nodes[source_idx].position.x;
                let dy = state.graph.nodes[target_idx].position.y
                    - state.graph.nodes[source_idx].position.y;
                let dist = (dx * dx + dy * dy).sqrt();

                let attraction = dist / spacing;
                let fx = attraction * dx / dist.max(1.0);
                let fy = attraction * dy / dist.max(1.0);

                forces[source_idx].x += fx * 0.1;
                forces[source_idx].y += fy * 0.1;
                forces[target_idx].x -= fx * 0.1;
                forces[target_idx].y -= fy * 0.1;
            }

            // Apply forces with damping
            let damping = 0.85;
            for (i, node) in state.graph.nodes.iter_mut().enumerate() {
                node.position.x += forces[i].x * damping;
                node.position.y += forces[i].y * damping;

                // Keep in bounds
                node.position.x = node.position.x.max(50.0).min(state.graph.width - 50.0);
                node.position.y = node.position.y.max(50.0).min(state.graph.height - 50.0);
            }
        }

        Ok(())
    }

    fn apply_hierarchical_layout(&self, state: &mut DesignerState) -> Result<()> {
        if state.graph.nodes.is_empty() {
            return Ok(());
        }

        let spacing = state.layout_options.node_spacing;
        let layer_spacing = state.layout_options.layer_spacing;

        // Group by type if enabled
        let groups: Vec<Vec<usize>> = if state.layout_options.group_by_type {
            let mut type_groups: HashMap<NodeType, Vec<usize>> = HashMap::new();
            for (i, node) in state.graph.nodes.iter().enumerate() {
                type_groups.entry(node.node_type).or_default().push(i);
            }
            type_groups.into_values().collect()
        } else {
            vec![(0..state.graph.nodes.len()).collect()]
        };

        let mut y = 50.0;
        for group in groups {
            let mut x = 50.0;
            for idx in group {
                state.graph.nodes[idx].position.x = x;
                state.graph.nodes[idx].position.y = y;
                x += state.graph.nodes[idx].size.width + spacing;
            }
            y += layer_spacing;
        }

        Ok(())
    }

    fn apply_circular_layout(&self, state: &mut DesignerState) -> Result<()> {
        let node_count = state.graph.nodes.len();
        if node_count == 0 {
            return Ok(());
        }

        let center_x = state.graph.width / 2.0;
        let center_y = state.graph.height / 2.0;
        let radius = (state.graph.width.min(state.graph.height) / 2.0) - 150.0;

        for (i, node) in state.graph.nodes.iter_mut().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (node_count as f64);
            node.position.x = center_x + radius * angle.cos();
            node.position.y = center_y + radius * angle.sin();
        }

        Ok(())
    }

    fn apply_grid_layout(&self, state: &mut DesignerState) -> Result<()> {
        let node_count = state.graph.nodes.len();
        if node_count == 0 {
            return Ok(());
        }

        let cols = (node_count as f64).sqrt().ceil() as usize;
        let spacing = state.layout_options.node_spacing;

        for (i, node) in state.graph.nodes.iter_mut().enumerate() {
            let row = i / cols;
            let col = i % cols;
            node.position.x = 50.0 + (col as f64) * (node.size.width + spacing);
            node.position.y = 50.0 + (row as f64) * (node.size.height + spacing);
        }

        Ok(())
    }

    /// Set layout options
    pub async fn set_layout_options(&self, options: LayoutOptions) {
        let mut state = self.state.write().await;
        state.layout_options = options;
    }

    /// Validate the schema
    pub async fn validate(&self) -> ValidationResult {
        let state = self.state.read().await;

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for Query type
        if !state.graph.nodes.iter().any(|n| n.name == "Query") {
            errors.push(ValidationError {
                code: "MISSING_QUERY".to_string(),
                message: "Schema must have a Query type".to_string(),
                node_id: None,
                field_name: None,
            });
        }

        // Check for unresolved type references
        let type_names: HashSet<_> = state.graph.nodes.iter().map(|n| n.name.as_str()).collect();
        let builtin_types: HashSet<&str> = ["String", "Int", "Float", "Boolean", "ID"]
            .iter()
            .copied()
            .collect();

        for node in &state.graph.nodes {
            for field in &node.fields {
                let field_type = field
                    .field_type
                    .trim_matches(|c| c == '[' || c == ']' || c == '!');
                if !type_names.contains(field_type) && !builtin_types.contains(field_type) {
                    errors.push(ValidationError {
                        code: "UNRESOLVED_TYPE".to_string(),
                        message: format!("Unknown type '{}' in field '{}'", field_type, field.name),
                        node_id: Some(node.id.clone()),
                        field_name: Some(field.name.clone()),
                    });
                }
            }

            // Check for empty types
            if matches!(
                node.node_type,
                NodeType::ObjectType | NodeType::InterfaceType | NodeType::InputType
            ) && node.fields.is_empty()
            {
                warnings.push(ValidationWarning {
                    code: "EMPTY_TYPE".to_string(),
                    message: format!("Type '{}' has no fields", node.name),
                    node_id: Some(node.id.clone()),
                });
            }

            // Check for empty enum
            if node.node_type == NodeType::EnumType && node.enum_values.is_empty() {
                errors.push(ValidationError {
                    code: "EMPTY_ENUM".to_string(),
                    message: format!("Enum '{}' has no values", node.name),
                    node_id: Some(node.id.clone()),
                    field_name: None,
                });
            }
        }

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Export to SDL (Schema Definition Language)
    pub async fn export_sdl(&self) -> String {
        let state = self.state.read().await;
        let mut sdl = String::new();

        for node in &state.graph.nodes {
            // Add description
            if let Some(desc) = &node.description {
                sdl.push_str(&format!("\"\"\"{}\"\"\"\n", desc));
            }

            match node.node_type {
                NodeType::ObjectType => {
                    let implements = if !node.implements.is_empty() {
                        format!(" implements {}", node.implements.join(" & "))
                    } else {
                        String::new()
                    };
                    sdl.push_str(&format!("type {}{} {{\n", node.name, implements));
                    for field in &node.fields {
                        if let Some(desc) = &field.description {
                            sdl.push_str(&format!("  \"\"\"{}\"\"\"\n", desc));
                        }
                        sdl.push_str(&format!("  {}: {}\n", field.name, field.type_string()));
                    }
                    sdl.push_str("}\n\n");
                }
                NodeType::InterfaceType => {
                    sdl.push_str(&format!("interface {} {{\n", node.name));
                    for field in &node.fields {
                        sdl.push_str(&format!("  {}: {}\n", field.name, field.type_string()));
                    }
                    sdl.push_str("}\n\n");
                }
                NodeType::InputType => {
                    sdl.push_str(&format!("input {} {{\n", node.name));
                    for field in &node.fields {
                        sdl.push_str(&format!("  {}: {}\n", field.name, field.type_string()));
                    }
                    sdl.push_str("}\n\n");
                }
                NodeType::EnumType => {
                    sdl.push_str(&format!("enum {} {{\n", node.name));
                    for value in &node.enum_values {
                        sdl.push_str(&format!("  {}\n", value.name));
                    }
                    sdl.push_str("}\n\n");
                }
                NodeType::UnionType => {
                    sdl.push_str(&format!(
                        "union {} = {}\n\n",
                        node.name,
                        node.union_members.join(" | ")
                    ));
                }
                NodeType::ScalarType => {
                    sdl.push_str(&format!("scalar {}\n\n", node.name));
                }
                NodeType::Directive => {
                    // Skip directive definitions in basic export
                }
            }
        }

        sdl
    }

    /// Import from SDL
    pub async fn import_sdl(&self, sdl: &str) -> Result<()> {
        // Basic SDL parser - in production, use a proper GraphQL parser
        let mut state = self.state.write().await;
        state.graph.nodes.clear();
        state.graph.edges.clear();

        let lines: Vec<&str> = sdl.lines().collect();
        let mut current_node: Option<SchemaNode> = None;
        let mut in_type = false;

        for line in lines {
            let line = line.trim();

            if line.starts_with("type ")
                || line.starts_with("interface ")
                || line.starts_with("input ")
                || line.starts_with("enum ")
            {
                // Save previous node
                if let Some(node) = current_node.take() {
                    state.graph.nodes.push(node);
                }

                let (kind, rest) = if let Some(rest) = line.strip_prefix("type ") {
                    (NodeType::ObjectType, rest)
                } else if let Some(rest) = line.strip_prefix("interface ") {
                    (NodeType::InterfaceType, rest)
                } else if let Some(rest) = line.strip_prefix("input ") {
                    (NodeType::InputType, rest)
                } else if let Some(rest) = line.strip_prefix("enum ") {
                    (NodeType::EnumType, rest)
                } else {
                    continue;
                };

                let name = rest
                    .split(|c: char| c.is_whitespace() || c == '{')
                    .next()
                    .unwrap_or("");
                current_node = Some(SchemaNode::new(
                    &uuid::Uuid::new_v4().to_string(),
                    name,
                    kind,
                ));
                in_type = true;
            } else if line == "}" {
                if let Some(node) = current_node.take() {
                    state.graph.nodes.push(node);
                }
                in_type = false;
            } else if in_type && !line.is_empty() && !line.starts_with("\"\"\"") {
                if let Some(ref mut node) = current_node {
                    // Parse field
                    if let Some((name, type_part)) = line.split_once(':') {
                        let name = name.trim();
                        let type_str = type_part.trim().trim_end_matches(',');
                        let field = FieldDefinition::new(name, type_str);
                        node.fields.push(field);
                    } else if node.node_type == NodeType::EnumType {
                        // Enum value
                        node.enum_values.push(EnumValue::new(line));
                    }
                }
            } else if let Some(rest) = line.strip_prefix("scalar ") {
                let name = rest.trim();
                let node = SchemaNode::new(
                    &uuid::Uuid::new_v4().to_string(),
                    name,
                    NodeType::ScalarType,
                );
                state.graph.nodes.push(node);
            } else if let Some(rest) = line.strip_prefix("union ") {
                if let Some((name, members)) = rest.split_once('=') {
                    let name = name.trim();
                    let members: Vec<String> =
                        members.split('|').map(|s| s.trim().to_string()).collect();
                    let mut node = SchemaNode::new(
                        &uuid::Uuid::new_v4().to_string(),
                        name,
                        NodeType::UnionType,
                    );
                    node.union_members = members;
                    state.graph.nodes.push(node);
                }
            }
        }

        // Save last node
        if let Some(node) = current_node {
            state.graph.nodes.push(node);
        }

        // Rebuild indices
        state.rebuild_indices();

        // Create edges for field references
        self.create_edges_from_fields(&mut state);

        // Apply layout
        drop(state);
        self.apply_layout().await?;

        Ok(())
    }

    fn create_edges_from_fields(&self, state: &mut DesignerState) {
        let type_map: HashMap<String, String> = state
            .graph
            .nodes
            .iter()
            .map(|n| (n.name.clone(), n.id.clone()))
            .collect();

        let builtin_types: HashSet<&str> = ["String", "Int", "Float", "Boolean", "ID"]
            .iter()
            .copied()
            .collect();

        let mut edges = Vec::new();

        for node in &state.graph.nodes {
            for field in &node.fields {
                let field_type = field
                    .field_type
                    .trim_matches(|c| c == '[' || c == ']' || c == '!');

                if builtin_types.contains(field_type) {
                    continue;
                }

                if let Some(target_id) = type_map.get(field_type) {
                    let edge = SchemaEdge::new(
                        &uuid::Uuid::new_v4().to_string(),
                        &node.id,
                        target_id,
                        EdgeType::Field,
                    )
                    .with_source_field(&field.name);
                    edges.push(edge);
                }
            }

            // Implements edges
            for interface in &node.implements {
                if let Some(target_id) = type_map.get(interface) {
                    let edge = SchemaEdge::new(
                        &uuid::Uuid::new_v4().to_string(),
                        &node.id,
                        target_id,
                        EdgeType::Implements,
                    );
                    edges.push(edge);
                }
            }

            // Union member edges
            for member in &node.union_members {
                if let Some(target_id) = type_map.get(member) {
                    let edge = SchemaEdge::new(
                        &uuid::Uuid::new_v4().to_string(),
                        &node.id,
                        target_id,
                        EdgeType::UnionMember,
                    );
                    edges.push(edge);
                }
            }
        }

        state.graph.edges = edges;
        state.rebuild_indices();
    }

    /// Can undo
    pub async fn can_undo(&self) -> bool {
        let state = self.state.read().await;
        state.history_position > 0
    }

    /// Undo last operation
    pub async fn undo(&self) -> Result<()> {
        let mut state = self.state.write().await;
        if state.history_position == 0 {
            return Err(anyhow!("Nothing to undo"));
        }
        state.history_position -= 1;
        // In a full implementation, would reverse the operation
        Ok(())
    }

    /// Can redo
    pub async fn can_redo(&self) -> bool {
        let state = self.state.read().await;
        state.history_position < state.history.len()
    }

    /// Redo operation
    pub async fn redo(&self) -> Result<()> {
        let mut state = self.state.write().await;
        if state.history_position >= state.history.len() {
            return Err(anyhow!("Nothing to redo"));
        }
        state.history_position += 1;
        // In a full implementation, would reapply the operation
        Ok(())
    }
}

impl Default for SchemaDesigner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_designer_creation() {
        let designer = SchemaDesigner::new();
        let graph = designer.get_visualization().await;
        assert!(graph.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_add_node() {
        let designer = SchemaDesigner::new();
        let node = SchemaNode::new("", "User", NodeType::ObjectType);
        let node_id = designer.add_node(node).await.unwrap();

        let graph = designer.get_visualization().await;
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].id, node_id);
    }

    #[tokio::test]
    async fn test_add_field() {
        let designer = SchemaDesigner::new();
        let node = SchemaNode::new("", "User", NodeType::ObjectType);
        let node_id = designer.add_node(node).await.unwrap();

        let field = FieldDefinition::new("name", "String").non_null();
        designer.add_field(&node_id, field).await.unwrap();

        let graph = designer.get_visualization().await;
        assert_eq!(graph.nodes[0].fields.len(), 1);
        assert_eq!(graph.nodes[0].fields[0].name, "name");
    }

    #[tokio::test]
    async fn test_validation() {
        let designer = SchemaDesigner::new();

        // Empty schema should fail validation
        let result = designer.validate().await;
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.code == "MISSING_QUERY"));

        // Add Query type
        let query = SchemaNode::new("", "Query", NodeType::ObjectType)
            .with_field(FieldDefinition::new("hello", "String"));
        designer.add_node(query).await.unwrap();

        let result = designer.validate().await;
        assert!(result.valid);
    }

    #[tokio::test]
    async fn test_export_sdl() {
        let designer = SchemaDesigner::new();

        let user = SchemaNode::new("", "User", NodeType::ObjectType)
            .with_field(FieldDefinition::new("id", "ID").non_null())
            .with_field(FieldDefinition::new("name", "String"));
        designer.add_node(user).await.unwrap();

        let sdl = designer.export_sdl().await;
        assert!(sdl.contains("type User"));
        assert!(sdl.contains("id: ID!"));
        assert!(sdl.contains("name: String"));
    }

    #[tokio::test]
    async fn test_import_sdl() {
        let designer = SchemaDesigner::new();

        let sdl = r#"
            type Query {
                users: [User]
            }

            type User {
                id: ID!
                name: String
            }
        "#;

        designer.import_sdl(sdl).await.unwrap();

        let graph = designer.get_visualization().await;
        assert_eq!(graph.nodes.len(), 2);
    }

    #[tokio::test]
    async fn test_layout_algorithms() {
        let designer = SchemaDesigner::new();

        // Add some nodes
        for i in 0..5 {
            let node = SchemaNode::new("", &format!("Type{}", i), NodeType::ObjectType);
            designer.add_node(node).await.unwrap();
        }

        // Test force-directed
        designer
            .set_layout_options(LayoutOptions {
                algorithm: LayoutAlgorithm::ForceDirected,
                ..Default::default()
            })
            .await;
        designer.apply_layout().await.unwrap();

        // Test circular
        designer
            .set_layout_options(LayoutOptions {
                algorithm: LayoutAlgorithm::Circular,
                ..Default::default()
            })
            .await;
        designer.apply_layout().await.unwrap();

        // Test grid
        designer
            .set_layout_options(LayoutOptions {
                algorithm: LayoutAlgorithm::Grid,
                ..Default::default()
            })
            .await;
        designer.apply_layout().await.unwrap();

        let graph = designer.get_visualization().await;
        // All nodes should have non-zero positions after layout
        for node in &graph.nodes {
            assert!(node.position.x > 0.0 || node.position.y > 0.0);
        }
    }

    #[tokio::test]
    async fn test_remove_node() {
        let designer = SchemaDesigner::new();

        let node = SchemaNode::new("", "User", NodeType::ObjectType);
        let node_id = designer.add_node(node).await.unwrap();

        designer.remove_node(&node_id).await.unwrap();

        let graph = designer.get_visualization().await;
        assert!(graph.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_field_type_string() {
        let field = FieldDefinition::new("items", "Item").list().non_null();
        assert_eq!(field.type_string(), "[Item]!");
    }

    #[tokio::test]
    async fn test_position_distance() {
        let p1 = Position::new(0.0, 0.0);
        let p2 = Position::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.0001);
    }
}
