//! Unit tests for the rich-content layer.
//!
//! Sibling module of [`crate::rich_content`]. Compiled only under
//! `cfg(test)` so the contents add no weight to the normal build.

#![cfg(test)]

use std::collections::HashMap;

use crate::rich_content_renderer::{html_escape, validate_sparql_query, RichMessage};
use crate::rich_content_types::{
    EdgeThickness, GraphLayout, GraphNode, GraphStyling, InteractiveFeatures, NodeShape, NodeSize,
    NodeStyling, RichContent,
};

#[test]
fn test_rich_message_creation() {
    let mut message = RichMessage::new();
    message.add_text("Hello, world!");
    message.add_code("fn main() { println!(\"Hello!\"); }", "rust");

    assert_eq!(message.content_blocks.len(), 2);

    let markdown = message.to_markdown();
    assert!(markdown.contains("Hello, world!"));
    assert!(markdown.contains("```rust"));
}

#[test]
fn test_sparql_validation() {
    let (valid, error) = validate_sparql_query("SELECT ?s WHERE { ?s ?p ?o }");
    assert!(valid);
    assert!(error.is_none());

    let (valid, error) = validate_sparql_query("invalid query");
    assert!(!valid);
    assert!(error.is_some());
}

#[test]
fn test_html_escape() {
    assert_eq!(html_escape("<script>"), "&lt;script&gt;");
    assert_eq!(html_escape("&amp;"), "&amp;amp;");
}

#[test]
fn test_graph_visualization() {
    let mut message = RichMessage::new();

    let nodes = vec![GraphNode {
        id: "node1".to_string(),
        label: "Node 1".to_string(),
        node_type: "entity".to_string(),
        properties: HashMap::new(),
        position: None,
        styling: NodeStyling {
            color: "#FF0000".to_string(),
            border_color: "#000000".to_string(),
            border_width: 1.0,
            opacity: 1.0,
            font_size: 12.0,
            font_color: "#000000".to_string(),
        },
        size: 10.0,
        shape: NodeShape::Circle,
    }];

    let edges = vec![];

    let layout = GraphLayout::ForceDirected;
    let styling = GraphStyling {
        background_color: "#FFFFFF".to_string(),
        grid_enabled: false,
        physics_enabled: true,
        clustering_enabled: false,
        smooth_curves: true,
        node_color: "#3498db".to_string(),
        edge_color: "#95a5a6".to_string(),
        node_size: NodeSize::Medium,
        edge_thickness: EdgeThickness::Medium,
        layout_algorithm: "force-directed".to_string(),
        show_labels: true,
    };
    let interactive_features = InteractiveFeatures {
        pan_enabled: true,
        zoom_enabled: true,
        node_selection: true,
        edge_selection: true,
        hover_effects: true,
        click_to_expand: false,
        context_menu: true,
    };

    message.add_content(RichContent::GraphVisualization {
        nodes,
        edges,
        layout,
        styling,
        interactive_features,
        metadata: HashMap::new(),
    });

    assert_eq!(message.content_blocks.len(), 1);

    let markdown = message.to_markdown();
    assert!(markdown.contains("Graph"));
}
