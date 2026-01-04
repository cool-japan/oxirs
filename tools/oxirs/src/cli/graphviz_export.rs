//! Graphviz DOT Format Export
//!
//! Exports RDF graphs and SPARQL query plans to Graphviz DOT format for visualization.
//! DOT files can be rendered using Graphviz tools (dot, neato, fdp, circo, etc.).
//!
//! # Features
//!
//! - **RDF Graph Visualization**: Export triples as directed graphs
//! - **Query Plan Visualization**: Visualize SPARQL query execution plans
//! - **Customizable Styling**: Node and edge styling, colors, shapes
//! - **Clustering**: Group related nodes (e.g., by namespace, graph)
//! - **Layout Options**: Support for different Graphviz layout engines
//!
//! # Examples
//!
//! ## RDF Graph Export
//! ```rust,no_run
//! use oxirs::cli::graphviz_export::{GraphvizExporter, GraphOptions};
//!
//! let mut exporter = GraphvizExporter::new(GraphOptions::default());
//! exporter.add_triple(
//!     "http://example.org/Alice",
//!     "http://xmlns.com/foaf/0.1/knows",
//!     "http://example.org/Bob"
//! );
//! let dot = exporter.to_dot();
//! std::fs::write("graph.dot", dot)?;
//! // Render: `dot -Tpng graph.dot -o graph.png`
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! ## Query Plan Export
//! ```rust,no_run
//! use oxirs::cli::graphviz_export::{QueryPlanExporter, PlanNode, PlanOptions};
//!
//! let mut exporter = QueryPlanExporter::new(PlanOptions::default());
//! let root = PlanNode::new("BGP", "Basic Graph Pattern");
//! let join = PlanNode::new("Join", "Hash Join");
//! exporter.add_edge(0, 1, "left");
//! let dot = exporter.to_dot();
//! # Ok::<(), std::io::Error>(())
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::Path;

/// RDF Graph exporter to Graphviz DOT
pub struct GraphvizExporter {
    nodes: HashSet<String>,
    edges: Vec<(String, String, String)>, // (subject, predicate, object)
    options: GraphOptions,
    node_labels: HashMap<String, String>,
    node_styles: HashMap<String, NodeStyle>,
}

/// Graph visualization options
#[derive(Debug, Clone)]
pub struct GraphOptions {
    /// Graph title
    pub title: String,
    /// Layout engine (dot, neato, fdp, circo, twopi)
    pub layout: LayoutEngine,
    /// Shorten URIs to local names
    pub shorten_uris: bool,
    /// Group by namespace
    pub cluster_by_namespace: bool,
    /// Show edge labels
    pub show_edge_labels: bool,
    /// Node shape for resources
    pub resource_shape: NodeShape,
    /// Node shape for literals
    pub literal_shape: NodeShape,
    /// Node shape for blank nodes
    pub bnode_shape: NodeShape,
}

impl Default for GraphOptions {
    fn default() -> Self {
        Self {
            title: "RDF Graph".to_string(),
            layout: LayoutEngine::Dot,
            shorten_uris: true,
            cluster_by_namespace: true,
            show_edge_labels: true,
            resource_shape: NodeShape::Ellipse,
            literal_shape: NodeShape::Box,
            bnode_shape: NodeShape::Circle,
        }
    }
}

/// Graphviz layout engines
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutEngine {
    /// Hierarchical (default)
    Dot,
    /// Spring model
    Neato,
    /// Force-directed
    Fdp,
    /// Circular
    Circo,
    /// Radial
    Twopi,
}

impl LayoutEngine {
    pub fn as_str(&self) -> &'static str {
        match self {
            LayoutEngine::Dot => "dot",
            LayoutEngine::Neato => "neato",
            LayoutEngine::Fdp => "fdp",
            LayoutEngine::Circo => "circo",
            LayoutEngine::Twopi => "twopi",
        }
    }
}

/// Node shapes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeShape {
    Box,
    Circle,
    Ellipse,
    Diamond,
    Triangle,
    Hexagon,
    Octagon,
}

impl NodeShape {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeShape::Box => "box",
            NodeShape::Circle => "circle",
            NodeShape::Ellipse => "ellipse",
            NodeShape::Diamond => "diamond",
            NodeShape::Triangle => "triangle",
            NodeShape::Hexagon => "hexagon",
            NodeShape::Octagon => "octagon",
        }
    }
}

/// Node styling
#[derive(Debug, Clone)]
pub struct NodeStyle {
    pub shape: NodeShape,
    pub color: String,
    pub fillcolor: String,
    pub fontcolor: String,
}

impl Default for NodeStyle {
    fn default() -> Self {
        Self {
            shape: NodeShape::Ellipse,
            color: "black".to_string(),
            fillcolor: "lightblue".to_string(),
            fontcolor: "black".to_string(),
        }
    }
}

impl GraphvizExporter {
    /// Create new exporter with options
    pub fn new(options: GraphOptions) -> Self {
        Self {
            nodes: HashSet::new(),
            edges: Vec::new(),
            options,
            node_labels: HashMap::new(),
            node_styles: HashMap::new(),
        }
    }

    /// Add an RDF triple
    pub fn add_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        self.nodes.insert(subject.to_string());
        self.nodes.insert(object.to_string());
        self.edges.push((
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        ));
    }

    /// Set custom label for a node
    pub fn set_node_label(&mut self, node: &str, label: &str) {
        self.node_labels.insert(node.to_string(), label.to_string());
    }

    /// Set custom style for a node
    pub fn set_node_style(&mut self, node: &str, style: NodeStyle) {
        self.node_styles.insert(node.to_string(), style);
    }

    /// Generate DOT format output
    pub fn to_dot(&self) -> String {
        let mut output = String::new();

        // Graph header
        writeln!(
            &mut output,
            "digraph \"{}\" {{",
            escape_dot_string(&self.options.title)
        )
        .expect("writing to String should not fail");
        writeln!(&mut output, "  rankdir=LR;").expect("writing to String should not fail");
        writeln!(&mut output, "  node [style=filled];").expect("writing to String should not fail");
        writeln!(&mut output).expect("writing to String should not fail");

        // Clustering by namespace
        if self.options.cluster_by_namespace {
            let namespaces = self.extract_namespaces();
            for (idx, (ns, nodes)) in namespaces.iter().enumerate() {
                writeln!(&mut output, "  subgraph cluster_{} {{", idx)
                    .expect("writing to String should not fail");
                writeln!(&mut output, "    label=\"{}\";", escape_dot_string(ns))
                    .expect("writing to String should not fail");
                writeln!(&mut output, "    style=dashed;")
                    .expect("writing to String should not fail");

                for node in nodes {
                    self.write_node(&mut output, node);
                }

                writeln!(&mut output, "  }}").expect("writing to String should not fail");
                writeln!(&mut output).expect("writing to String should not fail");
            }

            // Nodes without namespace
            for node in &self.nodes {
                if !Self::has_namespace(node) {
                    self.write_node(&mut output, node);
                }
            }
        } else {
            // All nodes without clustering
            for node in &self.nodes {
                self.write_node(&mut output, node);
            }
        }

        writeln!(&mut output).expect("writing to String should not fail");

        // Edges
        for (subj, pred, obj) in &self.edges {
            let subj_id = Self::node_id(subj);
            let obj_id = Self::node_id(obj);

            if self.options.show_edge_labels {
                let pred_label = if self.options.shorten_uris {
                    Self::shorten_uri(pred)
                } else {
                    pred.clone()
                };
                writeln!(
                    &mut output,
                    "  {} -> {} [label=\"{}\"];",
                    subj_id,
                    obj_id,
                    escape_dot_string(&pred_label)
                )
                .expect("writing to String should not fail");
            } else {
                writeln!(&mut output, "  {} -> {};", subj_id, obj_id)
                    .expect("writing to String should not fail");
            }
        }

        writeln!(&mut output, "}}").expect("writing to String should not fail");
        output
    }

    /// Write node definition
    fn write_node(&self, output: &mut String, node: &str) {
        let node_id = Self::node_id(node);
        let label = self.node_labels.get(node).cloned().unwrap_or_else(|| {
            if self.options.shorten_uris {
                Self::shorten_uri(node)
            } else {
                node.to_string()
            }
        });

        let style = self.node_styles.get(node).cloned().unwrap_or_else(|| {
            let shape = if node.starts_with("http://") || node.starts_with("https://") {
                self.options.resource_shape
            } else if node.starts_with("_:") {
                self.options.bnode_shape
            } else {
                self.options.literal_shape
            };

            let fillcolor = if node.starts_with("http://") || node.starts_with("https://") {
                "#E8F4F8"
            } else if node.starts_with("_:") {
                "#FFF4E6"
            } else {
                "#E8F8E8"
            };

            NodeStyle {
                shape,
                color: "black".to_string(),
                fillcolor: fillcolor.to_string(),
                fontcolor: "black".to_string(),
            }
        });

        writeln!(
            output,
            "  {} [label=\"{}\", shape={}, fillcolor=\"{}\", color=\"{}\", fontcolor=\"{}\"];",
            node_id,
            escape_dot_string(&label),
            style.shape.as_str(),
            style.fillcolor,
            style.color,
            style.fontcolor
        )
        .expect("writing to String should not fail");
    }

    /// Extract namespaces from nodes
    fn extract_namespaces(&self) -> HashMap<String, Vec<String>> {
        let mut namespaces: HashMap<String, Vec<String>> = HashMap::new();

        for node in &self.nodes {
            if let Some(ns) = Self::extract_namespace(node) {
                namespaces.entry(ns).or_default().push(node.clone());
            }
        }

        namespaces
    }

    /// Extract namespace from URI
    fn extract_namespace(uri: &str) -> Option<String> {
        if uri.starts_with("http://") || uri.starts_with("https://") {
            uri.rfind('#')
                .or_else(|| uri.rfind('/'))
                .map(|pos| uri[..pos + 1].to_string())
        } else {
            None
        }
    }

    /// Check if node has namespace
    fn has_namespace(uri: &str) -> bool {
        Self::extract_namespace(uri).is_some()
    }

    /// Shorten URI to local name
    fn shorten_uri(uri: &str) -> String {
        if uri.starts_with("http://") || uri.starts_with("https://") {
            uri.rsplit_once('#')
                .or_else(|| uri.rsplit_once('/'))
                .map(|(_, local)| local.to_string())
                .unwrap_or_else(|| uri.to_string())
        } else {
            uri.to_string()
        }
    }

    /// Generate safe node ID
    fn node_id(uri: &str) -> String {
        format!(
            "n{}",
            uri.chars()
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect::<String>()
        )
    }

    /// Write DOT to file
    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        fs::write(path, self.to_dot())
    }
}

/// SPARQL query plan exporter
pub struct QueryPlanExporter {
    nodes: Vec<PlanNode>,
    edges: Vec<(usize, usize, String)>, // (from_idx, to_idx, label)
    options: PlanOptions,
}

/// Query plan visualization options
#[derive(Debug, Clone)]
pub struct PlanOptions {
    pub title: String,
    pub layout: LayoutEngine,
    pub show_statistics: bool,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            title: "SPARQL Query Plan".to_string(),
            layout: LayoutEngine::Dot,
            show_statistics: true,
        }
    }
}

/// Query plan node
#[derive(Debug, Clone)]
pub struct PlanNode {
    pub id: usize,
    pub operation: String,
    pub description: String,
    pub cost: Option<f64>,
    pub cardinality: Option<usize>,
}

impl PlanNode {
    pub fn new(operation: &str, description: &str) -> Self {
        Self {
            id: 0, // Will be set when added to exporter
            operation: operation.to_string(),
            description: description.to_string(),
            cost: None,
            cardinality: None,
        }
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    pub fn with_cardinality(mut self, cardinality: usize) -> Self {
        self.cardinality = Some(cardinality);
        self
    }
}

impl QueryPlanExporter {
    pub fn new(options: PlanOptions) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            options,
        }
    }

    /// Add a plan node
    pub fn add_node(&mut self, mut node: PlanNode) -> usize {
        let id = self.nodes.len();
        node.id = id;
        self.nodes.push(node);
        id
    }

    /// Add an edge between nodes
    pub fn add_edge(&mut self, from_id: usize, to_id: usize, label: &str) {
        self.edges.push((from_id, to_id, label.to_string()));
    }

    /// Generate DOT format
    pub fn to_dot(&self) -> String {
        let mut output = String::new();

        writeln!(
            &mut output,
            "digraph \"{}\" {{",
            escape_dot_string(&self.options.title)
        )
        .expect("writing to String should not fail");
        writeln!(&mut output, "  rankdir=TB;").expect("writing to String should not fail");
        writeln!(&mut output, "  node [style=filled, shape=box];")
            .expect("writing to String should not fail");
        writeln!(&mut output).expect("writing to String should not fail");

        // Nodes
        for node in &self.nodes {
            let mut label = format!("{}\n{}", node.operation, node.description);

            if self.options.show_statistics {
                if let Some(cost) = node.cost {
                    label.push_str(&format!("\nCost: {:.2}", cost));
                }
                if let Some(card) = node.cardinality {
                    label.push_str(&format!("\nCardinality: {}", card));
                }
            }

            let color = match node.operation.as_str() {
                "Scan" | "BGP" => "#E8F4F8",
                "Join" | "LeftJoin" => "#FFF4E6",
                "Filter" => "#F8E8E8",
                "Project" | "Distinct" => "#E8F8E8",
                _ => "#F0F0F0",
            };

            writeln!(
                &mut output,
                "  n{} [label=\"{}\", fillcolor=\"{}\"];",
                node.id,
                escape_dot_string(&label),
                color
            )
            .expect("writing to String should not fail");
        }

        writeln!(&mut output).expect("writing to String should not fail");

        // Edges
        for (from, to, label) in &self.edges {
            writeln!(
                &mut output,
                "  n{} -> n{} [label=\"{}\"];",
                from,
                to,
                escape_dot_string(label)
            )
            .expect("writing to String should not fail");
        }

        writeln!(&mut output, "}}").expect("writing to String should not fail");
        output
    }

    /// Write to file
    pub fn write_to_file(&self, path: &Path) -> std::io::Result<()> {
        fs::write(path, self.to_dot())
    }
}

/// Escape string for DOT format
fn escape_dot_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rdf_graph_export() {
        let mut exporter = GraphvizExporter::new(GraphOptions::default());
        exporter.add_triple(
            "http://example.org/Alice",
            "http://xmlns.com/foaf/0.1/knows",
            "http://example.org/Bob",
        );
        exporter.add_triple(
            "http://example.org/Bob",
            "http://xmlns.com/foaf/0.1/name",
            "Bob Smith",
        );

        let dot = exporter.to_dot();

        assert!(dot.contains("digraph"));
        assert!(dot.contains("Alice"));
        assert!(dot.contains("Bob"));
        assert!(dot.contains("knows"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_query_plan_export() {
        let mut exporter = QueryPlanExporter::new(PlanOptions::default());

        let scan = exporter.add_node(
            PlanNode::new("Scan", "Table scan")
                .with_cost(100.0)
                .with_cardinality(1000),
        );
        let filter = exporter.add_node(PlanNode::new("Filter", "?x > 10"));
        let project = exporter.add_node(PlanNode::new("Project", "?x, ?y"));

        exporter.add_edge(scan, filter, "input");
        exporter.add_edge(filter, project, "filtered");

        let dot = exporter.to_dot();

        assert!(dot.contains("Scan"));
        assert!(dot.contains("Filter"));
        assert!(dot.contains("Project"));
        assert!(dot.contains("Cost: 100.00"));
        assert!(dot.contains("Cardinality: 1000"));
    }

    #[test]
    fn test_shorten_uri() {
        assert_eq!(
            GraphvizExporter::shorten_uri("http://example.org/person#Alice"),
            "Alice"
        );
        assert_eq!(
            GraphvizExporter::shorten_uri("http://xmlns.com/foaf/0.1/name"),
            "name"
        );
        assert_eq!(
            GraphvizExporter::shorten_uri("literal value"),
            "literal value"
        );
    }

    #[test]
    fn test_escape_dot_string() {
        assert_eq!(escape_dot_string("simple"), "simple");
        assert_eq!(escape_dot_string("with\"quote"), "with\\\"quote");
        assert_eq!(escape_dot_string("with\nnewline"), "with\\nnewline");
    }
}
