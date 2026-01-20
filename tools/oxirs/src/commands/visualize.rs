//! RDF graph visualization export commands

use super::CommandResult;
use crate::cli::{progress::helpers, CliContext};
use oxirs_core::model::{GraphName, NamedNode};
use oxirs_core::rdf_store::RdfStore;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;

/// Visualization format
#[derive(Debug, Clone, Copy)]
pub enum VisFormat {
    Dot,       // Graphviz DOT format
    Mermaid,   // Mermaid diagram
    Cytoscape, // Cytoscape JSON
}

impl VisFormat {
    fn from_string(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "dot" | "graphviz" => Ok(VisFormat::Dot),
            "mermaid" | "mmd" => Ok(VisFormat::Mermaid),
            "cytoscape" | "cyto" | "json" => Ok(VisFormat::Cytoscape),
            _ => Err(format!(
                "Invalid visualization format: {}. Use dot/mermaid/cytoscape",
                s
            )),
        }
    }
}

/// Export RDF graph as visualization
pub async fn export(
    dataset: String,
    output: PathBuf,
    format: String,
    graph: Option<String>,
    max_nodes: Option<usize>,
) -> CommandResult {
    let ctx = CliContext::new();
    ctx.info(&format!(
        "Exporting visualization for dataset '{}'",
        dataset
    ));

    // Parse format
    let vis_format = VisFormat::from_string(&format)?;
    ctx.info(&format!("Format: {}", format));

    if let Some(ref g) = graph {
        ctx.info(&format!("Graph: {}", g));
    }

    if let Some(max) = max_nodes {
        ctx.info(&format!("Maximum nodes: {}", max));
    }

    // Load dataset
    let dataset_path = PathBuf::from(&dataset);
    if !dataset_path.exists() {
        return Err(format!("Dataset '{}' not found", dataset).into());
    }

    let store =
        RdfStore::open(&dataset_path).map_err(|e| format!("Failed to open dataset: {}", e))?;

    // Create progress spinner
    let progress = helpers::query_progress();
    progress.set_message("Analyzing graph structure");

    // Extract graph structure
    let graph_data = extract_graph_data(&store, graph.as_deref(), max_nodes)?;

    progress.set_message("Generating visualization");

    // Generate visualization
    let output_content = match vis_format {
        VisFormat::Dot => generate_dot(&graph_data),
        VisFormat::Mermaid => generate_mermaid(&graph_data),
        VisFormat::Cytoscape => generate_cytoscape(&graph_data),
    };

    progress.set_message("Writing output file");

    // Write output
    fs::write(&output, output_content)?;

    progress.finish_with_message("Visualization exported");

    ctx.success(&format!(
        "✓ Visualization exported to: {}",
        output.display()
    ));
    ctx.info(&format!("  Nodes: {}", graph_data.nodes.len()));
    ctx.info(&format!("  Edges: {}", graph_data.edges.len()));

    if let Some(max) = max_nodes {
        if graph_data.nodes.len() >= max {
            ctx.warn(&format!(
                "Graph truncated to {} nodes (use --max-nodes to increase)",
                max
            ));
        }
    }

    Ok(())
}

// Helper structures

#[derive(Debug)]
struct GraphData {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

#[derive(Debug)]
struct Node {
    id: String,
    label: String,
    node_type: NodeType,
}

#[derive(Debug, Clone, PartialEq)]
enum NodeType {
    Resource,
    Literal,
    BlankNode,
}

#[derive(Debug)]
struct Edge {
    source: String,
    target: String,
    label: String,
}

fn extract_graph_data(
    store: &RdfStore,
    graph: Option<&str>,
    max_nodes: Option<usize>,
) -> Result<GraphData, Box<dyn std::error::Error>> {
    let quads = store.quads().map_err(|e| e.to_string())?;

    let mut nodes_set: HashSet<String> = HashSet::new();
    let mut node_info: HashMap<String, (String, NodeType)> = HashMap::new();
    let mut edges = Vec::new();

    let max = max_nodes.unwrap_or(1000); // Default max 1000 nodes

    // Filter by graph if specified
    let filtered_quads: Vec<_> = if let Some(g) = graph {
        let target_graph = if g == "default" {
            GraphName::DefaultGraph
        } else {
            GraphName::NamedNode(NamedNode::new(g).map_err(|e| e.to_string())?)
        };
        quads
            .into_iter()
            .filter(|q| q.graph_name() == &target_graph)
            .collect()
    } else {
        quads
    };

    for quad in filtered_quads {
        if nodes_set.len() >= max {
            break;
        }

        // Subject
        let subject_id = format_term_id(&quad.subject().to_string());
        let subject_label = format_term_label(&quad.subject().to_string());
        nodes_set.insert(subject_id.clone());
        node_info.insert(subject_id.clone(), (subject_label, NodeType::Resource));

        // Object
        let object_str = quad.object().to_string();
        let object_id = format_term_id(&object_str);
        let object_label = format_term_label(&object_str);

        let object_type = if object_str.starts_with('"') {
            NodeType::Literal
        } else if object_str.starts_with("_:") {
            NodeType::BlankNode
        } else {
            NodeType::Resource
        };

        nodes_set.insert(object_id.clone());
        node_info.insert(object_id.clone(), (object_label, object_type));

        // Edge (predicate)
        let predicate_label = format_term_label(&quad.predicate().to_string());
        edges.push(Edge {
            source: subject_id,
            target: object_id,
            label: predicate_label,
        });
    }

    // Convert to node list
    let nodes: Vec<Node> = nodes_set
        .into_iter()
        .map(|id| {
            let (label, node_type) = node_info
                .get(&id)
                .cloned()
                .unwrap_or_else(|| (id.clone(), NodeType::Resource));
            Node {
                id,
                label,
                node_type,
            }
        })
        .collect();

    Ok(GraphData { nodes, edges })
}

fn format_term_id(term: &str) -> String {
    // Create a safe ID from the term
    let mut result = term.replace(['<', '>', '"', ' ', ':', '/', '#'], "_");

    // Collapse consecutive underscores into a single underscore
    while result.contains("__") {
        result = result.replace("__", "_");
    }

    result.trim_matches('_').to_string()
}

fn format_term_label(term: &str) -> String {
    // Extract a readable label from the term
    if let Some(stripped) = term.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        // IRI - extract last part
        if let Some(pos) = stripped.rfind(['/', '#']) {
            stripped[pos + 1..].to_string()
        } else {
            stripped.to_string()
        }
    } else if let Some(literal) = term.strip_prefix('"') {
        // Literal - extract value
        if let Some(end) = literal.find('"') {
            literal[..end].to_string()
        } else {
            literal.to_string()
        }
    } else {
        term.to_string()
    }
}

fn generate_dot(data: &GraphData) -> String {
    let mut output = String::new();
    output.push_str("digraph RDF {\n");
    output.push_str("  rankdir=LR;\n");
    output.push_str("  node [shape=box];\n\n");

    // Nodes with styling based on type
    for node in &data.nodes {
        let (shape, style) = match node.node_type {
            NodeType::Resource => ("box", "filled,rounded"),
            NodeType::Literal => ("ellipse", "filled"),
            NodeType::BlankNode => ("diamond", "filled"),
        };

        output.push_str(&format!(
            "  \"{}\" [label=\"{}\", shape={}, style=\"{}\"];\n",
            escape_dot(&node.id),
            escape_dot(&node.label),
            shape,
            style
        ));
    }

    output.push('\n');

    // Edges
    for edge in &data.edges {
        output.push_str(&format!(
            "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
            escape_dot(&edge.source),
            escape_dot(&edge.target),
            escape_dot(&edge.label)
        ));
    }

    output.push_str("}\n");
    output
}

fn generate_mermaid(data: &GraphData) -> String {
    let mut output = String::new();
    output.push_str("graph LR\n");

    // Create node definitions
    for node in &data.nodes {
        let (brackets_open, brackets_close) = match node.node_type {
            NodeType::Resource => ("[", "]"),
            NodeType::Literal => ("([", "])"),
            NodeType::BlankNode => ("{{", "}}"),
        };

        output.push_str(&format!(
            "  {}{}{}{}\n",
            escape_mermaid(&node.id),
            brackets_open,
            escape_mermaid(&node.label),
            brackets_close
        ));
    }

    output.push('\n');

    // Edges
    for edge in &data.edges {
        output.push_str(&format!(
            "  {} -->|{}| {}\n",
            escape_mermaid(&edge.source),
            escape_mermaid(&edge.label),
            escape_mermaid(&edge.target)
        ));
    }

    output
}

fn generate_cytoscape(data: &GraphData) -> String {
    let mut output = String::new();
    output.push_str("{\n");
    output.push_str("  \"elements\": {\n");

    // Nodes
    output.push_str("    \"nodes\": [\n");
    for (i, node) in data.nodes.iter().enumerate() {
        output.push_str("      {\n");
        output.push_str("        \"data\": {\n");
        output.push_str(&format!(
            "          \"id\": \"{}\",\n",
            escape_json(&node.id)
        ));
        output.push_str(&format!(
            "          \"label\": \"{}\",\n",
            escape_json(&node.label)
        ));
        output.push_str(&format!(
            "          \"type\": \"{}\"\n",
            match node.node_type {
                NodeType::Resource => "resource",
                NodeType::Literal => "literal",
                NodeType::BlankNode => "blank",
            }
        ));
        output.push_str("        }\n");
        output.push_str("      }");
        if i < data.nodes.len() - 1 {
            output.push(',');
        }
        output.push('\n');
    }
    output.push_str("    ],\n");

    // Edges
    output.push_str("    \"edges\": [\n");
    for (i, edge) in data.edges.iter().enumerate() {
        output.push_str("      {\n");
        output.push_str("        \"data\": {\n");
        output.push_str(&format!(
            "          \"id\": \"{}_{}\",\n",
            escape_json(&edge.source),
            escape_json(&edge.target)
        ));
        output.push_str(&format!(
            "          \"source\": \"{}\",\n",
            escape_json(&edge.source)
        ));
        output.push_str(&format!(
            "          \"target\": \"{}\",\n",
            escape_json(&edge.target)
        ));
        output.push_str(&format!(
            "          \"label\": \"{}\"\n",
            escape_json(&edge.label)
        ));
        output.push_str("        }\n");
        output.push_str("      }");
        if i < data.edges.len() - 1 {
            output.push(',');
        }
        output.push('\n');
    }
    output.push_str("    ]\n");

    output.push_str("  }\n");
    output.push_str("}\n");
    output
}

fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn escape_mermaid(s: &str) -> String {
    s.replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vis_format_parsing() {
        assert!(matches!(VisFormat::from_string("dot"), Ok(VisFormat::Dot)));
        assert!(matches!(
            VisFormat::from_string("graphviz"),
            Ok(VisFormat::Dot)
        ));
        assert!(matches!(
            VisFormat::from_string("mermaid"),
            Ok(VisFormat::Mermaid)
        ));
        assert!(matches!(
            VisFormat::from_string("mmd"),
            Ok(VisFormat::Mermaid)
        ));
        assert!(matches!(
            VisFormat::from_string("cytoscape"),
            Ok(VisFormat::Cytoscape)
        ));
        assert!(VisFormat::from_string("invalid").is_err());
    }

    #[test]
    fn test_format_term_id() {
        assert_eq!(
            format_term_id("<http://example.org/Person>"),
            "http_example.org_Person"
        );
        assert_eq!(format_term_id("\"John Doe\""), "John_Doe");
    }

    #[test]
    fn test_format_term_label() {
        assert_eq!(format_term_label("<http://example.org/name>"), "name");
        assert_eq!(format_term_label("<http://example.org#name>"), "name");
        assert_eq!(format_term_label("\"John Doe\""), "John Doe");
    }

    #[test]
    fn test_escape_functions() {
        assert_eq!(escape_dot("test\"quote"), "test\\\"quote");
        assert_eq!(escape_mermaid("test<tag>"), "test&lt;tag&gt;");
        assert_eq!(escape_json("test\\slash"), "test\\\\slash");
    }
}
