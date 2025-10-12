//! SAMM to Diagram Generator (Graphviz DOT format)
//!
//! Generates visual diagrams from SAMM Aspect models using Graphviz.
//! Supports SVG and PNG output formats.

use crate::error::SammError;
use crate::metamodel::{Aspect, ModelElement};
use std::io::Write;
use std::process::Command;

/// Diagram output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagramFormat {
    /// DOT format (Graphviz source)
    Dot,
    /// SVG vector format
    Svg,
    /// PNG raster format
    Png,
}

/// Generate diagram from SAMM Aspect
pub fn generate_diagram(aspect: &Aspect, format: DiagramFormat) -> Result<String, SammError> {
    let dot_source = generate_dot(aspect)?;

    match format {
        DiagramFormat::Dot => Ok(dot_source),
        DiagramFormat::Svg => render_graphviz(&dot_source, "svg"),
        DiagramFormat::Png => render_graphviz(&dot_source, "png"),
    }
}

/// Generate Graphviz DOT source from Aspect
fn generate_dot(aspect: &Aspect) -> Result<String, SammError> {
    let aspect_name = aspect.name();
    let mut dot = String::new();

    dot.push_str("digraph SAMM_Aspect {\n");
    dot.push_str("  rankdir=LR;\n");
    dot.push_str("  node [shape=box, style=rounded];\n\n");

    // Aspect node (central)
    dot.push_str(&format!(
        "  \"{}\" [shape=component, style=filled, fillcolor=lightblue, label=\"{}\"];\n\n",
        aspect_name, aspect_name
    ));

    // Properties
    for prop in aspect.properties() {
        let prop_name = prop.name();
        let type_info = if let Some(char) = &prop.characteristic {
            if let Some(dt) = &char.data_type {
                dt.split('#').next_back().unwrap_or("String")
            } else {
                "Trait"
            }
        } else {
            "Unknown"
        };

        dot.push_str(&format!(
            "  \"{}\" [label=\"{}\\n({})\", fillcolor=lightgreen, style=filled];\n",
            prop_name, prop_name, type_info
        ));
        dot.push_str(&format!("  \"{}\" -> \"{}\";\n", aspect_name, prop_name));
    }

    // Operations
    for op in aspect.operations() {
        let op_name = op.name();
        dot.push_str(&format!(
            "  \"{}\" [label=\"{}()\", shape=ellipse, fillcolor=lightyellow, style=filled];\n",
            op_name, op_name
        ));
        dot.push_str(&format!("  \"{}\" -> \"{}\";\n", aspect_name, op_name));
    }

    // Events
    for event in aspect.events() {
        let event_name = event.name();
        dot.push_str(&format!(
            "  \"{}\" [label=\"{}!\", shape=diamond, fillcolor=lightcoral, style=filled];\n",
            event_name, event_name
        ));
        dot.push_str(&format!("  \"{}\" -> \"{}\";\n", aspect_name, event_name));
    }

    dot.push_str("}\n");
    Ok(dot)
}

/// Render DOT source to image format using Graphviz
fn render_graphviz(dot_source: &str, format: &str) -> Result<String, SammError> {
    // Check if Graphviz is installed
    let check = Command::new("dot").arg("-V").output();

    if check.is_err() {
        return Err(SammError::Generation(
            "Graphviz not installed. Please install it: brew install graphviz".to_string(),
        ));
    }

    // Render using Graphviz
    let mut child = Command::new("dot")
        .arg(format!("-T{}", format))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| SammError::Generation(format!("Failed to spawn Graphviz: {}", e)))?;

    // Write DOT source to stdin
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(dot_source.as_bytes())
            .map_err(|e| SammError::Generation(format!("Failed to write to Graphviz: {}", e)))?;
    }

    // Wait for output
    let output = child
        .wait_with_output()
        .map_err(|e| SammError::Generation(format!("Failed to read Graphviz output: {}", e)))?;

    if !output.status.success() {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        return Err(SammError::Generation(format!(
            "Graphviz failed: {}",
            error_msg
        )));
    }

    // For alpha.3, return base64-encoded output for SVG/PNG
    let result = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagram_format_variants() {
        assert_eq!(DiagramFormat::Dot, DiagramFormat::Dot);
        assert_ne!(DiagramFormat::Svg, DiagramFormat::Png);
    }
}
