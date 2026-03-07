//! SAMM to Diagram Generator
//!
//! Generates visual diagrams from SAMM Aspect models in multiple formats:
//! - Graphviz DOT (with SVG/PNG rendering)
//! - Mermaid.js (for GitHub/markdown)
//! - PlantUML (for documentation)
//! - HTML reports with embedded diagrams
//!
//! # Examples
//!
//! ```no_run
//! use oxirs_samm::generators::diagram::{generate_diagram, DiagramFormat, DiagramStyle};
//! use oxirs_samm::metamodel::Aspect;
//!
//! # fn example(aspect: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
//! // Generate Mermaid.js diagram
//! let mermaid = generate_diagram(aspect, DiagramFormat::Mermaid(DiagramStyle::default()))?;
//! println!("```mermaid\n{}\n```", mermaid);
//!
//! // Generate PlantUML diagram
//! let plantuml = generate_diagram(aspect, DiagramFormat::PlantUml(DiagramStyle::default()))?;
//!
//! // Generate Graphviz DOT
//! let dot = generate_diagram(aspect, DiagramFormat::Dot(DiagramStyle::default()))?;
//! # Ok(())
//! # }
//! ```

use crate::error::SammError;
use crate::metamodel::{Aspect, CharacteristicKind, ModelElement};
use std::io::Write;
use std::process::Command;

/// Diagram styling options
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagramStyle {
    /// Show property types
    pub show_types: bool,
    /// Show optional/required indicators
    pub show_optionality: bool,
    /// Show cardinality (collection characteristics)
    pub show_cardinality: bool,
    /// Include metadata (descriptions, preferred names)
    pub include_metadata: bool,
    /// Color scheme: "default", "minimal", "colorful"
    pub color_scheme: String,
    /// Layout direction: "LR" (left-right), "TB" (top-bottom)
    pub layout_direction: String,
}

impl Default for DiagramStyle {
    fn default() -> Self {
        Self {
            show_types: true,
            show_optionality: true,
            show_cardinality: true,
            include_metadata: false,
            color_scheme: "default".to_string(),
            layout_direction: "LR".to_string(),
        }
    }
}

/// Diagram output format with styling options
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagramFormat {
    /// DOT format (Graphviz source)
    Dot(DiagramStyle),
    /// SVG vector format (rendered via Graphviz)
    Svg(DiagramStyle),
    /// PNG raster format (rendered via Graphviz)
    Png(DiagramStyle),
    /// Mermaid.js format (for GitHub/markdown)
    Mermaid(DiagramStyle),
    /// PlantUML format (for documentation)
    PlantUml(DiagramStyle),
    /// HTML report with embedded diagram
    HtmlReport(DiagramStyle),
}

/// Generate diagram from SAMM Aspect
///
/// # Arguments
///
/// * `aspect` - The SAMM Aspect model to visualize
/// * `format` - The desired output format with styling options
///
/// # Returns
///
/// The diagram as a string in the requested format
///
/// # Examples
///
/// ```no_run
/// use oxirs_samm::generators::diagram::{generate_diagram, DiagramFormat, DiagramStyle};
/// # use oxirs_samm::metamodel::Aspect;
/// # fn example(aspect: &Aspect) -> Result<(), Box<dyn std::error::Error>> {
/// let mermaid = generate_diagram(aspect, DiagramFormat::Mermaid(DiagramStyle::default()))?;
/// # Ok(())
/// # }
/// ```
pub fn generate_diagram(aspect: &Aspect, format: DiagramFormat) -> Result<String, SammError> {
    match format {
        DiagramFormat::Dot(style) => generate_dot(aspect, &style),
        DiagramFormat::Svg(style) => {
            let dot_source = generate_dot(aspect, &style)?;
            render_graphviz(&dot_source, "svg")
        }
        DiagramFormat::Png(style) => {
            let dot_source = generate_dot(aspect, &style)?;
            render_graphviz(&dot_source, "png")
        }
        DiagramFormat::Mermaid(style) => generate_mermaid(aspect, &style),
        DiagramFormat::PlantUml(style) => generate_plantuml(aspect, &style),
        DiagramFormat::HtmlReport(style) => generate_html_report(aspect, &style),
    }
}

/// Generate Graphviz DOT source from Aspect
fn generate_dot(aspect: &Aspect, style: &DiagramStyle) -> Result<String, SammError> {
    let aspect_name = aspect.name();
    let mut dot = String::new();

    dot.push_str("digraph SAMM_Aspect {\n");
    dot.push_str(&format!("  rankdir={};\n", style.layout_direction));
    dot.push_str("  node [shape=box, style=rounded];\n\n");

    // Aspect node (central)
    let aspect_label = if style.include_metadata {
        let desc = aspect.metadata.get_description("en").unwrap_or_default();
        if !desc.is_empty() {
            format!("{}\\n{}", aspect_name, desc)
        } else {
            aspect_name.to_string()
        }
    } else {
        aspect_name.to_string()
    };

    dot.push_str(&format!(
        "  \"{}\" [shape=component, style=filled, fillcolor=lightblue, label=\"{}\"];\n\n",
        aspect_name, aspect_label
    ));

    // Properties
    for prop in aspect.properties() {
        let prop_name = prop.name();
        let mut label = prop_name.to_string();

        if style.show_types {
            let type_info = if let Some(char) = &prop.characteristic {
                if let Some(dt) = &char.data_type {
                    dt.split('#').next_back().unwrap_or("String")
                } else {
                    "Trait"
                }
            } else {
                "Unknown"
            };
            label = format!("{}\\n({})", label, type_info);
        }

        if style.show_optionality {
            let opt_marker = if prop.optional {
                " [optional]"
            } else {
                " [required]"
            };
            label.push_str(opt_marker);
        }

        if style.show_cardinality && prop.is_collection {
            label.push_str("\\n[collection]");
        }

        dot.push_str(&format!(
            "  \"{}\" [label=\"{}\", fillcolor=lightgreen, style=filled];\n",
            prop_name, label
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

/// Generate Mermaid.js diagram from Aspect
///
/// Mermaid.js is a popular diagramming tool that works in GitHub markdown,
/// GitLab, and many documentation platforms.
fn generate_mermaid(aspect: &Aspect, style: &DiagramStyle) -> Result<String, SammError> {
    let aspect_name = aspect.name();
    let mut mermaid = String::new();

    // Determine graph direction
    let direction = match style.layout_direction.as_str() {
        "TB" => "TD", // Top-down
        "LR" => "LR", // Left-right
        _ => "LR",
    };

    mermaid.push_str(&format!("graph {}\n", direction));

    // Aspect node (central)
    mermaid.push_str(&format!(
        "    {}[\"🎯 {}\"]\n",
        sanitize_mermaid_id(&aspect_name),
        aspect_name
    ));
    mermaid.push_str(&format!(
        "    style {} fill:#87CEEB,stroke:#4682B4,stroke-width:3px\n\n",
        sanitize_mermaid_id(&aspect_name)
    ));

    // Properties
    for prop in aspect.properties() {
        let prop_name = prop.name();
        let prop_id = sanitize_mermaid_id(&prop_name);

        let mut label = prop_name.to_string();

        if style.show_types {
            if let Some(char) = &prop.characteristic {
                let type_info = if let Some(dt) = &char.data_type {
                    dt.split('#').next_back().unwrap_or("String")
                } else {
                    "Trait"
                };
                label = format!("{}<br/><i>{}</i>", label, type_info);
            }
        }

        if style.show_optionality {
            let marker = if prop.optional { "❓" } else { "✅" };
            label = format!("{} {}", marker, label);
        }

        if style.show_cardinality && prop.is_collection {
            label = format!("{}<br/>📦 collection", label);
        }

        mermaid.push_str(&format!("    {}[\"{}\"]\n", prop_id, label));
        mermaid.push_str(&format!(
            "    {} --> {}\n",
            sanitize_mermaid_id(&aspect_name),
            prop_id
        ));
        mermaid.push_str(&format!(
            "    style {} fill:#90EE90,stroke:#228B22\n",
            prop_id
        ));
    }

    // Operations
    for op in aspect.operations() {
        let op_name = op.name();
        let op_id = sanitize_mermaid_id(&op_name);

        mermaid.push_str(&format!("    {}{{\"⚙️ {}()\"}}\n", op_id, op_name));
        mermaid.push_str(&format!(
            "    {} --> {}\n",
            sanitize_mermaid_id(&aspect_name),
            op_id
        ));
        mermaid.push_str(&format!(
            "    style {} fill:#FFFFE0,stroke:#FFD700\n",
            op_id
        ));
    }

    // Events
    for event in aspect.events() {
        let event_name = event.name();
        let event_id = sanitize_mermaid_id(&event_name);

        mermaid.push_str(&format!("    {}{{{{\"⚡ {}!\"}}}}\n", event_id, event_name));
        mermaid.push_str(&format!(
            "    {} --> {}\n",
            sanitize_mermaid_id(&aspect_name),
            event_id
        ));
        mermaid.push_str(&format!(
            "    style {} fill:#FFB6C1,stroke:#DC143C\n",
            event_id
        ));
    }

    Ok(mermaid)
}

/// Generate PlantUML diagram from Aspect
///
/// PlantUML is widely used for documentation and supports many diagram types.
fn generate_plantuml(aspect: &Aspect, style: &DiagramStyle) -> Result<String, SammError> {
    let aspect_name = aspect.name();
    let mut puml = String::new();

    puml.push_str("@startuml\n");

    // Layout direction
    if style.layout_direction == "TB" {
        puml.push_str("top to bottom direction\n");
    } else {
        puml.push_str("left to right direction\n");
    }

    puml.push('\n');

    // Define Aspect as a component
    puml.push_str(&format!(
        "component \"{}\" as {} #LightBlue {{\n",
        aspect_name,
        sanitize_plantuml_id(&aspect_name)
    ));

    // Properties
    if !aspect.properties().is_empty() {
        puml.push_str("  frame \"Properties\" {\n");
        for prop in aspect.properties() {
            let prop_name = prop.name();
            let mut label = String::new();

            if style.show_optionality {
                label.push_str(if prop.optional { "[O] " } else { "[R] " });
            }

            label.push_str(&prop_name);

            if style.show_types {
                if let Some(char) = &prop.characteristic {
                    let type_info = if let Some(dt) = &char.data_type {
                        dt.split('#').next_back().unwrap_or("String")
                    } else {
                        "Trait"
                    };
                    label.push_str(&format!(" : {}", type_info));
                }
            }

            if style.show_cardinality && prop.is_collection {
                label.push_str(" [*]");
            }

            puml.push_str(&format!(
                "    card \"{}\" as {} #LightGreen\n",
                label,
                sanitize_plantuml_id(&prop_name)
            ));
        }
        puml.push_str("  }\n");
    }

    // Operations
    if !aspect.operations().is_empty() {
        puml.push_str("  frame \"Operations\" {\n");
        for op in aspect.operations() {
            let op_name = op.name();
            puml.push_str(&format!(
                "    card \"{}()\" as {} #LightYellow\n",
                op_name,
                sanitize_plantuml_id(&op_name)
            ));
        }
        puml.push_str("  }\n");
    }

    // Events
    if !aspect.events().is_empty() {
        puml.push_str("  frame \"Events\" {\n");
        for event in aspect.events() {
            let event_name = event.name();
            puml.push_str(&format!(
                "    card \"{}!\" as {} #LightCoral\n",
                event_name,
                sanitize_plantuml_id(&event_name)
            ));
        }
        puml.push_str("  }\n");
    }

    puml.push_str("}}\n");

    // Metadata notes
    if style.include_metadata {
        if let Some(desc) = aspect.metadata.get_description("en") {
            if !desc.is_empty() {
                puml.push_str(&format!(
                    "\nnote right of {}\n  {}\nend note\n",
                    sanitize_plantuml_id(&aspect_name),
                    desc
                ));
            }
        }
    }

    puml.push_str("\n@enduml\n");
    Ok(puml)
}

/// Generate HTML report with embedded diagram
///
/// Creates a standalone HTML file with an embedded Mermaid.js diagram
/// and detailed model information.
fn generate_html_report(aspect: &Aspect, style: &DiagramStyle) -> Result<String, SammError> {
    let aspect_name = aspect.name();
    let mermaid_diagram = generate_mermaid(aspect, style)?;

    let mut html = String::new();

    html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
    html.push_str("  <meta charset=\"UTF-8\">\n");
    html.push_str("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
    html.push_str(&format!("  <title>SAMM Aspect: {}</title>\n", aspect_name));
    html.push_str(
        "  <script src=\"https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js\"></script>\n",
    );
    html.push_str("  <style>\n");
    html.push_str("    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }\n");
    html.push_str("    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }\n");
    html.push_str(
        "    h1 { color: #333; border-bottom: 3px solid #4682B4; padding-bottom: 10px; }\n",
    );
    html.push_str("    h2 { color: #555; margin-top: 30px; }\n");
    html.push_str(
        "    .diagram { background: white; padding: 20px; border-radius: 4px; margin: 20px 0; }\n",
    );
    html.push_str("    .property { background: #f0f8f0; padding: 10px; margin: 5px 0; border-left: 4px solid #228B22; border-radius: 4px; }\n");
    html.push_str("    .operation { background: #fffff0; padding: 10px; margin: 5px 0; border-left: 4px solid #FFD700; border-radius: 4px; }\n");
    html.push_str("    .event { background: #fff0f0; padding: 10px; margin: 5px 0; border-left: 4px solid #DC143C; border-radius: 4px; }\n");
    html.push_str("    .badge { display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; font-weight: bold; }\n");
    html.push_str("    .required { background: #4CAF50; color: white; }\n");
    html.push_str("    .optional { background: #FFC107; color: black; }\n");
    html.push_str("    .collection { background: #2196F3; color: white; }\n");
    html.push_str("    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }\n");
    html.push_str("    .stat-card { background: #f8f9fa; padding: 15px; border-radius: 4px; text-align: center; }\n");
    html.push_str("    .stat-value { font-size: 32px; font-weight: bold; color: #4682B4; }\n");
    html.push_str("    .stat-label { color: #666; font-size: 14px; margin-top: 5px; }\n");
    html.push_str("  </style>\n</head>\n<body>\n");

    html.push_str("  <div class=\"container\">\n");
    html.push_str(&format!("    <h1>📊 SAMM Aspect: {}</h1>\n", aspect_name));

    // Metadata section
    if let Some(desc) = aspect.metadata.get_description("en") {
        html.push_str(&format!("    <p>{}</p>\n", desc));
    }

    // Statistics
    html.push_str("    <div class=\"stats\">\n");
    html.push_str("      <div class=\"stat-card\">\n");
    html.push_str(&format!(
        "        <div class=\"stat-value\">{}</div>\n",
        aspect.properties().len()
    ));
    html.push_str("        <div class=\"stat-label\">Properties</div>\n");
    html.push_str("      </div>\n");
    html.push_str("      <div class=\"stat-card\">\n");
    html.push_str(&format!(
        "        <div class=\"stat-value\">{}</div>\n",
        aspect.operations().len()
    ));
    html.push_str("        <div class=\"stat-label\">Operations</div>\n");
    html.push_str("      </div>\n");
    html.push_str("      <div class=\"stat-card\">\n");
    html.push_str(&format!(
        "        <div class=\"stat-value\">{}</div>\n",
        aspect.events().len()
    ));
    html.push_str("        <div class=\"stat-label\">Events</div>\n");
    html.push_str("      </div>\n");
    html.push_str("    </div>\n");

    // Diagram
    html.push_str("    <h2>📈 Visualization</h2>\n");
    html.push_str("    <div class=\"diagram mermaid\">\n");
    html.push_str(&mermaid_diagram);
    html.push_str("    </div>\n");

    // Properties details
    if !aspect.properties().is_empty() {
        html.push_str("    <h2>🔧 Properties</h2>\n");
        for prop in aspect.properties() {
            html.push_str("    <div class=\"property\">\n");
            html.push_str(&format!("      <strong>{}</strong>\n", prop.name()));

            if prop.optional {
                html.push_str("      <span class=\"badge optional\">OPTIONAL</span>\n");
            } else {
                html.push_str("      <span class=\"badge required\">REQUIRED</span>\n");
            }

            if prop.is_collection {
                html.push_str("      <span class=\"badge collection\">COLLECTION</span>\n");
            }

            if let Some(char) = &prop.characteristic {
                if let Some(dt) = &char.data_type {
                    let type_name = dt.split('#').next_back().unwrap_or("String");
                    html.push_str(&format!("<br/><em>Type: {}</em>\n", type_name));
                }
            }

            html.push_str("    </div>\n");
        }
    }

    // Operations details
    if !aspect.operations().is_empty() {
        html.push_str("    <h2>⚙️ Operations</h2>\n");
        for op in aspect.operations() {
            html.push_str("    <div class=\"operation\">\n");
            html.push_str(&format!("      <strong>{}()</strong>\n", op.name()));
            html.push_str("    </div>\n");
        }
    }

    // Events details
    if !aspect.events().is_empty() {
        html.push_str("    <h2>⚡ Events</h2>\n");
        for event in aspect.events() {
            html.push_str("    <div class=\"event\">\n");
            html.push_str(&format!("      <strong>{}!</strong>\n", event.name()));
            html.push_str("    </div>\n");
        }
    }

    // Footer
    html.push_str("    <hr style=\"margin-top: 40px;\">\n");
    html.push_str("    <p style=\"text-align: center; color: #666; font-size: 12px;\">\n");
    html.push_str("      Generated by OxiRS SAMM • SAMM Specification 2.3.0\n");
    html.push_str("    </p>\n");
    html.push_str("  </div>\n");

    html.push_str("  <script>\n");
    html.push_str("    mermaid.initialize({ startOnLoad: true, theme: 'default' });\n");
    html.push_str("  </script>\n");
    html.push_str("</body>\n</html>\n");

    Ok(html)
}

/// Sanitize identifier for Mermaid.js
fn sanitize_mermaid_id(name: &str) -> String {
    name.replace([':', '#', '.', '-', ' '], "_")
}

/// Sanitize identifier for PlantUML
fn sanitize_plantuml_id(name: &str) -> String {
    name.replace([':', '#', '.', '-', ' '], "_")
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
    use crate::metamodel::{Aspect, Characteristic, CharacteristicKind, ElementMetadata, Property};

    fn create_test_aspect() -> Aspect {
        let mut aspect = Aspect::new("urn:samm:test:1.0.0#TestAspect".to_string());

        // Add properties with different characteristics
        let mut prop1 = Property::new("urn:samm:test:1.0.0#temperature".to_string());
        prop1.optional = false;
        prop1.is_collection = false;
        prop1.characteristic = Some(Characteristic {
            metadata: ElementMetadata::new(
                "urn:samm:test:1.0.0#TemperatureCharacteristic".to_string(),
            ),
            kind: CharacteristicKind::Measurement {
                unit: "unit:degreeCelsius".to_string(),
            },
            data_type: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
            constraints: vec![],
        });
        aspect.properties.push(prop1);

        let mut prop2 = Property::new("urn:samm:test:1.0.0#status".to_string());
        prop2.optional = true;
        prop2.characteristic = Some(Characteristic {
            metadata: ElementMetadata::new("urn:samm:test:1.0.0#StatusCharacteristic".to_string()),
            kind: CharacteristicKind::Trait,
            data_type: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
            constraints: vec![],
        });
        aspect.properties.push(prop2);

        let mut prop3 = Property::new("urn:samm:test:1.0.0#values".to_string());
        prop3.is_collection = true;
        prop3.characteristic = Some(Characteristic {
            metadata: ElementMetadata::new("urn:samm:test:1.0.0#ValuesCharacteristic".to_string()),
            kind: CharacteristicKind::Collection {
                element_characteristic: None,
            },
            data_type: Some("http://www.w3.org/2001/XMLSchema#int".to_string()),
            constraints: vec![],
        });
        aspect.properties.push(prop3);

        aspect.metadata.add_description(
            "en".to_string(),
            "A test aspect for diagram generation".to_string(),
        );

        aspect
    }

    #[test]
    fn test_diagram_style_default() {
        let style = DiagramStyle::default();
        assert!(style.show_types);
        assert!(style.show_optionality);
        assert!(style.show_cardinality);
        assert!(!style.include_metadata);
        assert_eq!(style.color_scheme, "default");
        assert_eq!(style.layout_direction, "LR");
    }

    #[test]
    fn test_generate_mermaid_basic() {
        let aspect = create_test_aspect();
        let style = DiagramStyle::default();

        let result = generate_mermaid(&aspect, &style);
        assert!(result.is_ok());

        let mermaid = result.expect("result should be Ok");
        assert!(mermaid.contains("graph LR"));
        assert!(mermaid.contains("TestAspect"));
        assert!(mermaid.contains("temperature"));
        assert!(mermaid.contains("status"));
        assert!(mermaid.contains("values"));
        assert!(mermaid.contains("✅")); // required marker
        assert!(mermaid.contains("❓")); // optional marker
        assert!(mermaid.contains("📦 collection")); // collection marker
    }

    #[test]
    fn test_generate_mermaid_top_down() {
        let aspect = create_test_aspect();
        let style = DiagramStyle {
            layout_direction: "TB".to_string(),
            ..Default::default()
        };

        let result = generate_mermaid(&aspect, &style);
        assert!(result.is_ok());

        let mermaid = result.expect("result should be Ok");
        assert!(mermaid.contains("graph TD"));
    }

    #[test]
    fn test_generate_mermaid_minimal() {
        let aspect = create_test_aspect();
        let style = DiagramStyle {
            show_types: false,
            show_optionality: false,
            show_cardinality: false,
            include_metadata: false,
            color_scheme: "minimal".to_string(),
            layout_direction: "LR".to_string(),
        };

        let result = generate_mermaid(&aspect, &style);
        assert!(result.is_ok());

        let mermaid = result.expect("result should be Ok");
        // Should not contain markers when disabled
        assert!(!mermaid.contains("✅"));
        assert!(!mermaid.contains("❓"));
    }

    #[test]
    fn test_generate_plantuml_basic() {
        let aspect = create_test_aspect();
        let style = DiagramStyle::default();

        let result = generate_plantuml(&aspect, &style);
        assert!(result.is_ok());

        let puml = result.expect("result should be Ok");
        assert!(puml.starts_with("@startuml"));
        assert!(puml.ends_with("@enduml\n"));
        assert!(puml.contains("TestAspect"));
        assert!(puml.contains("temperature"));
        assert!(puml.contains("status"));
        assert!(puml.contains("values"));
        assert!(puml.contains("[R]")); // required
        assert!(puml.contains("[O]")); // optional
        assert!(puml.contains("[*]")); // collection
    }

    #[test]
    fn test_generate_plantuml_with_metadata() {
        let aspect = create_test_aspect();
        let style = DiagramStyle {
            include_metadata: true,
            ..Default::default()
        };

        let result = generate_plantuml(&aspect, &style);
        assert!(result.is_ok());

        let puml = result.expect("result should be Ok");
        assert!(puml.contains("note right of"));
        assert!(puml.contains("test aspect"));
    }

    #[test]
    fn test_generate_dot_basic() {
        let aspect = create_test_aspect();
        let style = DiagramStyle::default();

        let result = generate_dot(&aspect, &style);
        assert!(result.is_ok());

        let dot = result.expect("result should be Ok");
        assert!(dot.starts_with("digraph SAMM_Aspect"));
        assert!(dot.contains("rankdir=LR"));
        assert!(dot.contains("TestAspect"));
        assert!(dot.contains("temperature"));
        assert!(dot.contains("[required]"));
        assert!(dot.contains("[optional]"));
        assert!(dot.contains("[collection]"));
    }

    #[test]
    fn test_generate_html_report() {
        let aspect = create_test_aspect();
        let style = DiagramStyle::default();

        let result = generate_html_report(&aspect, &style);
        assert!(result.is_ok());

        let html = result.expect("result should be Ok");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("TestAspect"));
        assert!(html.contains("mermaid"));
        assert!(html.contains("Properties"));
        assert!(html.contains("temperature"));
        assert!(html.contains("REQUIRED"));
        assert!(html.contains("OPTIONAL"));
        assert!(html.contains("COLLECTION"));
        assert!(html.contains("OxiRS SAMM"));
    }

    #[test]
    fn test_generate_diagram_all_formats() {
        let aspect = create_test_aspect();
        let style = DiagramStyle::default();

        // Test all format variants
        let formats = vec![
            DiagramFormat::Dot(style.clone()),
            DiagramFormat::Mermaid(style.clone()),
            DiagramFormat::PlantUml(style.clone()),
            DiagramFormat::HtmlReport(style.clone()),
        ];

        for format in formats {
            let result = generate_diagram(&aspect, format);
            assert!(result.is_ok(), "Failed to generate diagram");
            let output = result.expect("generation should succeed");
            assert!(!output.is_empty(), "Generated diagram is empty");
        }
    }

    #[test]
    fn test_sanitize_mermaid_id() {
        assert_eq!(sanitize_mermaid_id("test:name#value"), "test_name_value");
        assert_eq!(sanitize_mermaid_id("my-prop.name"), "my_prop_name");
        assert_eq!(sanitize_mermaid_id("simple"), "simple");
    }

    #[test]
    fn test_sanitize_plantuml_id() {
        assert_eq!(sanitize_plantuml_id("test:name#value"), "test_name_value");
        assert_eq!(sanitize_plantuml_id("my-prop.name"), "my_prop_name");
        assert_eq!(sanitize_plantuml_id("simple"), "simple");
    }

    #[test]
    fn test_diagram_with_empty_aspect() {
        let aspect = Aspect::new("urn:samm:test:1.0.0#EmptyAspect".to_string());
        let style = DiagramStyle::default();

        // Should work even with no properties
        let mermaid = generate_mermaid(&aspect, &style);
        assert!(mermaid.is_ok());

        let puml = generate_plantuml(&aspect, &style);
        assert!(puml.is_ok());

        let dot = generate_dot(&aspect, &style);
        assert!(dot.is_ok());

        let html = generate_html_report(&aspect, &style);
        assert!(html.is_ok());
    }

    #[test]
    fn test_different_color_schemes() {
        let aspect = create_test_aspect();

        let schemes = vec!["default", "minimal", "colorful"];
        for scheme in schemes {
            let style = DiagramStyle {
                show_types: true,
                show_optionality: true,
                show_cardinality: true,
                include_metadata: false,
                color_scheme: scheme.to_string(),
                layout_direction: "LR".to_string(),
            };

            let result = generate_mermaid(&aspect, &style);
            assert!(result.is_ok());
        }
    }
}
