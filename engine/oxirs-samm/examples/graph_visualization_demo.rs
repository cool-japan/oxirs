//! Graph Visualization Demo
//!
//! This example demonstrates how to generate visual representations of SAMM model dependency graphs.
//! It shows DOT format generation for use with Graphviz, and optionally SVG/PNG rendering
//! if the `graphviz` feature is enabled.

use oxirs_samm::graph_analytics::{ColorScheme, ModelGraph, VisualizationStyle};
use oxirs_samm::metamodel::{
    Aspect, BoundDefinition, Characteristic, CharacteristicKind, Constraint, ElementMetadata,
    Property,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🎨 SAMM Model Graph Visualization Demo\n");
    println!("{}", "=".repeat(70));

    // Create a sample SAMM model
    let aspect = create_sample_aspect();
    println!("\n📦 Created Sample Aspect Model:");
    println!("   Aspect: {}", aspect.metadata.urn);
    println!("   Properties: {}", aspect.properties.len());

    // Build dependency graph
    println!("\n🔗 Building Dependency Graph...");
    let graph = ModelGraph::from_aspect(&aspect)?;
    println!("   ✓ Graph constructed successfully");
    println!("   Nodes: {}", graph.num_nodes());
    println!("   Edges: {}", graph.num_edges());

    // Generate DOT format visualizations
    println!("\n📊 Generating DOT Format Visualizations...");

    // 1. Compact style
    println!("\n   1. Compact Style (minimal labels):");
    let dot_compact = graph.to_dot(VisualizationStyle::Compact)?;
    std::fs::write("vehicle_model_compact.dot", &dot_compact)?;
    println!("      ✓ Saved to: vehicle_model_compact.dot");
    println!("      Preview:");
    print_preview(&dot_compact, 8);

    // 2. Detailed style (default)
    println!("\n   2. Detailed Style (full information):");
    let dot_detailed = graph.to_dot(VisualizationStyle::Detailed)?;
    std::fs::write("vehicle_model_detailed.dot", &dot_detailed)?;
    println!("      ✓ Saved to: vehicle_model_detailed.dot");
    println!("      Preview:");
    print_preview(&dot_detailed, 8);

    // 3. Hierarchical style
    println!("\n   3. Hierarchical Style (top-down layout):");
    let dot_hierarchical = graph.to_dot(VisualizationStyle::Hierarchical)?;
    std::fs::write("vehicle_model_hierarchical.dot", &dot_hierarchical)?;
    println!("      ✓ Saved to: vehicle_model_hierarchical.dot");

    // 4. Custom color scheme
    println!("\n   4. Custom Color Scheme:");
    let custom_colors = ColorScheme {
        aspect_color: "#FFE6E6".to_string(),         // Light red
        property_color: "#E6F3FF".to_string(),       // Light blue
        characteristic_color: "#E6FFE6".to_string(), // Light green
        edge_color: "#333333".to_string(),           // Dark gray
    };
    let dot_custom = graph.to_dot_with_colors(VisualizationStyle::Detailed, custom_colors)?;
    std::fs::write("vehicle_model_custom_colors.dot", &dot_custom)?;
    println!("      ✓ Saved to: vehicle_model_custom_colors.dot");

    // Render to SVG/PNG if graphviz feature is enabled
    #[cfg(feature = "graphviz")]
    {
        println!("\n🖼️  Rendering to Image Files...");
        graph.render_svg("vehicle_model.svg", VisualizationStyle::Hierarchical)?;
        println!("   ✓ SVG: vehicle_model.svg");

        graph.render_png("vehicle_model.png", VisualizationStyle::Hierarchical)?;
        println!("   ✓ PNG: vehicle_model.png");
    }

    #[cfg(not(feature = "graphviz"))]
    {
        println!("\n💡 Tip: Enable the 'graphviz' feature to render SVG/PNG images:");
        println!("   cargo run --example graph_visualization_demo --features graphviz");
    }

    // Usage instructions
    println!("\n{}", "=".repeat(70));
    println!("\n📖 How to View the Generated Visualizations:\n");
    println!("1. Using Graphviz command-line:");
    println!("   dot -Tsvg vehicle_model_detailed.dot -o output.svg");
    println!("   dot -Tpng vehicle_model_detailed.dot -o output.png");
    println!("   dot -Tpdf vehicle_model_detailed.dot -o output.pdf");
    println!("\n2. Online tools:");
    println!("   • GraphvizOnline: https://dreampuf.github.io/GraphvizOnline/");
    println!("   • Edotor: https://edotor.net/");
    println!("\n3. Desktop applications:");
    println!("   • Install Graphviz: https://graphviz.org/download/");
    println!("   • Use xdot (interactive viewer)");

    println!("\n{}", "=".repeat(70));
    println!("\n✅ Visualization Demo Complete!\n");
    println!("Generated {} DOT files in the current directory", 4);

    Ok(())
}

/// Create a sample SAMM aspect model with various characteristics
fn create_sample_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#VehicleTelemetry".to_string());

    // Add metadata
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Vehicle Telemetry".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "Real-time vehicle telemetry and status information".to_string(),
    );

    // Property 1: Speed with Quantifiable characteristic
    let speed_char = Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:org.example:1.0.0#SpeedCharacteristic".to_string(),
        ),
        data_type: Some("xsd:float".to_string()),
        kind: CharacteristicKind::Quantifiable {
            unit: "kilometer-per-hour".to_string(),
        },
        constraints: vec![Constraint::RangeConstraint {
            min_value: Some("0".to_string()),
            max_value: Some("300".to_string()),
            lower_bound_definition: BoundDefinition::AtLeast,
            upper_bound_definition: BoundDefinition::LessThan,
        }],
    };
    let mut speed_property = Property::new("urn:samm:org.example:1.0.0#Speed".to_string())
        .with_characteristic(speed_char);
    speed_property
        .metadata
        .add_preferred_name("en".to_string(), "Speed".to_string());
    aspect.add_property(speed_property);

    // Property 2: Location
    let location_char = Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:org.example:1.0.0#LocationCharacteristic".to_string(),
        ),
        data_type: Some("xsd:string".to_string()),
        kind: CharacteristicKind::Trait,
        constraints: vec![],
    };
    let location_property = Property::new("urn:samm:org.example:1.0.0#Location".to_string())
        .with_characteristic(location_char);
    aspect.add_property(location_property);

    // Property 3: Status with Enumeration
    let status_char = Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:org.example:1.0.0#StatusCharacteristic".to_string(),
        ),
        data_type: Some("xsd:string".to_string()),
        kind: CharacteristicKind::Enumeration {
            values: vec![
                "Active".to_string(),
                "Idle".to_string(),
                "Maintenance".to_string(),
                "Emergency".to_string(),
            ],
        },
        constraints: vec![],
    };
    let status_property = Property::new("urn:samm:org.example:1.0.0#Status".to_string())
        .with_characteristic(status_char);
    aspect.add_property(status_property);

    // Property 4: FuelLevel with Measurement
    let fuel_char = Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:org.example:1.0.0#FuelLevelCharacteristic".to_string(),
        ),
        data_type: Some("xsd:integer".to_string()),
        kind: CharacteristicKind::Measurement {
            unit: "percent".to_string(),
        },
        constraints: vec![Constraint::RangeConstraint {
            min_value: Some("0".to_string()),
            max_value: Some("100".to_string()),
            lower_bound_definition: BoundDefinition::AtLeast,
            upper_bound_definition: BoundDefinition::AtLeast,
        }],
    };
    let fuel_property = Property::new("urn:samm:org.example:1.0.0#FuelLevel".to_string())
        .with_characteristic(fuel_char);
    aspect.add_property(fuel_property);

    // Property 5: Timestamp
    let timestamp_char = Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:org.example:1.0.0#TimestampCharacteristic".to_string(),
        ),
        data_type: Some("xsd:dateTime".to_string()),
        kind: CharacteristicKind::Trait,
        constraints: vec![],
    };
    let timestamp_property = Property::new("urn:samm:org.example:1.0.0#Timestamp".to_string())
        .with_characteristic(timestamp_char);
    aspect.add_property(timestamp_property);

    aspect
}

/// Print a preview of the DOT content (first N lines)
fn print_preview(dot: &str, lines: usize) {
    for (i, line) in dot.lines().take(lines).enumerate() {
        println!("         {}", line);
        if i == lines - 1 && dot.lines().count() > lines {
            println!("         ... ({} more lines)", dot.lines().count() - lines);
        }
    }
}
