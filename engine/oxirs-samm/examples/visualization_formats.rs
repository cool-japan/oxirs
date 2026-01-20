//! Visualization Formats Example
//!
//! Demonstrates all available diagram generation formats in oxirs-samm:
//! - Mermaid.js (for GitHub/markdown)
//! - PlantUML (for documentation)
//! - Graphviz DOT (for SVG/PNG rendering)
//! - HTML reports (standalone interactive reports)
//!
//! Run this example with:
//! ```bash
//! cargo run --example visualization_formats
//! ```

use oxirs_samm::generators::diagram::{generate_diagram, DiagramFormat, DiagramStyle};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, ModelElement, Property,
};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 SAMM Visualization Formats Demo\n");
    println!("This example demonstrates all visualization formats available in oxirs-samm");
    println!("{}", "=".repeat(80));
    println!();

    // Create a sample SAMM aspect model
    let aspect = create_sample_aspect();

    // Create output directory
    let output_dir = std::env::temp_dir().join("oxirs_samm_visualization_demo");
    fs::create_dir_all(&output_dir)?;
    println!("📁 Output directory: {}\n", output_dir.display());

    // Example 1: Mermaid.js Diagram (default style)
    example_1_mermaid_default(&aspect, &output_dir)?;

    // Example 2: Mermaid.js with top-down layout
    example_2_mermaid_topdown(&aspect, &output_dir)?;

    // Example 3: Mermaid.js minimal style
    example_3_mermaid_minimal(&aspect, &output_dir)?;

    // Example 4: PlantUML diagram
    example_4_plantuml(&aspect, &output_dir)?;

    // Example 5: PlantUML with metadata
    example_5_plantuml_metadata(&aspect, &output_dir)?;

    // Example 6: Graphviz DOT
    example_6_dot(&aspect, &output_dir)?;

    // Example 7: HTML Report (standalone)
    example_7_html_report(&aspect, &output_dir)?;

    // Example 8: All formats side-by-side
    example_8_comparison(&aspect, &output_dir)?;

    println!();
    println!("{}", "=".repeat(80));
    println!("✅ All examples completed successfully!");
    println!("📂 Check the output directory for generated files");

    Ok(())
}

/// Create a sample SAMM aspect for demonstration
fn create_sample_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:com.example:1.0.0#VehicleStatus".to_string());

    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Vehicle Status".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "Represents the current status and telemetry of a vehicle".to_string(),
    );

    // Add temperature property (required, measurement)
    let mut temp_prop = Property::new("urn:samm:com.example:1.0.0#engineTemperature".to_string());
    temp_prop.optional = false;
    temp_prop.is_collection = false;
    temp_prop.characteristic = Some(Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:com.example:1.0.0#TemperatureCharacteristic".to_string(),
        ),
        kind: CharacteristicKind::Measurement {
            unit: "unit:degreeCelsius".to_string(),
        },
        data_type: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
        constraints: vec![],
    });
    temp_prop
        .metadata
        .add_preferred_name("en".to_string(), "Engine Temperature".to_string());
    temp_prop.metadata.add_description(
        "en".to_string(),
        "Current engine temperature in Celsius".to_string(),
    );
    aspect.properties.push(temp_prop);

    // Add speed property (required, measurement)
    let mut speed_prop = Property::new("urn:samm:com.example:1.0.0#currentSpeed".to_string());
    speed_prop.optional = false;
    speed_prop.characteristic = Some(Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:com.example:1.0.0#SpeedCharacteristic".to_string(),
        ),
        kind: CharacteristicKind::Measurement {
            unit: "unit:kilometrePerHour".to_string(),
        },
        data_type: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
        constraints: vec![],
    });
    speed_prop
        .metadata
        .add_preferred_name("en".to_string(), "Current Speed".to_string());
    aspect.properties.push(speed_prop);

    // Add status property (optional, enumeration)
    let mut status_prop = Property::new("urn:samm:com.example:1.0.0#vehicleStatus".to_string());
    status_prop.optional = true;
    status_prop.characteristic = Some(Characteristic {
        metadata: ElementMetadata::new("urn:samm:com.example:1.0.0#VehicleStatusEnum".to_string()),
        kind: CharacteristicKind::Enumeration {
            values: vec![
                "RUNNING".to_string(),
                "STOPPED".to_string(),
                "IDLE".to_string(),
            ],
        },
        data_type: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
        constraints: vec![],
    });
    status_prop
        .metadata
        .add_preferred_name("en".to_string(), "Vehicle Status".to_string());
    aspect.properties.push(status_prop);

    // Add diagnostic codes (collection)
    let mut diag_prop = Property::new("urn:samm:com.example:1.0.0#diagnosticCodes".to_string());
    diag_prop.is_collection = true;
    diag_prop.optional = true;
    diag_prop.characteristic = Some(Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:com.example:1.0.0#DiagnosticCodesList".to_string(),
        ),
        kind: CharacteristicKind::Collection {
            element_characteristic: None,
        },
        data_type: Some("http://www.w3.org/2001/XMLSchema#string".to_string()),
        constraints: vec![],
    });
    diag_prop
        .metadata
        .add_preferred_name("en".to_string(), "Diagnostic Codes".to_string());
    aspect.properties.push(diag_prop);

    // Add fuel level (required, measurement)
    let mut fuel_prop = Property::new("urn:samm:com.example:1.0.0#fuelLevel".to_string());
    fuel_prop.optional = false;
    fuel_prop.characteristic = Some(Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:com.example:1.0.0#PercentageCharacteristic".to_string(),
        ),
        kind: CharacteristicKind::Measurement {
            unit: "unit:percent".to_string(),
        },
        data_type: Some("http://www.w3.org/2001/XMLSchema#decimal".to_string()),
        constraints: vec![],
    });
    fuel_prop
        .metadata
        .add_preferred_name("en".to_string(), "Fuel Level".to_string());
    aspect.properties.push(fuel_prop);

    aspect
}

fn example_1_mermaid_default(
    aspect: &Aspect,
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 1: Mermaid.js Diagram (Default Style)");
    println!("   - Shows all information: types, optionality, cardinality");
    println!("   - Left-to-right layout");

    let style = DiagramStyle::default();
    let diagram = generate_diagram(aspect, DiagramFormat::Mermaid(style))?;

    let output_path = output_dir.join("mermaid_default.md");
    fs::write(&output_path, format!("```mermaid\n{}\n```\n", diagram))?;

    println!("   ✅ Saved to: {}", output_path.display());
    println!("   💡 Tip: This format works great in GitHub README files!\n");

    Ok(())
}

fn example_2_mermaid_topdown(
    aspect: &Aspect,
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 2: Mermaid.js Diagram (Top-Down Layout)");
    println!("   - Vertical layout for hierarchical views");

    let style = DiagramStyle {
        layout_direction: "TB".to_string(),
        ..Default::default()
    };
    let diagram = generate_diagram(aspect, DiagramFormat::Mermaid(style))?;

    let output_path = output_dir.join("mermaid_topdown.md");
    fs::write(&output_path, format!("```mermaid\n{}\n```\n", diagram))?;

    println!("   ✅ Saved to: {}", output_path.display());
    println!();

    Ok(())
}

fn example_3_mermaid_minimal(
    aspect: &Aspect,
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 3: Mermaid.js Diagram (Minimal Style)");
    println!("   - Clean view with only property names");
    println!("   - No types, optionality, or cardinality indicators");

    let style = DiagramStyle {
        show_types: false,
        show_optionality: false,
        show_cardinality: false,
        include_metadata: false,
        color_scheme: "minimal".to_string(),
        layout_direction: "LR".to_string(),
    };
    let diagram = generate_diagram(aspect, DiagramFormat::Mermaid(style))?;

    let output_path = output_dir.join("mermaid_minimal.md");
    fs::write(&output_path, format!("```mermaid\n{}\n```\n", diagram))?;

    println!("   ✅ Saved to: {}", output_path.display());
    println!();

    Ok(())
}

fn example_4_plantuml(
    aspect: &Aspect,
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 4: PlantUML Diagram");
    println!("   - Structured component view");
    println!("   - [R] = Required, [O] = Optional, [*] = Collection");

    let style = DiagramStyle::default();
    let diagram = generate_diagram(aspect, DiagramFormat::PlantUml(style))?;

    let output_path = output_dir.join("plantuml_diagram.puml");
    fs::write(&output_path, diagram)?;

    println!("   ✅ Saved to: {}", output_path.display());
    println!("   💡 Tip: Use PlantUML server or CLI to render to PNG/SVG\n");

    Ok(())
}

fn example_5_plantuml_metadata(
    aspect: &Aspect,
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 5: PlantUML Diagram (With Metadata)");
    println!("   - Includes descriptions as notes");

    let style = DiagramStyle {
        include_metadata: true,
        ..Default::default()
    };
    let diagram = generate_diagram(aspect, DiagramFormat::PlantUml(style))?;

    let output_path = output_dir.join("plantuml_with_metadata.puml");
    fs::write(&output_path, diagram)?;

    println!("   ✅ Saved to: {}", output_path.display());
    println!();

    Ok(())
}

fn example_6_dot(
    aspect: &Aspect,
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 6: Graphviz DOT");
    println!("   - DOT format for Graphviz rendering");
    println!("   - Can be converted to SVG/PNG with: dot -Tsvg input.dot -o output.svg");

    let style = DiagramStyle::default();
    let diagram = generate_diagram(aspect, DiagramFormat::Dot(style))?;

    let output_path = output_dir.join("graphviz.dot");
    fs::write(&output_path, diagram)?;

    println!("   ✅ Saved to: {}", output_path.display());
    println!("   💡 Tip: Install Graphviz with: brew install graphviz\n");

    Ok(())
}

fn example_7_html_report(
    aspect: &Aspect,
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 7: HTML Report (Standalone)");
    println!("   - Interactive HTML report with embedded Mermaid.js diagram");
    println!("   - Includes statistics and detailed property information");
    println!("   - Styled with modern CSS");

    let style = DiagramStyle::default();
    let diagram = generate_diagram(aspect, DiagramFormat::HtmlReport(style))?;

    let output_path = output_dir.join("vehicle_status_report.html");
    fs::write(&output_path, diagram)?;

    println!("   ✅ Saved to: {}", output_path.display());
    println!("   💡 Tip: Open this file in your browser to see the interactive report\n");

    Ok(())
}

fn example_8_comparison(
    aspect: &Aspect,
    output_dir: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 8: Format Comparison");
    println!("   - Generates all formats for easy comparison");

    let style = DiagramStyle::default();

    // Generate all formats
    let mermaid = generate_diagram(aspect, DiagramFormat::Mermaid(style.clone()))?;
    let plantuml = generate_diagram(aspect, DiagramFormat::PlantUml(style.clone()))?;
    let dot = generate_diagram(aspect, DiagramFormat::Dot(style.clone()))?;
    let html = generate_diagram(aspect, DiagramFormat::HtmlReport(style))?;

    // Create comparison markdown
    let comparison = format!(
        r#"# SAMM Visualization Formats Comparison

## Aspect: {}

---

## 1. Mermaid.js (GitHub/Markdown)

```mermaid
{}
```

**Use cases:**
- GitHub README files
- GitLab documentation
- Markdown-based wikis
- Blog posts

---

## 2. PlantUML (Documentation)

```plantuml
{}
```

**Use cases:**
- Technical documentation
- Architecture diagrams
- PDF reports
- Confluence pages

---

## 3. Graphviz DOT (Rendering)

```dot
{}
```

**Use cases:**
- High-quality SVG/PNG exports
- Print materials
- Presentations
- Custom styling

---

## 4. HTML Report

The HTML report is saved as a separate file: `vehicle_status_report.html`

**Use cases:**
- Standalone documentation
- Model reviews
- Stakeholder presentations
- Web-based documentation portals

---

## Format Comparison Table

| Format | Pros | Cons | Best For |
|--------|------|------|----------|
| **Mermaid.js** | ✅ Works in GitHub<br/>✅ Easy to edit<br/>✅ No tools needed | ⚠️ Limited customization | Quick documentation, READMEs |
| **PlantUML** | ✅ Rich features<br/>✅ Multiple diagram types<br/>✅ Good tooling | ⚠️ Requires renderer | Technical docs, architecture |
| **Graphviz DOT** | ✅ High quality output<br/>✅ Extensive customization<br/>✅ SVG/PNG export | ⚠️ Requires installation | Print, presentations |
| **HTML Report** | ✅ Interactive<br/>✅ Standalone<br/>✅ Rich metadata | ⚠️ Larger file size | Reviews, portals |

---

Generated with OxiRS SAMM • SAMM Specification 2.3.0
"#,
        aspect.name(),
        mermaid,
        plantuml,
        dot
    );

    let output_path = output_dir.join("FORMAT_COMPARISON.md");
    fs::write(&output_path, comparison)?;

    // Save HTML separately
    let html_path = output_dir.join("vehicle_status_report.html");
    fs::write(&html_path, html)?;

    println!("   ✅ Saved comparison to: {}", output_path.display());
    println!("   ✅ Saved HTML report to: {}", html_path.display());
    println!();

    Ok(())
}
