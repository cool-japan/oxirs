//! # Model Comparison Example
//!
//! This example demonstrates the model comparison and diffing capabilities:
//! 1. Create two versions of a model
//! 2. Compare them to detect changes
//! 3. Identify breaking changes
//! 4. Generate human-readable diff reports
//! 5. Generate visual comparison diagrams (HTML, Mermaid.js)
//! 6. Create side-by-side visual diffs
//!
//! Run with: `cargo run --example model_comparison`

use oxirs_samm::comparison::ModelComparison;
use oxirs_samm::metamodel::{Aspect, Characteristic, CharacteristicKind, ModelElement, Property};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SAMM Model Comparison Example ===\n");

    // Step 1: Create original model (v1.0.0)
    println!("Step 1: Creating original model (v1.0.0)...");
    let original = create_v1_aspect();
    println!("✓ Original model created");
    println!("  Version: 1.0.0");
    println!("  Properties: {}", original.properties().len());
    for prop in original.properties() {
        println!("    - {}: optional={}", prop.name(), prop.optional);
    }
    println!();

    // Step 2: Create modified model (v2.0.0)
    println!("Step 2: Creating modified model (v2.0.0)...");
    let modified = create_v2_aspect();
    println!("✓ Modified model created");
    println!("  Version: 2.0.0");
    println!("  Properties: {}", modified.properties().len());
    for prop in modified.properties() {
        println!("    - {}: optional={}", prop.name(), prop.optional);
    }
    println!();

    // Step 3: Compare models
    println!("Step 3: Comparing models...");
    let comparison = ModelComparison::compare(&original, &modified);
    println!("✓ Comparison complete");
    println!();

    // Step 4: Analyze changes
    println!("Step 4: Analyzing changes...");
    println!("  Added properties: {}", comparison.properties_added.len());
    for prop_name in &comparison.properties_added {
        println!("    + {}", prop_name);
    }
    println!();

    println!(
        "  Removed properties: {}",
        comparison.properties_removed.len()
    );
    for prop_name in &comparison.properties_removed {
        println!("    - {}", prop_name);
    }
    println!();

    println!(
        "  Modified properties: {}",
        comparison.properties_modified.len()
    );
    for (prop_name, changes) in &comparison.properties_modified {
        println!("    ~ {}", prop_name);
        if changes.optional_changed {
            println!("      - Optional flag changed");
        }
        if changes.characteristic_changed {
            println!("      - Characteristic changed");
        }
    }
    println!();

    // Step 5: Check for breaking changes
    println!("Step 5: Checking for breaking changes...");
    let has_breaking = comparison.has_breaking_changes();
    println!("  Breaking changes detected: {}", has_breaking);
    if has_breaking {
        let breaking = comparison.get_breaking_changes();
        println!("  Breaking changes:");
        for change in &breaking {
            println!("    ⚠️  {}", change);
        }
    }
    println!();

    // Step 6: Generate diff report
    println!("Step 6: Generating diff report...");
    let report = comparison.generate_report();
    println!("{}", report);

    // Step 7: Compare metadata changes
    println!("Step 7: Analyzing metadata changes...");
    if !comparison.metadata_changes.is_empty() {
        println!("  Metadata changes detected:");
        for change in &comparison.metadata_changes {
            println!("    - {:?}", change);
        }
    } else {
        println!("  No metadata changes");
    }
    println!();

    // Step 8: Compare operations
    println!("Step 8: Analyzing operation changes...");
    println!("  Added operations: {}", comparison.operations_added.len());
    for op_name in &comparison.operations_added {
        println!("    + Operation: {}", op_name);
    }

    println!(
        "  Removed operations: {}",
        comparison.operations_removed.len()
    );
    for op_name in &comparison.operations_removed {
        println!("    - Operation: {}", op_name);
    }
    println!();

    // Step 9: Use case - Version control workflow
    println!("Step 9: Use case - Version control workflow...");
    println!("  Scenario: Reviewing changes before release");
    println!();

    if has_breaking {
        println!("  ⚠️  ATTENTION: This release contains breaking changes!");
        println!("  Consider:");
        println!("    - Incrementing major version");
        println!("    - Providing migration guide");
        println!("    - Notifying API consumers");
        println!();
    } else {
        println!("  ✓ No breaking changes detected");
        println!("  This can be released as a minor/patch version");
        println!();
    }

    // Step 10: Generate visual comparison diagrams
    println!("Step 10: Generating visual comparison diagrams...");

    // Create output directory
    let output_dir = std::env::temp_dir().join("oxirs_model_comparison");
    fs::create_dir_all(&output_dir)?;
    println!("  Output directory: {}", output_dir.display());
    println!();

    // Generate HTML visual diff report
    println!("  Generating HTML visual diff report...");
    let html_report = comparison.generate_visual_diff_html(&original, &modified)?;
    let html_path = output_dir.join("comparison_report.html");
    fs::write(&html_path, html_report)?;
    println!("  ✓ Saved HTML report: {}", html_path.display());
    println!("    💡 Open this file in your browser to see interactive comparison");
    println!();

    // Generate Mermaid comparison diagram
    println!("  Generating Mermaid.js comparison diagram...");
    let mermaid_diagram = comparison.generate_mermaid_comparison(&original, &modified)?;
    let mermaid_path = output_dir.join("comparison.mermaid.md");
    fs::write(
        &mermaid_path,
        format!("```mermaid\n{}\n```\n", mermaid_diagram),
    )?;
    println!("  ✓ Saved Mermaid diagram: {}", mermaid_path.display());
    println!("    💡 This format works in GitHub README files");
    println!();

    // Step 11: Demonstrate visual diff features
    println!("Step 11: Visual diff features...");
    println!("  The HTML report includes:");
    println!("    • Side-by-side before/after diagrams");
    println!("    • Color-coded change indicators:");
    println!("      - 🟢 Green: Added properties");
    println!("      - 🔴 Red: Removed properties");
    println!("      - 🟡 Orange: Modified properties");
    println!("      - ⚪ Gray: Unchanged properties");
    println!("    • Interactive Mermaid.js diagrams");
    println!("    • Change statistics dashboard");
    println!("    • Breaking changes warning (if applicable)");
    println!("    • Detailed property-level changes");
    println!();

    println!("  The Mermaid diagram shows:");
    println!("    • ❌ Removed elements");
    println!("    • ✅ Added elements");
    println!("    • 🔄 Modified elements");
    println!("    • Connections showing unchanged elements");
    println!();

    // Step 12: Integration scenarios
    println!("Step 12: Integration scenarios...");
    println!("  Visual comparison can be used for:");
    println!("    📝 Code review processes");
    println!("    📚 Documentation generation");
    println!("    🔄 CI/CD pipeline validation");
    println!("    📊 Change impact analysis");
    println!("    🎓 Training and onboarding");
    println!("    📈 Model evolution tracking");
    println!();

    // Summary
    println!("=== Comparison Summary ===");
    println!("Comparison capabilities demonstrated:");
    println!("  ✓ Property-level change detection");
    println!("  ✓ Breaking change identification");
    println!("  ✓ Metadata comparison");
    println!("  ✓ Operation comparison");
    println!("  ✓ Human-readable diff reports");
    println!("  ✓ Visual comparison diagrams (HTML, Mermaid.js)");
    println!("  ✓ Side-by-side visual diffs");
    println!("  ✓ Color-coded change highlighting");
    println!("  ✓ Interactive HTML reports");
    println!("  ✓ Version control integration patterns");
    println!();
    println!("The comparison API enables safe model evolution and version management.");
    println!();
    println!("📂 Generated files:");
    println!("  • {}", html_path.display());
    println!("  • {}", mermaid_path.display());

    Ok(())
}

/// Create version 1.0.0 of the aspect
fn create_v1_aspect() -> Aspect {
    let mut id_prop = Property::new("urn:samm:com.example:1.0.0#id".to_string());
    id_prop.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:1.0.0#Id".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = id_prop.characteristic {
        char.data_type = Some("xsd:string".to_string());
    }
    id_prop.example_values = vec!["12345".to_string()];
    id_prop.optional = false;

    let mut name_prop = Property::new("urn:samm:com.example:1.0.0#name".to_string());
    name_prop.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:1.0.0#Text".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = name_prop.characteristic {
        char.data_type = Some("xsd:string".to_string());
    }
    name_prop.example_values = vec!["Sample".to_string()];
    name_prop.optional = false;

    let mut description_prop = Property::new("urn:samm:com.example:1.0.0#description".to_string());
    description_prop.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:1.0.0#Text".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = description_prop.characteristic {
        char.data_type = Some("xsd:string".to_string());
    }
    description_prop.example_values = vec!["A sample description".to_string()];
    description_prop.optional = true;

    let mut aspect = Aspect::new("urn:samm:com.example:1.0.0#MyModel".to_string());
    aspect.properties = vec![id_prop, name_prop, description_prop];

    aspect
        .metadata
        .add_preferred_name("en".to_string(), "My Model".to_string());
    aspect
        .metadata
        .add_description("en".to_string(), "Original model version".to_string());

    aspect
}

/// Create version 2.0.0 of the aspect with changes
fn create_v2_aspect() -> Aspect {
    // Keep id (unchanged)
    let mut id_prop = Property::new("urn:samm:com.example:2.0.0#id".to_string());
    id_prop.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:2.0.0#Id".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = id_prop.characteristic {
        char.data_type = Some("xsd:string".to_string());
    }
    id_prop.example_values = vec!["12345".to_string()];
    id_prop.optional = false;

    // Rename 'name' to 'title' (this counts as remove + add)
    let mut title_prop = Property::new("urn:samm:com.example:2.0.0#title".to_string());
    title_prop.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:2.0.0#Text".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = title_prop.characteristic {
        char.data_type = Some("xsd:string".to_string());
    }
    title_prop.example_values = vec!["Sample Title".to_string()];
    title_prop.optional = false;

    // Make description required (breaking change: optional -> required)
    let mut description_prop = Property::new("urn:samm:com.example:2.0.0#description".to_string());
    description_prop.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:2.0.0#Text".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = description_prop.characteristic {
        char.data_type = Some("xsd:string".to_string());
    }
    description_prop.example_values = vec!["A required description".to_string()];
    description_prop.optional = false; // Changed from true to false

    // Add new property
    let mut status_prop = Property::new("urn:samm:com.example:2.0.0#status".to_string());
    status_prop.characteristic = Some(Characteristic::new(
        "urn:samm:com.example:2.0.0#Status".to_string(),
        CharacteristicKind::Trait,
    ));
    if let Some(ref mut char) = status_prop.characteristic {
        char.data_type = Some("xsd:string".to_string());
    }
    status_prop.example_values = vec!["active".to_string()];
    status_prop.optional = true;

    let mut aspect = Aspect::new("urn:samm:com.example:2.0.0#MyModel".to_string());
    aspect.properties = vec![id_prop, title_prop, description_prop, status_prop];

    aspect
        .metadata
        .add_preferred_name("en".to_string(), "My Model v2".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "Enhanced model version with new features".to_string(),
    );

    aspect
}
