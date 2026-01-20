//! Enhanced Analytics Demo
//!
//! This example demonstrates the enhanced analytics features added in Session 24:
//! - Dependency impact analysis
//! - Cyclic dependency repair suggestions
//! - Graph comparison for model versioning
//! - Spearman and Kendall correlation analysis
//!
//! Run with: cargo run --example enhanced_analytics_demo

use oxirs_samm::analytics::ModelAnalytics;
use oxirs_samm::graph_analytics::{ChangeMagnitude, ModelGraph, RiskLevel};
use oxirs_samm::metamodel::{Aspect, Characteristic, CharacteristicKind, Property};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Enhanced Analytics Demo ===\n");

    // Create two versions of a model to demonstrate all features
    let v1_aspect = create_vehicle_model_v1();
    let v2_aspect = create_vehicle_model_v2();

    // === 1. Dependency Impact Analysis ===
    println!("🎯 1. DEPENDENCY IMPACT ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let graph_v1 = ModelGraph::from_aspect(&v1_aspect)?;

    // Analyze impact of changing the root aspect
    let aspect_impact = graph_v1.analyze_impact("VehicleAspect")?;
    println!("\n📊 Impact Analysis for 'VehicleAspect':");
    println!("  - Source node: {}", aspect_impact.source_node);
    println!(
        "  - Direct dependents: {}",
        aspect_impact.direct_dependents.len()
    );
    println!(
        "  - Total affected nodes: {}",
        aspect_impact.all_dependents.len()
    );
    println!(
        "  - Impact score: {:.1}%",
        aspect_impact.impact_score * 100.0
    );
    println!("  - Risk level: {:?}", aspect_impact.risk_level);

    if !aspect_impact.direct_dependents.is_empty() {
        println!("\n  Direct dependencies:");
        for dep in &aspect_impact.direct_dependents {
            println!("    • {}", dep);
        }
    }

    // Analyze impact of a property
    let property_impact = graph_v1.analyze_impact("VehicleSpeed")?;
    println!("\n📊 Impact Analysis for 'VehicleSpeed':");
    println!("  - Source node: {}", property_impact.source_node);
    println!(
        "  - Direct dependents: {}",
        property_impact.direct_dependents.len()
    );
    println!(
        "  - Total affected nodes: {}",
        property_impact.all_dependents.len()
    );
    println!(
        "  - Impact score: {:.1}%",
        property_impact.impact_score * 100.0
    );
    println!("  - Risk level: {:?}", property_impact.risk_level);

    // === 2. Cyclic Dependency Detection and Repair ===
    println!("\n\n🔄 2. CYCLIC DEPENDENCY DETECTION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let has_cycles = graph_v1.has_cycles()?;
    println!("\n  Circular dependencies detected: {}", has_cycles);

    if has_cycles {
        println!("\n  🔧 Suggested fixes:");
        let suggestions = graph_v1.suggest_cycle_breaks()?;
        for (i, suggestion) in suggestions.iter().enumerate() {
            println!("\n  Suggestion #{}:", i + 1);
            println!("    Edge to remove: {:?}", suggestion.edge_to_remove);
            println!("    Reason: {}", suggestion.reason);
            println!("    Impact: {}", suggestion.impact);
            println!("    Alternatives:");
            for alt in &suggestion.alternatives {
                println!("      • {}", alt);
            }
        }
    } else {
        println!("  ✅ No circular dependencies found - model structure is healthy!");
    }

    // === 3. Graph Comparison (Model Versioning) ===
    println!("\n\n🔍 3. MODEL VERSIONING - GRAPH COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let graph_v2 = ModelGraph::from_aspect(&v2_aspect)?;
    let comparison = graph_v1.compare(&graph_v2)?;

    println!("\n  Comparing Vehicle Model v1.0 → v2.0:");
    println!(
        "  ├─ Similarity score: {:.1}%",
        comparison.similarity_score * 100.0
    );
    println!("  ├─ Change magnitude: {:?}", comparison.change_magnitude);
    println!("  ├─ Nodes added: {}", comparison.added_nodes.len());
    println!("  ├─ Nodes removed: {}", comparison.removed_nodes.len());
    println!("  ├─ Edges added: {}", comparison.added_edges.len());
    println!("  └─ Edges removed: {}", comparison.removed_edges.len());

    if !comparison.added_nodes.is_empty() {
        println!("\n  ➕ Added nodes:");
        for node in &comparison.added_nodes {
            println!("    • {}", node);
        }
    }

    if !comparison.removed_nodes.is_empty() {
        println!("\n  ➖ Removed nodes:");
        for node in &comparison.removed_nodes {
            println!("    • {}", node);
        }
    }

    // Interpretation based on change magnitude
    println!("\n  📝 Change assessment:");
    match comparison.change_magnitude {
        ChangeMagnitude::Minimal => {
            println!("    ✅ Minimal changes - safe to upgrade");
        }
        ChangeMagnitude::Minor => {
            println!("    ℹ️  Minor changes - review recommended");
        }
        ChangeMagnitude::Moderate => {
            println!("    ⚠️  Moderate changes - careful review required");
        }
        ChangeMagnitude::Major => {
            println!("    🔶 Major changes - thorough testing needed");
        }
        ChangeMagnitude::Extensive => {
            println!("    🔴 Extensive changes - migration plan recommended");
        }
    }

    // === 4. Advanced Correlation Analysis ===
    println!("\n\n📊 4. ADVANCED CORRELATION ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let analytics_v1 = ModelAnalytics::analyze(&v1_aspect);

    // Pearson correlation (parametric - assumes linear relationships)
    let pearson = analytics_v1.compute_property_correlations();
    println!("\n  📈 Pearson Correlation (Linear relationships):");
    println!("    Features analyzed: {:?}", pearson.feature_names);
    println!("    Significant insights: {}", pearson.insights.len());

    if !pearson.insights.is_empty() {
        println!("\n    Key correlations:");
        for insight in pearson.insights.iter().take(3) {
            println!(
                "      • {} ↔ {}: {:.3} ({:?}, {:?})",
                insight.feature1,
                insight.feature2,
                insight.coefficient,
                insight.strength,
                insight.direction
            );
            println!("        → {}", insight.interpretation);
        }
    }

    // Spearman correlation (non-parametric - monotonic relationships)
    let spearman = analytics_v1.compute_spearman_correlations();
    println!("\n  📊 Spearman Correlation (Monotonic relationships):");
    println!(
        "    Method: {} (rank-based, robust to outliers)",
        spearman.method
    );
    println!("    Significant insights: {}", spearman.insights.len());

    if !spearman.insights.is_empty() {
        println!("\n    Top correlations:");
        for insight in spearman.insights.iter().take(3) {
            println!(
                "      • {} ↔ {}: {:.3} ({:?})",
                insight.feature1, insight.feature2, insight.coefficient, insight.strength
            );
        }
    }

    // Kendall correlation (non-parametric - ordinal associations)
    let kendall = analytics_v1.compute_kendall_correlations();
    println!("\n  🎲 Kendall Tau Correlation (Ordinal associations):");
    println!(
        "    Method: {} (concordant pairs, very robust)",
        kendall.method
    );
    println!("    Significant insights: {}", kendall.insights.len());

    if !kendall.insights.is_empty() {
        println!("\n    Notable patterns:");
        for insight in kendall.insights.iter().take(3) {
            println!(
                "      • {} ↔ {}: {:.3}",
                insight.feature1, insight.feature2, insight.coefficient
            );
        }
    }

    // === 5. Partial Correlation Analysis ===
    println!("\n\n🎯 5. PARTIAL CORRELATION ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let partial = analytics_v1.compute_partial_correlations();
    println!("\n  📐 Partial Correlation (Controlling for other features):");
    println!("    Method: {}", partial.method);
    println!("    Purpose: Remove confounding effects, reveal true relationships");
    println!("    Significant insights: {}", partial.insights.len());

    if !partial.insights.is_empty() {
        println!("\n    Direct relationships (controlling for confounds):");
        for insight in partial.insights.iter().take(3) {
            println!(
                "      • {} ↔ {}: {:.3}",
                insight.feature1, insight.feature2, insight.coefficient
            );
            println!("        → {}", insight.interpretation);
        }
    }

    println!("\n  💡 Insight:");
    println!("    Partial correlation helps identify which relationships are");
    println!("    genuine vs. those caused by third-variable confounding.");

    // === 6. Distribution Fitting ===
    println!("\n\n📊 6. DISTRIBUTION FITTING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let distributions = analytics_v1.fit_distributions();
    println!("\n  🔬 Statistical Distribution Analysis:");
    println!("    Metrics analyzed: {}", distributions.len());

    println!("\n    Fitted distributions:");
    for dist in &distributions {
        println!("\n    📈 {}:", dist.metric_name);
        println!("      ├─ Distribution type: {:?}", dist.distribution_type);
        println!(
            "      ├─ Goodness-of-fit: {:.3} ({:?} confidence)",
            dist.goodness_of_fit, dist.confidence
        );
        match dist.distribution_type {
            oxirs_samm::analytics::DistributionType::Normal => {
                println!(
                    "      ├─ Parameters: μ={:.2}, σ={:.2}",
                    dist.parameters.param1,
                    dist.parameters.param2.unwrap_or(0.0)
                );
                println!("      └─ Interpretation: Symmetric, bell-shaped distribution");
            }
            oxirs_samm::analytics::DistributionType::Exponential => {
                println!("      ├─ Parameters: λ={:.3}", dist.parameters.param1);
                println!("      └─ Interpretation: Decay process, memoryless");
            }
            oxirs_samm::analytics::DistributionType::Uniform => {
                println!(
                    "      ├─ Range: [{:.2}, {:.2}]",
                    dist.parameters.lower_bound.unwrap_or(0.0),
                    dist.parameters.upper_bound.unwrap_or(0.0)
                );
                println!("      └─ Interpretation: Equal probability across range");
            }
            oxirs_samm::analytics::DistributionType::LogNormal => {
                println!(
                    "      ├─ Parameters: μ={:.2}, σ={:.2}",
                    dist.parameters.param1,
                    dist.parameters.param2.unwrap_or(0.0)
                );
                println!("      └─ Interpretation: Skewed right, multiplicative process");
            }
            _ => {}
        }
    }

    // === 7. Method Comparison ===
    println!("\n\n🔬 7. CORRELATION METHOD COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    println!("\n  When to use each method:");
    println!("  ┌─ Pearson:   Linear relationships, normal distribution");
    println!("  ├─ Spearman:  Monotonic relationships, ranked data, outliers present");
    println!("  ├─ Kendall:   Small samples, ordinal data, very robust to errors");
    println!("  └─ Partial:   Remove confounding effects, true direct relationships");

    println!("\n  Matrix diagonal (should all be 1.0 for self-correlation):");
    println!("    Pearson:  {:.3}", pearson.correlation_matrix[0][0]);
    println!("    Spearman: {:.3}", spearman.correlation_matrix[0][0]);
    println!("    Kendall:  {:.3}", kendall.correlation_matrix[0][0]);
    println!("    Partial:  {:.3}", partial.correlation_matrix[0][0]);

    // === Summary ===
    println!("\n\n📋 SUMMARY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n  ✅ Enhanced analytics features demonstrated:");
    println!("    1. Dependency impact analysis with risk levels");
    println!("    2. Cycle detection and repair suggestions");
    println!("    3. Model versioning with graph comparison");
    println!("    4. Four correlation methods (Pearson, Spearman, Kendall, Partial)");
    println!("    5. Distribution fitting for understanding metric patterns");
    println!("\n  All features are production-ready and fully tested! 🎉\n");

    Ok(())
}

/// Create Vehicle Model v1.0
fn create_vehicle_model_v1() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:example:1.0.0#VehicleAspect".to_string());

    // Add speed property
    let speed_char = Characteristic {
        metadata: oxirs_samm::metamodel::ElementMetadata::new(
            "urn:samm:example:1.0.0#SpeedCharacteristic".to_string(),
        ),
        data_type: Some("float".to_string()),
        kind: CharacteristicKind::Measurement {
            unit: "km/h".to_string(),
        },
        constraints: vec![],
    };
    let speed_property = Property::new("urn:samm:example:1.0.0#VehicleSpeed".to_string())
        .with_characteristic(speed_char);

    // Add position property
    let position_char = Characteristic {
        metadata: oxirs_samm::metamodel::ElementMetadata::new(
            "urn:samm:example:1.0.0#PositionCharacteristic".to_string(),
        ),
        data_type: Some("geo:Point".to_string()),
        kind: CharacteristicKind::Trait,
        constraints: vec![],
    };
    let position_property = Property::new("urn:samm:example:1.0.0#Position".to_string())
        .with_characteristic(position_char);

    // Add fuel level property
    let fuel_char = Characteristic {
        metadata: oxirs_samm::metamodel::ElementMetadata::new(
            "urn:samm:example:1.0.0#FuelCharacteristic".to_string(),
        ),
        data_type: Some("float".to_string()),
        kind: CharacteristicKind::Measurement {
            unit: "liters".to_string(),
        },
        constraints: vec![],
    };
    let fuel_property = Property::new("urn:samm:example:1.0.0#FuelLevel".to_string())
        .with_characteristic(fuel_char);

    aspect.add_property(speed_property);
    aspect.add_property(position_property);
    aspect.add_property(fuel_property);

    aspect
}

/// Create Vehicle Model v2.0 (with additional battery property)
fn create_vehicle_model_v2() -> Aspect {
    let mut aspect = create_vehicle_model_v1();

    // Add battery level property (new in v2.0)
    let battery_char = Characteristic {
        metadata: oxirs_samm::metamodel::ElementMetadata::new(
            "urn:samm:example:2.0.0#BatteryCharacteristic".to_string(),
        ),
        data_type: Some("float".to_string()),
        kind: CharacteristicKind::Measurement {
            unit: "percent".to_string(),
        },
        constraints: vec![],
    };
    let battery_property = Property::new("urn:samm:example:2.0.0#BatteryLevel".to_string())
        .with_characteristic(battery_char);

    aspect.add_property(battery_property);

    aspect
}
