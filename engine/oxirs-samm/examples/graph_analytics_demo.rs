//! Graph Analytics Demo
//!
//! This example demonstrates the graph analytics capabilities of oxirs-samm using scirs2-graph.
//! It shows how to analyze SAMM model dependencies, compute centrality metrics, detect communities,
//! and identify potential circular dependencies.

use oxirs_samm::graph_analytics::{CentralityMetrics, GraphMetrics, ModelGraph};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, ElementMetadata, Property,
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("🔍 SAMM Model Graph Analytics Demo\n");
    println!("Using scirs2-graph v0.1.0-rc.2 for advanced graph analysis\n");
    println!("{}", "=".repeat(70));

    // Create a sample SAMM aspect model with dependencies
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

    // Analyze graph structure
    println!("\n📊 Computing Graph Metrics...");
    let metrics: GraphMetrics = graph.compute_metrics()?;
    display_graph_metrics(&metrics);

    // Compute centrality metrics
    println!("\n⭐ Computing Centrality Metrics...");
    println!("   Using PageRank algorithm for directed graphs");
    let centrality: CentralityMetrics = graph.compute_centrality();
    display_centrality_metrics(&centrality);

    // Detect communities
    println!("\n🌐 Detecting Communities (Using SCCs for DiGraph)...");
    let communities = graph.detect_communities()?;
    display_communities(&communities);

    // Check for circular dependencies
    println!("\n🔄 Checking for Circular Dependencies...");
    let has_cycles = graph.has_cycles()?;
    if has_cycles {
        println!("   ⚠️  Warning: Circular dependencies detected!");
        println!("   This may indicate design issues in your model.");
    } else {
        println!("   ✓ No circular dependencies found");
        println!("   Model has a clean dependency structure");
    }

    // Analyze strongly connected components
    println!("\n🔗 Strongly Connected Components Analysis...");
    let sccs = graph.strongly_connected_components()?;
    println!("   Found {} strongly connected components", sccs.len());
    for (i, scc) in sccs.iter().enumerate() {
        if scc.len() > 1 {
            println!(
                "   Component {}: {} nodes (potential cycle)",
                i + 1,
                scc.len()
            );
            for node in scc {
                println!("      - {}", node);
            }
        }
    }

    // Find shortest path between elements
    println!("\n🗺️  Shortest Path Analysis...");
    let path = graph.shortest_path("VehicleAspect", "Status")?;
    if let Some(path_nodes) = path {
        println!("   Path from VehicleAspect to Status:");
        for (i, node) in path_nodes.iter().enumerate() {
            println!("      {}. {}", i + 1, node);
        }
    } else {
        println!("   No path found between specified nodes");
    }

    println!("\n{}", "=".repeat(70));
    println!("\n✅ Graph Analytics Demo Complete!\n");
    println!("This demo showcased:");
    println!("  • Dependency graph construction from SAMM models");
    println!("  • Graph metrics computation (density, diameter)");
    println!("  • Centrality analysis using PageRank");
    println!("  • Community detection using strongly connected components");
    println!("  • Circular dependency detection");
    println!("  • Shortest path finding\n");

    Ok(())
}

/// Create a sample SAMM aspect model with multiple properties and characteristics
fn create_sample_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#VehicleAspect".to_string());

    // Add metadata
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Vehicle Aspect".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "Represents vehicle telemetry data".to_string(),
    );

    // Property 1: Speed
    let speed_char = Characteristic {
        metadata: ElementMetadata::new(
            "urn:samm:org.example:1.0.0#SpeedCharacteristic".to_string(),
        ),
        data_type: Some("xsd:float".to_string()),
        kind: CharacteristicKind::Quantifiable {
            unit: "km/h".to_string(),
        },
        constraints: vec![],
    };
    let speed_property = Property::new("urn:samm:org.example:1.0.0#Speed".to_string())
        .with_characteristic(speed_char);
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

    // Property 3: Status
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
            ],
        },
        constraints: vec![],
    };
    let status_property = Property::new("urn:samm:org.example:1.0.0#Status".to_string())
        .with_characteristic(status_char);
    aspect.add_property(status_property);

    // Property 4: Timestamp
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

/// Display graph metrics in a formatted table
fn display_graph_metrics(metrics: &GraphMetrics) {
    println!("   ┌─────────────────────────────────────┐");
    println!("   │       Graph Metrics                 │");
    println!("   ├─────────────────────────────────────┤");
    println!("   │ Nodes:     {:>24} │", metrics.num_nodes);
    println!("   │ Edges:     {:>24} │", metrics.num_edges);
    println!("   │ Density:   {:>24.4} │", metrics.density);
    println!("   │ Diameter:  {:>24.1} │", metrics.diameter);
    println!("   └─────────────────────────────────────┘");
}

/// Display centrality metrics for top nodes
fn display_centrality_metrics(centrality: &CentralityMetrics) {
    println!("   Top 5 Most Central Elements (by PageRank):");
    for (i, (name, score)) in centrality.top_nodes(5).iter().enumerate() {
        println!("      {}. {:30} Score: {:.6}", i + 1, name, score);
    }

    if let Some((max_node, max_score)) = centrality.max_node() {
        println!("\n   Most Central Element:");
        println!("      {} (Score: {:.6})", max_node, max_score);
    }
}

/// Display detected communities
fn display_communities(communities: &[oxirs_samm::graph_analytics::Community]) {
    println!("   Found {} communities", communities.len());
    for (i, community) in communities.iter().enumerate().take(5) {
        println!(
            "   Community {}: {} members",
            i + 1,
            community.members.len()
        );
        for member in community.members.iter().take(3) {
            println!("      - {}", member);
        }
        if community.members.len() > 3 {
            println!("      ... and {} more", community.members.len() - 3);
        }
    }
}
