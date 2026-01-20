//! Comprehensive Model Analytics Demonstration
//!
//! This example demonstrates the full capabilities of the Model Analytics module:
//! - Quality scoring and assessment
//! - Complexity analysis across multiple dimensions
//! - Best practice compliance checking
//! - Statistical distribution analysis
//! - Dependency and coupling metrics
//! - Anomaly detection
//! - Actionable recommendations
//! - HTML report generation
//!
//! Run with: cargo run --example model_analytics_demo

use oxirs_samm::analytics::{ModelAnalytics, Severity};
use oxirs_samm::metamodel::{
    Aspect, Characteristic, CharacteristicKind, Entity, ModelElement, Operation, Property,
};
use std::fs;

fn main() {
    println!("═══════════════════════════════════════════════════");
    println!("  OxiRS SAMM - Model Analytics Demonstration");
    println!("═══════════════════════════════════════════════════\n");

    // Example 1: Analyze a well-designed model
    println!("Example 1: Analyzing a Well-Designed Model");
    println!("───────────────────────────────────────────────────");
    let good_aspect = create_good_aspect();
    demonstrate_analytics(&good_aspect, "Good Model");

    println!("\n");

    // Example 2: Analyze a poorly-designed model
    println!("Example 2: Analyzing a Poorly-Designed Model");
    println!("───────────────────────────────────────────────────");
    let poor_aspect = create_poor_aspect();
    demonstrate_analytics(&poor_aspect, "Poor Model");

    println!("\n");

    // Example 3: Analyze a complex model
    println!("Example 3: Analyzing a Complex Model");
    println!("───────────────────────────────────────────────────");
    let complex_aspect = create_complex_aspect();
    demonstrate_analytics(&complex_aspect, "Complex Model");

    println!("\n═══════════════════════════════════════════════════");
    println!("  Analytics demonstration complete!");
    println!("═══════════════════════════════════════════════════");
}

/// Demonstrate comprehensive analytics on an aspect
fn demonstrate_analytics(aspect: &Aspect, model_name: &str) {
    let analytics = ModelAnalytics::analyze(aspect);

    println!("\n📊 Overall Quality Score");
    println!("   Score: {:.1}/100", analytics.quality_score);
    println!(
        "   Rating: {}",
        match analytics.quality_score {
            s if s >= 90.0 => "Excellent ⭐⭐⭐⭐⭐",
            s if s >= 75.0 => "Good ⭐⭐⭐⭐",
            s if s >= 60.0 => "Fair ⭐⭐⭐",
            s if s >= 40.0 => "Poor ⭐⭐",
            _ => "Very Poor ⭐",
        }
    );

    println!("\n🔍 Complexity Assessment");
    println!(
        "   Overall Level: {:?}",
        analytics.complexity_assessment.overall_level
    );
    println!(
        "   Structural: {:.1}",
        analytics.complexity_assessment.structural
    );
    println!(
        "   Cognitive: {:.1}",
        analytics.complexity_assessment.cognitive
    );
    println!(
        "   Cyclomatic: {:.1}",
        analytics.complexity_assessment.cyclomatic
    );
    println!(
        "   Coupling: {:.1}",
        analytics.complexity_assessment.coupling
    );

    println!("\n✅ Best Practice Compliance");
    println!(
        "   Compliance: {:.1}% ({}/{})",
        analytics.best_practices.compliance_percentage,
        analytics.best_practices.passed_checks,
        analytics.best_practices.total_checks
    );
    for check in &analytics.best_practices.checks {
        let icon = if check.passed { "✓" } else { "✗" };
        println!("   {} {}", icon, check.name);
    }

    println!("\n📈 Statistical Distributions");
    println!(
        "   Property Count: {}",
        analytics.distributions.property_distribution.mean as usize
    );
    println!(
        "   Optionality Ratio: {:.2}",
        analytics.distributions.optionality_ratio
    );
    println!(
        "   Collection Usage: {:.1}%",
        analytics.distributions.collection_percentage
    );
    if !analytics.distributions.type_distribution.is_empty() {
        println!("   Top Data Types:");
        for (dtype, count) in analytics.distributions.type_distribution.iter().take(5) {
            println!("     - {}: {} properties", dtype, count);
        }
    }

    println!("\n🔗 Dependency Metrics");
    println!(
        "   Total Dependencies: {}",
        analytics.dependency_metrics.total_dependencies
    );
    println!(
        "   Avg per Property: {:.2}",
        analytics.dependency_metrics.avg_dependencies_per_property
    );
    println!(
        "   Coupling Factor: {:.2}",
        analytics.dependency_metrics.coupling_factor
    );
    println!(
        "   Cohesion Score: {:.2}",
        analytics.dependency_metrics.cohesion_score
    );
    if analytics.dependency_metrics.circular_dependencies > 0 {
        println!(
            "   ⚠ Circular Dependencies: {}",
            analytics.dependency_metrics.circular_dependencies
        );
    }

    if !analytics.anomalies.is_empty() {
        println!("\n⚠ Detected Anomalies");
        for anomaly in &analytics.anomalies {
            println!("   [{}] {:?}", anomaly.severity, anomaly.anomaly_type);
            println!("     Location: {}", anomaly.location);
            println!("     {}", anomaly.description);
        }
    }

    if !analytics.recommendations.is_empty() {
        println!("\n💡 Recommendations");
        let critical = analytics
            .recommendations
            .iter()
            .filter(|r| r.severity == Severity::Critical || r.severity == Severity::Error)
            .count();
        let warnings = analytics
            .recommendations
            .iter()
            .filter(|r| r.severity == Severity::Warning)
            .count();

        println!(
            "   Total: {} ({} critical/error, {} warning)",
            analytics.recommendations.len(),
            critical,
            warnings
        );
        for rec in analytics.recommendations.iter().take(5) {
            println!("\n   [{}] {}", rec.severity, rec.message);
            println!("     Target: {}", rec.target);
            println!("     Action: {}", rec.suggested_action);
        }
        if analytics.recommendations.len() > 5 {
            println!(
                "\n   ... and {} more recommendations",
                analytics.recommendations.len() - 5
            );
        }
    }

    println!("\n📊 Benchmark Comparison");
    println!("   Overall: {:?}", analytics.benchmark.comparison);
    println!(
        "   Property Count: {:.0}th percentile",
        analytics.benchmark.property_count_percentile
    );
    println!(
        "   Complexity: {:.0}th percentile",
        analytics.benchmark.complexity_percentile
    );
    println!(
        "   Documentation: {:.0}th percentile",
        analytics.benchmark.documentation_percentile
    );

    // Generate HTML report
    let html_report = analytics.generate_html_report();
    let filename = format!(
        "/tmp/analytics_report_{}.html",
        model_name.replace(' ', "_")
    );
    if let Err(e) = fs::write(&filename, html_report) {
        eprintln!("Failed to write HTML report: {}", e);
    } else {
        println!("\n📄 HTML Report Generated");
        println!("   File: {}", filename);
        println!("   Open in browser to view detailed analysis");
    }
}

/// Create a well-designed aspect for demonstration
fn create_good_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#Vehicle".to_string());

    // Add comprehensive metadata
    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Vehicle".to_string());
    aspect
        .metadata
        .add_preferred_name("de".to_string(), "Fahrzeug".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "Represents a vehicle with its basic properties".to_string(),
    );
    aspect.metadata.add_description(
        "de".to_string(),
        "Repräsentiert ein Fahrzeug mit seinen grundlegenden Eigenschaften".to_string(),
    );

    // Add well-structured properties
    let mut vin = Property::new("urn:samm:org.example:1.0.0#vin".to_string());
    vin.metadata
        .add_preferred_name("en".to_string(), "VIN".to_string());
    vin.metadata.add_description(
        "en".to_string(),
        "Vehicle Identification Number".to_string(),
    );
    let mut vin_char = Characteristic::new(
        "urn:samm:org.example:1.0.0#VinTrait".to_string(),
        CharacteristicKind::Trait,
    );
    vin_char.data_type = Some("xsd:string".to_string());
    vin.characteristic = Some(vin_char);
    vin.optional = false;
    aspect.add_property(vin);

    let mut manufacturer = Property::new("urn:samm:org.example:1.0.0#manufacturer".to_string());
    manufacturer
        .metadata
        .add_preferred_name("en".to_string(), "Manufacturer".to_string());
    let mut mfr_char = Characteristic::new(
        "urn:samm:org.example:1.0.0#ManufacturerTrait".to_string(),
        CharacteristicKind::Trait,
    );
    mfr_char.data_type = Some("xsd:string".to_string());
    manufacturer.characteristic = Some(mfr_char);
    manufacturer.optional = false;
    aspect.add_property(manufacturer);

    let mut model_year = Property::new("urn:samm:org.example:1.0.0#modelYear".to_string());
    model_year
        .metadata
        .add_preferred_name("en".to_string(), "Model Year".to_string());
    let mut year_char = Characteristic::new(
        "urn:samm:org.example:1.0.0#YearTrait".to_string(),
        CharacteristicKind::Trait,
    );
    year_char.data_type = Some("xsd:gYear".to_string());
    model_year.characteristic = Some(year_char);
    model_year.optional = false;
    aspect.add_property(model_year);

    let mut color = Property::new("urn:samm:org.example:1.0.0#color".to_string());
    color
        .metadata
        .add_preferred_name("en".to_string(), "Color".to_string());
    let mut color_char = Characteristic::new(
        "urn:samm:org.example:1.0.0#ColorTrait".to_string(),
        CharacteristicKind::Trait,
    );
    color_char.data_type = Some("xsd:string".to_string());
    color.characteristic = Some(color_char);
    color.optional = true;
    aspect.add_property(color);

    let mut mileage = Property::new("urn:samm:org.example:1.0.0#mileage".to_string());
    mileage
        .metadata
        .add_preferred_name("en".to_string(), "Mileage".to_string());
    let mut mileage_char = Characteristic::new(
        "urn:samm:org.example:1.0.0#MileageMeasurement".to_string(),
        CharacteristicKind::Measurement {
            unit: "unit:kilometre".to_string(),
        },
    );
    mileage_char.data_type = Some("xsd:decimal".to_string());
    mileage.characteristic = Some(mileage_char);
    mileage.optional = true;
    aspect.add_property(mileage);

    aspect
}

/// Create a poorly-designed aspect for demonstration
fn create_poor_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#BadExample".to_string());
    // No preferred name or description!

    // Poor naming (PascalCase for properties)
    let prop1 = Property::new("urn:samm:org.example:1.0.0#BadProperty1".to_string());
    // No characteristic!
    aspect.add_property(prop1);

    let mut prop2 = Property::new("urn:samm:org.example:1.0.0#another_bad_name".to_string());
    let char2 = Characteristic::new(
        "urn:samm:org.example:1.0.0#SomeChar".to_string(),
        CharacteristicKind::Trait,
    );
    // No data type!
    prop2.characteristic = Some(char2);
    aspect.add_property(prop2);

    // Mix of naming conventions
    let mut prop3 = Property::new("urn:samm:org.example:1.0.0#goodName".to_string());
    let mut char3 = Characteristic::new(
        "urn:samm:org.example:1.0.0#GoodChar".to_string(),
        CharacteristicKind::Trait,
    );
    char3.data_type = Some("xsd:string".to_string());
    prop3.characteristic = Some(char3);
    aspect.add_property(prop3);

    aspect
}

/// Create a complex aspect for demonstration
fn create_complex_aspect() -> Aspect {
    let mut aspect = Aspect::new("urn:samm:org.example:1.0.0#ComplexSystem".to_string());

    aspect
        .metadata
        .add_preferred_name("en".to_string(), "Complex System".to_string());
    aspect.metadata.add_description(
        "en".to_string(),
        "A highly complex system with many properties".to_string(),
    );

    // Add many properties to increase complexity
    for i in 0..25 {
        let mut prop = Property::new(format!("urn:samm:org.example:1.0.0#property{}", i));
        prop.metadata
            .add_preferred_name("en".to_string(), format!("Property {}", i));

        let data_type = match i % 5 {
            0 => "xsd:string",
            1 => "xsd:integer",
            2 => "xsd:boolean",
            3 => "xsd:decimal",
            _ => "xsd:dateTime",
        };

        let mut char = Characteristic::new(
            format!("urn:samm:org.example:1.0.0#char{}", i),
            if i % 3 == 0 {
                CharacteristicKind::Collection {
                    element_characteristic: None,
                }
            } else {
                CharacteristicKind::Trait
            },
        );
        char.data_type = Some(data_type.to_string());
        prop.characteristic = Some(char);
        prop.optional = i % 2 == 0;
        prop.is_collection = i % 3 == 0;

        aspect.add_property(prop);
    }

    // Add operations
    for i in 0..5 {
        let op = Operation::new(format!("urn:samm:org.example:1.0.0#operation{}", i));
        aspect.add_operation(op);
    }

    aspect
}
