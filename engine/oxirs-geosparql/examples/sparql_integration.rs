//! SPARQL Integration Example
//!
//! This example demonstrates how to integrate GeoSPARQL functions with oxirs-arq
//! for use in SPARQL queries.

use oxirs_geosparql::sparql_integration::{get_all_geosparql_functions, FunctionCategory};

fn main() {
    println!("=== GeoSPARQL Function Registration for oxirs-arq ===\n");

    // Get all GeoSPARQL functions
    let all_functions = get_all_geosparql_functions();
    println!(
        "Total GeoSPARQL functions available: {}\n",
        all_functions.len()
    );

    // Group by category
    let filter_functions: Vec<_> = all_functions
        .iter()
        .filter(|f| f.category == FunctionCategory::Filter)
        .collect();
    let property_functions: Vec<_> = all_functions
        .iter()
        .filter(|f| f.category == FunctionCategory::Property)
        .collect();
    let distance_functions: Vec<_> = all_functions
        .iter()
        .filter(|f| f.category == FunctionCategory::Distance)
        .collect();

    println!("Filter Functions ({})", filter_functions.len());
    println!("{}", "=".repeat(60));
    for func in &filter_functions {
        println!("  {} - {}", func.name, func.description);
        println!("    URI: {}", func.uri);
        println!("    Arity: {}", func.arity);
        println!();
    }

    println!("\nProperty Functions ({})", property_functions.len());
    println!("{}", "=".repeat(60));
    for func in &property_functions {
        println!("  {} - {}", func.name, func.description);
        println!("    URI: {}", func.uri);
        println!("    Arity: {}", func.arity);
        println!();
    }

    println!("\nDistance Functions ({})", distance_functions.len());
    println!("{}", "=".repeat(60));
    for func in &distance_functions {
        println!("  {} - {}", func.name, func.description);
        println!("    URI: {}", func.uri);
        println!("    Arity: {}", func.arity);
        println!();
    }

    // Example SPARQL queries using these functions
    println!("\n=== Example SPARQL Queries ===\n");

    println!("1. Find all features that contain a point:");
    println!(
        r#"
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>

SELECT ?feature WHERE {{
  ?feature geo:hasGeometry ?geom1 .
  ?point geo:hasGeometry ?geom2 .
  ?geom1 geo:asWKT ?wkt1 .
  ?geom2 geo:asWKT ?wkt2 .
  FILTER(geof:sfContains(?wkt1, ?wkt2))
}}
"#
    );

    println!("\n2. Find all features within 100 meters:");
    println!(
        r#"
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>

SELECT ?feature ?dist WHERE {{
  ?feature geo:hasGeometry ?geom1 .
  ?poi geo:hasGeometry ?geom2 .
  ?geom1 geo:asWKT ?wkt1 .
  ?geom2 geo:asWKT ?wkt2 .
  BIND(geof:distance(?wkt1, ?wkt2) AS ?dist)
  FILTER(?dist < 100)
}}
"#
    );

    println!("\n3. Find all features that intersect:");
    println!(
        r#"
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>

SELECT ?feature1 ?feature2 WHERE {{
  ?feature1 geo:hasGeometry/geo:asWKT ?wkt1 .
  ?feature2 geo:hasGeometry/geo:asWKT ?wkt2 .
  FILTER(?feature1 != ?feature2)
  FILTER(geof:sfIntersects(?wkt1, ?wkt2))
}}
"#
    );

    println!("\n4. Check if geometry is 3D:");
    println!(
        r#"
PREFIX geo: <http://www.opengis.net/ont/geosparql#>

SELECT ?feature WHERE {{
  ?feature geo:hasGeometry ?geom .
  ?geom geo:asWKT ?wkt .
  FILTER(geo:is3D(?wkt))
}}
"#
    );

    println!("\n=== Integration Code Template ===\n");
    println!("To integrate with oxirs-arq, use this template:\n");
    println!(
        r#"
use oxirs_geosparql::sparql_integration::{{get_all_geosparql_functions, FunctionCategory}};
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::{{simple_features, egenhofer, rcc8}};

fn register_geosparql_functions(registry: &mut FunctionRegistry) {{
    let functions = get_all_geosparql_functions();

    for func in functions {{
        match func.category {{
            FunctionCategory::Filter => {{
                // Register boolean filter function
                registry.register_filter(&func.uri, func.arity, move |args| {{
                    // Parse geometries from WKT arguments
                    let geom1 = Geometry::from_wkt(args[0].as_str())?;
                    let geom2 = Geometry::from_wkt(args[1].as_str())?;

                    // Call appropriate function based on URI
                    match func.name.as_str() {{
                        "sfContains" => simple_features::sf_contains(&geom1, &geom2),
                        "sfIntersects" => simple_features::sf_intersects(&geom1, &geom2),
                        // ... other functions
                        _ => Ok(false),
                    }}
                }});
            }}
            FunctionCategory::Property => {{
                // Register property function
                registry.register_property(&func.uri, func.arity, move |args| {{
                    let geom = Geometry::from_wkt(args[0].as_str())?;

                    match func.name.as_str() {{
                        "dimension" => Ok(geom.dimension()),
                        "isEmpty" => Ok(geom.is_empty()),
                        "is3D" => Ok(geom.is_3d()),
                        // ... other functions
                        _ => Ok(false),
                    }}
                }});
            }}
            FunctionCategory::Distance => {{
                // Register distance function
                registry.register_distance(&func.uri, func.arity, move |args| {{
                    let geom1 = Geometry::from_wkt(args[0].as_str())?;
                    let geom2 = Geometry::from_wkt(args[1].as_str())?;

                    // Calculate distance
                    crate::functions::geometric_operations::distance(&geom1, &geom2)
                }});
            }}
        }}
    }}
}}
"#
    );

    println!("\n=== Summary ===\n");
    println!(
        "- {} filter functions (spatial predicates)",
        filter_functions.len()
    );
    println!(
        "- {} property functions (geometry properties)",
        property_functions.len()
    );
    println!("- {} distance functions", distance_functions.len());
    println!(
        "\nTotal: {} GeoSPARQL functions ready for SPARQL integration",
        all_functions.len()
    );
}
