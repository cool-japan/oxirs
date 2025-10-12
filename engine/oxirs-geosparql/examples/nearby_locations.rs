//! Nearby locations example for oxirs-geosparql
//!
//! This example demonstrates a real-world scenario:
//! - Building a location database with spatial indexing
//! - Finding nearby points of interest
//! - Filtering results by distance and spatial relations
//! - Calculating routes and areas
//!
//! Run with: cargo run --example nearby_locations

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::geometric_operations::distance;
use oxirs_geosparql::functions::simple_features::sf_within;
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::index::SpatialIndex;

#[derive(Debug)]
#[allow(dead_code)]
struct PointOfInterest {
    id: u64,
    name: String,
    category: String,
    geometry: Geometry,
}

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Nearby Locations Example ===\n");
    println!("Scenario: Finding coffee shops and restaurants near your location\n");

    // 1. Create a spatial index for POIs
    println!("1. Building location database:");
    let index = SpatialIndex::new();
    let mut pois = Vec::new();

    // Add various points of interest around a fictional city center
    let poi_data = vec![
        ("Starbucks - Main St", "coffee", "POINT(0.001 0.001)"),
        ("Blue Bottle Coffee", "coffee", "POINT(0.002 0.003)"),
        ("Local Cafe", "coffee", "POINT(0.005 0.002)"),
        ("Coffee Bean", "coffee", "POINT(-0.003 0.004)"),
        ("Italian Bistro", "restaurant", "POINT(0.002 0.002)"),
        ("Sushi Bar", "restaurant", "POINT(0.004 0.001)"),
        ("Burger Joint", "restaurant", "POINT(-0.002 0.003)"),
        ("Thai Palace", "restaurant", "POINT(0.003 -0.002)"),
        ("Central Library", "library", "POINT(0.0 0.0)"),
        ("City Park", "park", "POINT(0.006 0.005)"),
    ];

    for (name, category, wkt) in poi_data.iter() {
        let geometry = Geometry::from_wkt(wkt)?;
        let id = index.insert(geometry.clone())?;
        pois.push(PointOfInterest {
            id,
            name: name.to_string(),
            category: category.to_string(),
            geometry,
        });
        println!("  Added: {} ({}) at {}", name, category, wkt);
    }

    println!("\n  Total locations indexed: {}", index.len());

    // 2. User's current location
    let user_location = Geometry::from_wkt("POINT(0.0 0.0)")?;
    println!("\n2. Your current location: {}", user_location.to_wkt());

    // 3. Find nearest location
    println!("\n3. Finding nearest location:");
    if let Some((nearest_geom, dist)) = index.nearest(0.0, 0.0) {
        // Find the POI that matches this geometry
        if let Some(poi) = pois.iter().find(|p| p.geometry == nearest_geom) {
            println!("  Nearest: {} ({})", poi.name, poi.category);
            println!("  Distance: {:.4} degrees", dist);
            println!("  Approximately {:.0} meters", dist * 111000.0);
        }
    }

    // 4. Find all locations within walking distance (500m = ~0.0045 degrees)
    println!("\n4. Locations within walking distance (500m):");
    let walking_distance = 0.0045;
    let nearby = index.query_within_distance(0.0, 0.0, walking_distance);

    println!("  Found {} locations within 500m:", nearby.len());
    for (geom, dist) in &nearby {
        if let Some(poi) = pois.iter().find(|p| p.geometry == *geom) {
            let meters = dist * 111000.0;
            println!("    - {} ({}): {:.0}m away", poi.name, poi.category, meters);
        }
    }

    // 5. Filter by category: Find coffee shops within 1km
    println!("\n5. Coffee shops within 1km:");
    let coffee_radius = 0.009; // ~1km
    let nearby_all = index.query_within_distance(0.0, 0.0, coffee_radius);

    let mut coffee_shops = Vec::new();
    for (geom, dist) in &nearby_all {
        if let Some(poi) = pois.iter().find(|p| p.geometry == *geom) {
            if poi.category == "coffee" {
                coffee_shops.push((poi, dist));
            }
        }
    }

    println!("  Found {} coffee shops:", coffee_shops.len());
    for (poi, dist) in &coffee_shops {
        let meters = *dist * 111000.0;
        println!("    - {}: {:.0}m away", poi.name, meters);
    }

    // 6. Find locations in a specific area (bounding box)
    println!("\n6. Locations in the north-east quadrant:");
    let ne_results = index.query_bbox(0.0, 0.0, 0.01, 0.01);
    println!("  Found {} locations in NE quadrant:", ne_results.len());
    for geom in &ne_results {
        if let Some(poi) = pois.iter().find(|p| p.geometry == *geom) {
            println!("    - {} ({})", poi.name, poi.category);
        }
    }

    // 7. Calculate distances between specific POIs
    println!("\n7. Distances between specific locations:");
    let library = pois.iter().find(|p| p.name == "Central Library").unwrap();
    let starbucks = pois
        .iter()
        .find(|p| p.name == "Starbucks - Main St")
        .unwrap();

    let lib_to_sb_dist = distance(&library.geometry, &starbucks.geometry)?;
    println!(
        "  Central Library to Starbucks: {:.0}m",
        lib_to_sb_dist * 111000.0
    );

    // 8. Define a neighborhood boundary and find contained locations
    println!("\n8. Locations within neighborhood boundary:");
    let neighborhood = Geometry::from_wkt(
        "POLYGON((-0.004 -0.004, 0.004 -0.004, 0.004 0.004, -0.004 0.004, -0.004 -0.004))",
    )?;

    let mut contained_count = 0;
    for poi in &pois {
        if sf_within(&poi.geometry, &neighborhood)? {
            println!("    - {} ({})", poi.name, poi.category);
            contained_count += 1;
        }
    }
    println!("  Total locations in neighborhood: {}", contained_count);

    // 9. Find top 3 nearest restaurants
    println!("\n9. Top 3 nearest restaurants:");
    let mut restaurants_with_dist: Vec<_> = pois
        .iter()
        .filter(|p| p.category == "restaurant")
        .map(|p| {
            let dist = distance(&user_location, &p.geometry).unwrap_or(f64::MAX);
            (p, dist)
        })
        .collect();

    restaurants_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (i, (poi, dist)) in restaurants_with_dist.iter().take(3).enumerate() {
        let meters = dist * 111000.0;
        println!("  {}. {}: {:.0}m away", i + 1, poi.name, meters);
    }

    // 10. Statistical summary
    println!("\n10. Statistical summary:");
    let avg_distance: f64 = pois
        .iter()
        .map(|p| distance(&user_location, &p.geometry).unwrap_or(0.0))
        .sum::<f64>()
        / pois.len() as f64;

    println!(
        "  Average distance to all POIs: {:.0}m",
        avg_distance * 111000.0
    );

    let category_counts: std::collections::HashMap<_, _> =
        pois.iter()
            .fold(std::collections::HashMap::new(), |mut acc, p| {
                *acc.entry(&p.category).or_insert(0) += 1;
                acc
            });

    println!("\n  Locations by category:");
    for (category, count) in category_counts {
        println!("    - {}: {}", category, count);
    }

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
