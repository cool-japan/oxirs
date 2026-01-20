//! Spatial index example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Creating a spatial index
//! - Inserting geometries
//! - Performing bounding box queries
//! - Finding nearest neighbors
//! - Querying within distance
//!
//! Run with: cargo run --example spatial_index_demo

use oxirs_geosparql::error::Result;
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::index::SpatialIndex;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Spatial Index Example ===\n");

    // 1. Create a spatial index
    println!("1. Creating spatial index:");
    let index = SpatialIndex::new();
    println!("  Created empty index");

    // 2. Insert major world cities
    println!("\n2. Inserting world cities:");
    let cities = vec![
        ("Tokyo", "POINT(139.6917 35.6895)"),
        ("New York", "POINT(-74.0060 40.7128)"),
        ("London", "POINT(-0.1276 51.5074)"),
        ("Paris", "POINT(2.3522 48.8566)"),
        ("Sydney", "POINT(151.2093 -33.8688)"),
        ("Beijing", "POINT(116.4074 39.9042)"),
        ("Mumbai", "POINT(72.8777 19.0760)"),
        ("São Paulo", "POINT(-46.6333 -23.5505)"),
        ("Mexico City", "POINT(-99.1332 19.4326)"),
        ("Cairo", "POINT(31.2357 30.0444)"),
    ];

    for (name, wkt) in &cities {
        let geom = Geometry::from_wkt(wkt)?;
        let id = index.insert(geom)?;
        println!("  Inserted {} (ID: {})", name, id);
    }

    println!("\n  Total cities in index: {}", index.len());

    // 3. Bounding box query - Find cities in Europe
    println!("\n3. Bounding box query - Cities in Europe:");
    println!("  Query bbox: [-10, 35] to [30, 60]");
    let europe_cities = index.query_bbox(-10.0, 35.0, 30.0, 60.0);
    println!("  Found {} cities in Europe:", europe_cities.len());
    for geom in &europe_cities {
        println!("    {}", geom.to_wkt());
    }

    // 4. Bounding box query - Find cities in Asia
    println!("\n4. Bounding box query - Cities in Asia:");
    println!("  Query bbox: [60, 0] to [160, 60]");
    let asia_cities = index.query_bbox(60.0, 0.0, 160.0, 60.0);
    println!("  Found {} cities in Asia:", asia_cities.len());
    for geom in &asia_cities {
        println!("    {}", geom.to_wkt());
    }

    // 5. Nearest neighbor query
    println!("\n5. Nearest neighbor query:");
    let query_point = (0.0, 50.0); // Somewhere in the English Channel
    println!("  Query point: ({}, {})", query_point.0, query_point.1);

    if let Some((nearest, distance)) = index.nearest(query_point.0, query_point.1) {
        println!("  Nearest city: {}", nearest.to_wkt());
        println!("  Distance: {:.2} degrees", distance);
    }

    // 6. Within distance query
    println!("\n6. Within distance query:");
    let paris_coords = (2.3522, 48.8566);
    let radius = 10.0;
    println!(
        "  Query center: ({}, {}) [Paris]",
        paris_coords.0, paris_coords.1
    );
    println!("  Radius: {} degrees (~1000 km)", radius);

    let nearby_cities = index.query_within_distance(paris_coords.0, paris_coords.1, radius);
    println!("  Found {} cities within radius:", nearby_cities.len());
    for (geom, dist) in &nearby_cities {
        println!("    {} (distance: {:.2} degrees)", geom.to_wkt(), dist);
    }

    // 7. Insert and query polygons
    println!("\n7. Working with polygons:");
    let polygon_index = SpatialIndex::new();

    let regions = vec![
        ("Region A", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"),
        ("Region B", "POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))"),
        ("Region C", "POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))"),
    ];

    for (name, wkt) in &regions {
        let geom = Geometry::from_wkt(wkt)?;
        polygon_index.insert(geom)?;
        println!("  Inserted {}", name);
    }

    // Query for regions overlapping with a search area
    let search_area = polygon_index.query_bbox(0.0, 0.0, 12.0, 12.0);
    println!(
        "\n  Regions overlapping with bbox [0,0] to [12,12]: {}",
        search_area.len()
    );

    // 8. Index statistics
    println!("\n8. Index statistics:");
    println!("  Total cities indexed: {}", index.len());
    println!("  Index is empty: {}", index.is_empty());
    println!("  Total regions indexed: {}", polygon_index.len());

    // 9. Remove a geometry
    println!("\n9. Removing a geometry:");
    let id_to_remove = 0; // Remove first city (Tokyo)
    let removed = index.remove(id_to_remove)?;
    if removed {
        println!("  Successfully removed geometry with ID {}", id_to_remove);
        println!("  Remaining cities: {}", index.len());
    }

    // 10. Clear the index
    println!("\n10. Clearing the index:");
    index.clear();
    println!("  Index cleared");
    println!("  Index is empty: {}", index.is_empty());
    println!("  Index length: {}", index.len());

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
