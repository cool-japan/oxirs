//! Topological queries example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Simple Features topological relations
//! - Geometric operations (distance, envelope, convex hull)
//! - Real-world spatial analysis scenarios
//!
//! Run with: cargo run --example topological_queries

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::geometric_operations::{convex_hull, distance, envelope};
use oxirs_geosparql::functions::simple_features::{
    sf_contains, sf_crosses, sf_disjoint, sf_equals, sf_intersects, sf_overlaps, sf_touches,
    sf_within,
};
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Topological Queries Example ===\n");

    // 1. Simple Features: Equals
    println!("1. Simple Features: Equals");
    let polygon1 = Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))")?;
    let polygon2 = Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))")?;
    let polygon3 = Geometry::from_wkt("POLYGON((1 1, 6 1, 6 6, 1 6, 1 1))")?;

    println!(
        "  Polygon1 == Polygon2: {}",
        sf_equals(&polygon1, &polygon2)?
    );
    println!(
        "  Polygon1 == Polygon3: {}",
        sf_equals(&polygon1, &polygon3)?
    );

    // 2. Simple Features: Disjoint
    println!("\n2. Simple Features: Disjoint");
    let point1 = Geometry::from_wkt("POINT(0 0)")?;
    let point2 = Geometry::from_wkt("POINT(10 10)")?;
    let nearby_point = Geometry::from_wkt("POINT(1 1)")?;

    println!(
        "  Point(0,0) disjoint from Point(10,10): {}",
        sf_disjoint(&point1, &point2)?
    );
    println!(
        "  Point(0,0) disjoint from Point(1,1): {}",
        sf_disjoint(&point1, &nearby_point)?
    );

    // 3. Simple Features: Intersects
    println!("\n3. Simple Features: Intersects");
    let line1 = Geometry::from_wkt("LINESTRING(0 0, 10 10)")?;
    let line2 = Geometry::from_wkt("LINESTRING(0 10, 10 0)")?;
    let line3 = Geometry::from_wkt("LINESTRING(20 20, 30 30)")?;

    println!(
        "  Line1 intersects Line2 (crossing): {}",
        sf_intersects(&line1, &line2)?
    );
    println!(
        "  Line1 intersects Line3 (separate): {}",
        sf_intersects(&line1, &line3)?
    );

    // 4. Simple Features: Touches
    println!("\n4. Simple Features: Touches");
    let square1 = Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))")?;
    let square2 = Geometry::from_wkt("POLYGON((5 0, 10 0, 10 5, 5 5, 5 0))")?;
    let square3 = Geometry::from_wkt("POLYGON((6 0, 11 0, 11 5, 6 5, 6 0))")?;

    println!(
        "  Square1 touches Square2 (adjacent): {}",
        sf_touches(&square1, &square2)?
    );
    println!(
        "  Square1 touches Square3 (separate): {}",
        sf_touches(&square1, &square3)?
    );

    // 5. Simple Features: Crosses
    println!("\n5. Simple Features: Crosses");
    let road = Geometry::from_wkt("LINESTRING(0 5, 10 5)")?;
    let region = Geometry::from_wkt("POLYGON((2 2, 8 2, 8 8, 2 8, 2 2))")?;

    println!("  Road crosses Region: {}", sf_crosses(&road, &region)?);

    // 6. Simple Features: Within and Contains
    println!("\n6. Simple Features: Within and Contains");
    let city_boundary = Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))")?;
    let building = Geometry::from_wkt("POLYGON((10 10, 20 10, 20 20, 10 20, 10 10))")?;
    let point_in_city = Geometry::from_wkt("POINT(50 50)")?;
    let point_outside = Geometry::from_wkt("POINT(150 150)")?;

    println!(
        "  Building within City: {}",
        sf_within(&building, &city_boundary)?
    );
    println!(
        "  City contains Building: {}",
        sf_contains(&city_boundary, &building)?
    );
    println!(
        "  Point(50,50) within City: {}",
        sf_within(&point_in_city, &city_boundary)?
    );
    println!(
        "  Point(150,150) within City: {}",
        sf_within(&point_outside, &city_boundary)?
    );

    // 7. Simple Features: Overlaps
    println!("\n7. Simple Features: Overlaps");
    let park1 = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    let park2 = Geometry::from_wkt("POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))")?;

    println!("  Park1 overlaps Park2: {}", sf_overlaps(&park1, &park2)?);

    // 8. Geometric Operations: Distance
    println!("\n8. Geometric Operations: Distance");
    let tokyo = Geometry::from_wkt("POINT(139.6917 35.6895)")?;
    let osaka = Geometry::from_wkt("POINT(135.5022 34.6937)")?;

    let dist = distance(&tokyo, &osaka)?;
    println!("  Distance Tokyo to Osaka: {:.2} degrees", dist);
    println!("  (approximately {:.0} km)", dist * 111.0); // Rough conversion

    let point_a = Geometry::from_wkt("POINT(0 0)")?;
    let point_b = Geometry::from_wkt("POINT(3 4)")?;
    let pythagorean_dist = distance(&point_a, &point_b)?;
    println!("\n  Distance (0,0) to (3,4): {:.2}", pythagorean_dist);
    println!("  (Pythagorean theorem: 5.0)");

    // 9. Geometric Operations: Envelope
    println!("\n9. Geometric Operations: Envelope (Bounding Box)");
    let complex_polygon =
        Geometry::from_wkt("POLYGON((0 0, 5 2, 10 0, 12 5, 10 10, 5 8, 0 10, -2 5, 0 0))")?;
    let bbox = envelope(&complex_polygon)?;
    println!("  Complex polygon: {}", complex_polygon.to_wkt());
    println!("  Envelope (bbox): {}", bbox.to_wkt());

    let multipoint = Geometry::from_wkt("MULTIPOINT((0 0), (5 5), (10 2), (3 8))")?;
    let mp_bbox = envelope(&multipoint)?;
    println!("\n  MultiPoint: {}", multipoint.to_wkt());
    println!("  Envelope (bbox): {}", mp_bbox.to_wkt());

    // 10. Geometric Operations: Convex Hull
    println!("\n10. Geometric Operations: Convex Hull");
    let scattered_points =
        Geometry::from_wkt("MULTIPOINT((0 0), (1 1), (2 0), (3 1), (1 2), (2 3), (0 2))")?;
    let hull = convex_hull(&scattered_points)?;
    println!("  Scattered points: {}", scattered_points.to_wkt());
    println!("  Convex hull: {}", hull.to_wkt());
    println!("  Hull type: {}", hull.geometry_type());

    // 11. Real-world scenario: Urban planning
    println!("\n11. Real-world scenario: Urban Planning");
    let proposed_building = Geometry::from_wkt("POLYGON((45 45, 55 45, 55 55, 45 55, 45 45))")?;
    let park_zone = Geometry::from_wkt("POLYGON((40 40, 60 40, 60 60, 40 60, 40 60))")?;
    let residential_zone = Geometry::from_wkt("POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))")?;

    println!(
        "  Proposed building intersects park zone: {}",
        sf_intersects(&proposed_building, &park_zone)?
    );
    println!(
        "  Proposed building within residential zone: {}",
        sf_within(&proposed_building, &residential_zone)?
    );
    println!(
        "  Distance to park boundary: {:.2}",
        distance(&proposed_building, &park_zone)?
    );

    // 12. Real-world scenario: Transportation network
    println!("\n12. Real-world scenario: Transportation Network");
    let highway = Geometry::from_wkt("LINESTRING(0 50, 100 50)")?;
    let railway = Geometry::from_wkt("LINESTRING(50 0, 50 100)")?;
    let city_center = Geometry::from_wkt("POLYGON((40 40, 60 40, 60 60, 40 60, 40 40))")?;

    println!(
        "  Highway crosses railway: {}",
        sf_crosses(&highway, &railway)?
    );
    println!(
        "  Highway intersects city center: {}",
        sf_intersects(&highway, &city_center)?
    );
    println!(
        "  Railway intersects city center: {}",
        sf_intersects(&railway, &city_center)?
    );

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
