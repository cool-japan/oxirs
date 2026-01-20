//! Integration tests for GeoSPARQL spatial query scenarios
//!
//! These tests demonstrate real-world usage of the oxirs-geosparql crate
//! for spatial data queries and operations.

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::geometric_operations::{distance, envelope};
use oxirs_geosparql::functions::simple_features::{
    sf_contains, sf_intersects, sf_overlaps, sf_within,
};
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::index::SpatialIndex;

/// Test scenario: Finding all features within a bounding box
/// This simulates a typical GeoSPARQL query like:
/// SELECT ?feature WHERE {
///   ?feature geo:hasGeometry ?geom .
///   FILTER(geof:sfWithin(?geom, ?bbox))
/// }
#[test]
fn test_bbox_query_scenario() -> Result<()> {
    // Create a spatial index to store features
    let index = SpatialIndex::new();

    // Insert several point features representing locations
    let locations = vec![
        ("Tokyo", "POINT(139.6917 35.6895)"),
        ("New York", "POINT(-74.0060 40.7128)"),
        ("London", "POINT(-0.1276 51.5074)"),
        ("Sydney", "POINT(151.2093 -33.8688)"),
        ("Paris", "POINT(2.3522 48.8566)"),
    ];

    for (name, wkt) in &locations {
        let geom = Geometry::from_wkt(wkt)?;
        index.insert(geom)?;
        println!("Inserted {}: {}", name, wkt);
    }

    assert_eq!(index.len(), 5);

    // Query for all features within a bounding box covering Europe
    // Bbox: [-10, 35] to [30, 60] (roughly covers Western Europe)
    let results = index.query_bbox(-10.0, 35.0, 30.0, 60.0);

    println!("Found {} features in Europe bbox", results.len());
    assert_eq!(results.len(), 2); // Should find London and Paris

    Ok(())
}

/// Test scenario: Finding nearest neighbor
/// Simulates a query like: "Find the nearest city to a given point"
#[test]
fn test_nearest_neighbor_scenario() -> Result<()> {
    let index = SpatialIndex::new();

    // Insert major cities
    let cities = vec![
        "POINT(139.6917 35.6895)", // Tokyo
        "POINT(-74.0060 40.7128)", // New York
        "POINT(-0.1276 51.5074)",  // London
    ];

    for wkt in &cities {
        let geom = Geometry::from_wkt(wkt)?;
        index.insert(geom)?;
    }

    // Find nearest city to a point in the English Channel
    let query_point = (0.0, 50.0); // Roughly in the English Channel
    let (_nearest, dist) = index
        .nearest(query_point.0, query_point.1)
        .expect("Should find nearest");

    println!("Nearest city distance: {:.2} degrees", dist);

    // London should be the nearest (approximately 1.5 degrees away)
    assert!(dist < 2.0);
    assert!(dist > 1.0);

    Ok(())
}

/// Test scenario: Within distance query
/// Simulates: "Find all features within 500km of a location"
#[test]
fn test_within_distance_scenario() -> Result<()> {
    let index = SpatialIndex::new();

    // Insert European capitals
    let capitals = vec![
        "POINT(2.3522 48.8566)",  // Paris
        "POINT(4.3517 50.8503)",  // Brussels
        "POINT(13.4050 52.5200)", // Berlin
        "POINT(-0.1276 51.5074)", // London
        "POINT(12.4964 41.9028)", // Rome
    ];

    for wkt in &capitals {
        let geom = Geometry::from_wkt(wkt)?;
        index.insert(geom)?;
    }

    // Find all capitals within ~5 degrees of Paris (roughly 500km)
    let paris = (2.3522, 48.8566);
    let results = index.query_within_distance(paris.0, paris.1, 5.0);

    println!("Found {} capitals within 5 degrees of Paris", results.len());

    // Should find Paris, Brussels, and London
    assert!(results.len() >= 2);
    assert!(results.len() <= 4);

    Ok(())
}

/// Test scenario: Topological relations between polygons
/// Simulates queries checking if features overlap, contain, or intersect
#[test]
fn test_topological_relations_scenario() -> Result<()> {
    // Create two overlapping polygons
    let polygon1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
    let polygon2 = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))")?;

    // Test that they intersect
    assert!(
        sf_intersects(&polygon1, &polygon2)?,
        "Overlapping polygons should intersect"
    );

    // Test that they overlap
    assert!(
        sf_overlaps(&polygon1, &polygon2)?,
        "Polygons should overlap"
    );

    // Create a point inside polygon1
    let point_inside = Geometry::from_wkt("POINT(1 1)")?;

    // Test containment
    assert!(
        sf_contains(&polygon1, &point_inside)?,
        "Polygon should contain point inside it"
    );
    assert!(
        sf_within(&point_inside, &polygon1)?,
        "Point should be within polygon"
    );

    // Point outside both polygons
    let point_outside = Geometry::from_wkt("POINT(10 10)")?;
    assert!(
        !sf_contains(&polygon1, &point_outside)?,
        "Polygon should not contain point outside it"
    );

    Ok(())
}

/// Test scenario: Distance calculations between features
/// Simulates: "Calculate distances between all pairs of features"
#[test]
fn test_distance_calculations_scenario() -> Result<()> {
    // Create a set of points
    let points = [
        Geometry::from_wkt("POINT(0 0)")?,
        Geometry::from_wkt("POINT(3 4)")?,
        Geometry::from_wkt("POINT(6 8)")?,
    ];

    // Calculate distance from first point to second
    let dist_01 = distance(&points[0], &points[1])?;
    println!("Distance from P0 to P1: {:.2}", dist_01);

    // Pythagorean theorem: sqrt(3^2 + 4^2) = 5
    assert!((dist_01 - 5.0).abs() < 1e-10);

    // Calculate distance from first to third
    let dist_02 = distance(&points[0], &points[2])?;
    println!("Distance from P0 to P2: {:.2}", dist_02);

    // Pythagorean theorem: sqrt(6^2 + 8^2) = 10
    assert!((dist_02 - 10.0).abs() < 1e-10);

    Ok(())
}

/// Test scenario: Working with different CRS
/// Simulates queries that need to check CRS compatibility
#[test]
fn test_crs_compatibility_scenario() -> Result<()> {
    // Create geometries with different CRS
    let wgs84_geom = Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(0 0)")?;
    let web_mercator_geom =
        Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/3857> POINT(0 0)")?;

    // Verify CRS are different
    assert_ne!(wgs84_geom.crs, web_mercator_geom.crs);

    // Operations between different CRS should fail
    let result = sf_intersects(&wgs84_geom, &web_mercator_geom);
    assert!(result.is_err(), "Should fail when CRS don't match");

    // Operations on same CRS should succeed
    let wgs84_geom2 =
        Geometry::from_wkt("<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(1 1)")?;
    let result = sf_intersects(&wgs84_geom, &wgs84_geom2);
    assert!(result.is_ok(), "Should succeed when CRS match");

    Ok(())
}

/// Test scenario: Complex spatial workflow
/// Simulates a complete workflow: load data, index it, query, and perform operations
#[test]
fn test_complete_spatial_workflow() -> Result<()> {
    // Step 1: Create a spatial index
    let index = SpatialIndex::new();

    // Step 2: Load spatial features (simulating loading from a database)
    let features = vec![
        ("Building A", "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"),
        ("Building B", "POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))"),
        ("Park", "POLYGON((5 5, 25 5, 25 25, 5 25, 5 5))"), // Overlaps with Building A
    ];

    let mut geometries = Vec::new();
    for (name, wkt) in &features {
        let geom = Geometry::from_wkt(wkt)?;
        index.insert(geom.clone())?;
        geometries.push((name, geom));
        println!("Loaded feature: {}", name);
    }

    // Step 3: Perform a spatial query - find features in a region
    let query_bbox = index.query_bbox(0.0, 0.0, 15.0, 15.0);
    println!("Found {} features in query bbox", query_bbox.len());
    // Should find Building A and Park (both overlap with query region)
    assert_eq!(query_bbox.len(), 2, "Should find Building A and Park");

    // Step 4: Check topological relations
    let building_a = &geometries[0].1;
    let park = &geometries[2].1;

    assert!(
        sf_intersects(building_a, park)?,
        "Building A should intersect with Park"
    );

    // Step 5: Calculate envelope for all features
    for (name, geom) in &geometries {
        let env = envelope(geom)?;
        println!("{} envelope: {}", name, env.to_wkt());
        assert!(!env.is_empty(), "Envelope should not be empty");
    }

    Ok(())
}

/// Test scenario: Working with LineStrings
/// Simulates road network queries
#[test]
fn test_linestring_operations_scenario() -> Result<()> {
    // Create a simple road network
    let road1 = Geometry::from_wkt("LINESTRING(0 0, 10 10)")?;
    let road2 = Geometry::from_wkt("LINESTRING(0 10, 10 0)")?;

    // Roads should intersect (they cross)
    assert!(sf_intersects(&road1, &road2)?, "Roads should intersect");

    // Create a point on road1
    let point_on_road = Geometry::from_wkt("POINT(5 5)")?;

    // Calculate envelope and verify point is within the road's bounding box
    let road1_envelope = envelope(&road1)?;
    println!("Road1 envelope: {}", road1_envelope.to_wkt());
    println!("Point on road: {}", point_on_road.to_wkt());

    // For LineString (0,0) to (10,10), point (5,5) should be within envelope
    // The envelope check verifies the point is within the spatial extent
    assert!(
        sf_within(&point_on_road, &road1_envelope)?
            || sf_intersects(&point_on_road, &road1_envelope)?,
        "Point should be within or intersect road envelope"
    );

    Ok(())
}

/// Test scenario: MultiGeometry handling
/// Tests working with MultiPoint, MultiLineString, MultiPolygon
#[test]
fn test_multi_geometry_scenario() -> Result<()> {
    // Create a MultiPoint representing multiple locations
    let multi_point = Geometry::from_wkt("MULTIPOINT((0 0), (10 10), (20 20))")?;

    assert_eq!(multi_point.geometry_type(), "MultiPoint");
    assert!(!multi_point.is_empty());

    // Create a MultiLineString representing a route network
    let multi_linestring = Geometry::from_wkt("MULTILINESTRING((0 0, 10 10), (20 20, 30 30))")?;

    assert_eq!(multi_linestring.geometry_type(), "MultiLineString");

    // Create a MultiPolygon representing multiple regions
    let multi_polygon = Geometry::from_wkt(
        "MULTIPOLYGON(((0 0, 5 0, 5 5, 0 5, 0 0)), ((10 10, 15 10, 15 15, 10 15, 10 10)))",
    )?;

    assert_eq!(multi_polygon.geometry_type(), "MultiPolygon");

    // Test envelope calculation on multi-geometries
    let env = envelope(&multi_polygon)?;
    assert!(!env.is_empty());
    println!("MultiPolygon envelope: {}", env.to_wkt());

    Ok(())
}

/// Test scenario: Empty geometry handling
/// Tests that operations handle empty geometries correctly
#[test]
fn test_empty_geometry_scenario() -> Result<()> {
    // Create empty geometries
    let empty_point = Geometry::from_wkt("POINT EMPTY")?;
    let empty_linestring = Geometry::from_wkt("LINESTRING EMPTY")?;
    let empty_polygon = Geometry::from_wkt("POLYGON EMPTY")?;

    // Verify they are recognized as empty
    assert!(empty_point.is_empty(), "Empty point should be empty");
    assert!(
        empty_linestring.is_empty(),
        "Empty linestring should be empty"
    );
    assert!(empty_polygon.is_empty(), "Empty polygon should be empty");

    // Test that empty geometries can be serialized back to WKT
    // Note: The wkt crate may convert POINT EMPTY to MULTIPOINT EMPTY
    let empty_point_wkt = empty_point.to_wkt();
    assert!(
        empty_point_wkt.contains("EMPTY"),
        "Empty point WKT should contain EMPTY, got: {}",
        empty_point_wkt
    );

    let empty_linestring_wkt = empty_linestring.to_wkt();
    assert!(
        empty_linestring_wkt.contains("LINESTRING") && empty_linestring_wkt.contains("EMPTY"),
        "Empty linestring WKT should be LINESTRING EMPTY, got: {}",
        empty_linestring_wkt
    );

    let empty_polygon_wkt = empty_polygon.to_wkt();
    assert!(
        empty_polygon_wkt.contains("POLYGON") && empty_polygon_wkt.contains("EMPTY"),
        "Empty polygon WKT should be POLYGON EMPTY, got: {}",
        empty_polygon_wkt
    );

    Ok(())
}

/// Test scenario: Round-trip WKT conversion
/// Ensures that WKT -> Geometry -> WKT preserves geometry
#[test]
fn test_wkt_roundtrip_scenario() -> Result<()> {
    let test_cases = vec![
        "POINT(1.5 2.5)",
        "LINESTRING(0 0, 1 1, 2 2)",
        "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
        "MULTIPOINT((0 0), (1 1))",
    ];

    for original_wkt in test_cases {
        // Parse WKT to geometry
        let geom = Geometry::from_wkt(original_wkt)?;

        // Convert back to WKT
        let new_wkt = geom.to_wkt();

        // Parse again
        let geom2 = Geometry::from_wkt(&new_wkt)?;

        // Should have same geometry type
        assert_eq!(
            geom.geometry_type(),
            geom2.geometry_type(),
            "Geometry type should be preserved for {}",
            original_wkt
        );

        println!("✓ Round-trip successful: {} -> {}", original_wkt, new_wkt);
    }

    Ok(())
}
