//! Geometric set operations example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Intersection - find common area between geometries
//! - Union - combine geometries into one
//! - Difference - subtract one geometry from another
//! - Symmetric difference - XOR operation on geometries
//!
//! Run with: cargo run --example geometric_set_operations

use oxirs_geosparql::error::Result;
use oxirs_geosparql::functions::geometric_operations::{
    difference, intersection, sym_difference, union,
};
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Geometric Set Operations Example ===\n");

    // Create two overlapping square regions
    let region_a = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
    let region_b = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))")?;

    println!("Region A: {}", region_a.to_wkt());
    println!("Region B: {}", region_b.to_wkt());

    // 1. Intersection - common area
    println!("\n1. INTERSECTION (A ∩ B) - Common area:");
    match intersection(&region_a, &region_b)? {
        Some(intersect) => {
            println!("   Result: {}", intersect.to_wkt());
            println!("   This is the area where both regions overlap (2,2) to (4,4)");
        }
        None => {
            println!("   No intersection found");
        }
    }

    // 2. Union - combined area
    println!("\n2. UNION (A ∪ B) - Combined area:");
    let union_result = union(&region_a, &region_b)?;
    println!("   Result: {}", union_result.to_wkt());
    println!("   This is the total area covered by either region");

    // 3. Difference - area in A but not in B
    println!("\n3. DIFFERENCE (A - B) - Area in A but not in B:");
    let diff_a_b = difference(&region_a, &region_b)?;
    println!("   Result: {}", diff_a_b.to_wkt());
    println!("   This is the part of Region A that doesn't overlap with Region B");

    // 4. Difference - area in B but not in A
    println!("\n4. DIFFERENCE (B - A) - Area in B but not in A:");
    let diff_b_a = difference(&region_b, &region_a)?;
    println!("   Result: {}", diff_b_a.to_wkt());
    println!("   This is the part of Region B that doesn't overlap with Region A");

    // 5. Symmetric Difference - XOR operation
    println!("\n5. SYMMETRIC DIFFERENCE (A ⊕ B) - XOR operation:");
    let sym_diff = sym_difference(&region_a, &region_b)?;
    println!("   Result: {}", sym_diff.to_wkt());
    println!("   This is the area in either region but not in both");
    println!("   Equivalent to: (A - B) ∪ (B - A)");

    // Real-world example: Urban planning
    println!("\n\n=== Real-World Example: Urban Planning ===\n");

    let commercial_zone = Geometry::from_wkt("POLYGON((0 0, 50 0, 50 50, 0 50, 0 0))")?;
    let park_zone = Geometry::from_wkt("POLYGON((30 30, 80 30, 80 80, 30 80, 30 30))")?;
    let residential_zone = Geometry::from_wkt("POLYGON((60 0, 100 0, 100 40, 60 40, 60 0))")?;

    println!("Commercial zone: {}", commercial_zone.to_wkt());
    println!("Park zone: {}", park_zone.to_wkt());
    println!("Residential zone: {}", residential_zone.to_wkt());

    // Find overlap between commercial and park zones
    println!("\nCommercial ∩ Park (area that needs rezoning):");
    if let Some(overlap) = intersection(&commercial_zone, &park_zone)? {
        println!(
            "   {} - Needs to be designated as either commercial or park",
            overlap.to_wkt()
        );
    }

    // Combine commercial and residential for tax purposes
    println!("\nCommercial ∪ Residential (total non-park area):");
    let non_park = union(&commercial_zone, &residential_zone)?;
    println!("   {}", non_park.to_wkt());

    // Find actual park area (excluding commercial overlap)
    println!("\nPark - Commercial (actual pure park area):");
    let pure_park = difference(&park_zone, &commercial_zone)?;
    println!("   {}", pure_park.to_wkt());

    // Complex example: Multiple operations
    println!("\n\n=== Complex Example: Multiple Operations ===\n");

    let area1 = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
    let area2 = Geometry::from_wkt("POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))")?;
    let area3 = Geometry::from_wkt("POLYGON((10 0, 20 0, 20 10, 10 10, 10 0))")?;

    println!("Three regions:");
    println!("  Area 1: {}", area1.to_wkt());
    println!("  Area 2: {}", area2.to_wkt());
    println!("  Area 3: {}", area3.to_wkt());

    // Step 1: Union of area1 and area3
    println!("\nStep 1: Union of Area1 and Area3:");
    let combined_1_3 = union(&area1, &area3)?;
    println!("   Result: {}", combined_1_3.to_wkt());

    // Step 2: Intersection with area2
    println!("\nStep 2: Intersection of (Area1 ∪ Area3) with Area2:");
    if let Some(final_result) = intersection(&combined_1_3, &area2)? {
        println!("   Result: {}", final_result.to_wkt());
        println!("   This is the area where Area2 overlaps with either Area1 or Area3");
    }

    // Polygon with hole example
    println!("\n\n=== Polygon with Hole Example ===\n");

    let outer_boundary = Geometry::from_wkt("POLYGON((0 0, 20 0, 20 20, 0 20, 0 0))")?;
    let inner_exclusion = Geometry::from_wkt("POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))")?;

    println!("Outer boundary: {}", outer_boundary.to_wkt());
    println!("Inner exclusion: {}", inner_exclusion.to_wkt());

    println!("\nCreating a donut shape (boundary - exclusion):");
    let donut = difference(&outer_boundary, &inner_exclusion)?;
    println!("   Result: {}", donut.to_wkt());
    println!("   This creates a frame or border shape");

    // Non-overlapping polygons
    println!("\n\n=== Non-Overlapping Polygons ===\n");

    let poly1 = Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))")?;
    let poly2 = Geometry::from_wkt("POLYGON((10 10, 15 10, 15 15, 10 15, 10 10))")?;

    println!("Polygon 1: {}", poly1.to_wkt());
    println!("Polygon 2: {}", poly2.to_wkt());

    println!("\nIntersection of non-overlapping polygons:");
    match intersection(&poly1, &poly2)? {
        Some(result) => println!("   Unexpected result: {}", result.to_wkt()),
        None => println!("   None (as expected - no overlap)"),
    }

    println!("\nUnion of non-overlapping polygons:");
    let union_separate = union(&poly1, &poly2)?;
    println!("   Result: {}", union_separate.to_wkt());
    println!("   Creates a MultiPolygon with both shapes");

    // MultiPolygon operations
    println!("\n\n=== MultiPolygon Operations ===\n");

    let multi = Geometry::from_wkt(
        "MULTIPOLYGON(((0 0, 5 0, 5 5, 0 5, 0 0)), ((10 10, 15 10, 15 15, 10 15, 10 10)))",
    )?;
    let cutting_polygon = Geometry::from_wkt("POLYGON((3 3, 12 3, 12 12, 3 12, 3 3))")?;

    println!("MultiPolygon: {}", multi.to_wkt());
    println!("Cutting polygon: {}", cutting_polygon.to_wkt());

    println!("\nIntersection:");
    if let Some(result) = intersection(&multi, &cutting_polygon)? {
        println!("   {}", result.to_wkt());
    }

    println!("\nDifference (Multi - Cutting):");
    let diff_result = difference(&multi, &cutting_polygon)?;
    println!("   {}", diff_result.to_wkt());

    println!("\n=== Example completed successfully! ===");
    Ok(())
}
