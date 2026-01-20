//! Egenhofer Topological Relations Example
//!
//! This example demonstrates the 8 Egenhofer topological relations based on
//! the 4-intersection model (boundary-boundary, boundary-interior,
//! interior-boundary, interior-interior).
//!
//! Run with: cargo run --example egenhofer_relations --features geos-backend

use oxirs_geosparql::error::Result;

#[cfg(feature = "geos-backend")]
use oxirs_geosparql::geometry::Geometry;

#[cfg(feature = "geos-backend")]
use oxirs_geosparql::functions::egenhofer::{
    eh_contains, eh_covered_by, eh_covers, eh_disjoint, eh_equals, eh_inside, eh_meet, eh_overlap,
};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL Egenhofer Relations Example ===\n");

    #[cfg(feature = "geos-backend")]
    {
        // Example 1: eh_equals - Geometries are spatially equal
        println!("1. EGENHOFER EQUALS (eh_equals):");
        println!("   Two geometries represent the same region\n");

        let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
        let poly2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;

        let equals = eh_equals(&poly1, &poly2)?;
        println!("   Polygon A equals Polygon B: {}", equals);
        assert!(equals);
        println!("   ✅ Same geometry\n");

        // Example 2: eh_disjoint - No intersection
        println!("2. EGENHOFER DISJOINT (eh_disjoint):");
        println!("   No common points, boundaries or interiors\n");

        let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")?;
        let poly2 = Geometry::from_wkt("POLYGON((3 3, 5 3, 5 5, 3 5, 3 3))")?;

        let disjoint = eh_disjoint(&poly1, &poly2)?;
        println!("   Polygon A disjoint from Polygon B: {}", disjoint);
        assert!(disjoint);
        println!("   ✅ No spatial overlap\n");

        // Example 3: eh_meet - Boundaries touch, interiors don't
        println!("3. EGENHOFER MEET (eh_meet):");
        println!("   Boundaries intersect, but interiors are disjoint\n");

        let poly1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")?;
        let poly2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))")?;

        let meet = eh_meet(&poly1, &poly2)?;
        println!("   Polygon A meets Polygon B: {}", meet);
        println!("   (Sharing edge at x=2)");
        println!("   ✅ Adjacent polygons\n");

        // Example 4: eh_overlap - Interiors and boundaries intersect
        println!("4. EGENHOFER OVERLAP (eh_overlap):");
        println!("   Both interiors and boundaries have intersection\n");

        let poly1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
        let poly2 = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))")?;

        let overlap = eh_overlap(&poly1, &poly2)?;
        println!("   Polygon A overlaps Polygon B: {}", overlap);
        assert!(overlap);
        println!("   ✅ Partial overlap\n");

        // Example 5: eh_covers - A covers B (B is inside or on boundary of A)
        println!("5. EGENHOFER COVERS (eh_covers):");
        println!("   Geometry A completely covers geometry B\n");

        let large_poly = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
        let small_poly = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))")?;

        let covers = eh_covers(&large_poly, &small_poly)?;
        println!("   Large polygon covers small polygon: {}", covers);
        assert!(covers);
        println!("   ✅ Complete coverage\n");

        // Example 6: eh_covered_by - B is covered by A (inverse of covers)
        println!("6. EGENHOFER COVERED_BY (eh_covered_by):");
        println!("   Geometry A is covered by geometry B\n");

        let covered_by = eh_covered_by(&small_poly, &large_poly)?;
        println!("   Small polygon covered by large polygon: {}", covered_by);
        assert!(covered_by);
        println!("   ✅ Inverse relationship\n");

        // Example 7: eh_inside - A is completely inside B (no boundary touch)
        println!("7. EGENHOFER INSIDE (eh_inside):");
        println!("   Geometry A is completely inside geometry B\n");

        let outer = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
        let inner = Geometry::from_wkt("POLYGON((3 3, 7 3, 7 7, 3 7, 3 3))")?;

        let inside = eh_inside(&inner, &outer)?;
        println!("   Inner polygon inside outer polygon: {}", inside);
        assert!(inside);
        println!("   ✅ Interior containment\n");

        // Example 8: eh_contains - A contains B (inverse of inside)
        println!("8. EGENHOFER CONTAINS (eh_contains):");
        println!("   Geometry A contains geometry B\n");

        let contains = eh_contains(&outer, &inner)?;
        println!("   Outer polygon contains inner polygon: {}", contains);
        assert!(contains);
        println!("   ✅ Inverse relationship\n");

        // Real-world example: Administrative boundaries
        println!("=== REAL-WORLD EXAMPLE: Administrative Boundaries ===\n");

        let country = Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))")?;
        let province = Geometry::from_wkt("POLYGON((10 10, 40 10, 40 40, 10 40, 10 10))")?;
        let city = Geometry::from_wkt("POLYGON((15 15, 30 15, 30 30, 15 30, 15 15))")?;

        println!(
            "Country contains Province: {}",
            eh_contains(&country, &province)?
        );
        println!("Province contains City: {}", eh_contains(&province, &city)?);
        println!("City inside Province: {}", eh_inside(&city, &province)?);
        println!(
            "City covered by Country: {}",
            eh_covered_by(&city, &country)?
        );
        println!("✅ Hierarchical administrative structure\n");

        // Real-world example: Land parcels
        println!("=== REAL-WORLD EXAMPLE: Land Parcels ===\n");

        let parcel1 = Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))")?;
        let parcel2 = Geometry::from_wkt("POLYGON((5 0, 10 0, 10 5, 5 5, 5 0))")?;
        let parcel3 = Geometry::from_wkt("POLYGON((10 5, 15 5, 15 10, 10 10, 10 5))")?;

        println!("Parcel 1 meets Parcel 2: {}", eh_meet(&parcel1, &parcel2)?);
        println!(
            "Parcel 2 disjoint from Parcel 3: {}",
            eh_disjoint(&parcel2, &parcel3)?
        );
        println!("✅ Adjacent and separated parcels\n");

        // Summary table
        println!("=== EGENHOFER RELATIONS SUMMARY ===\n");
        println!("┌─────────────┬──────────────────────────────────────────────────────┐");
        println!("│ Relation    │ Description                                          │");
        println!("├─────────────┼──────────────────────────────────────────────────────┤");
        println!("│ eh_equals   │ Same geometry (all intersections match)              │");
        println!("│ eh_disjoint │ No common points (completely separate)               │");
        println!("│ eh_meet     │ Boundaries touch, interiors don't intersect          │");
        println!("│ eh_overlap  │ Interiors AND boundaries intersect                   │");
        println!("│ eh_covers   │ A covers B (B inside or on boundary of A)            │");
        println!("│ eh_covered_by│ A covered by B (inverse of covers)                  │");
        println!("│ eh_inside   │ A completely inside B (no boundary touch)            │");
        println!("│ eh_contains │ A contains B (inverse of inside)                     │");
        println!("└─────────────┴──────────────────────────────────────────────────────┘\n");

        println!("=== 4-INTERSECTION MODEL ===\n");
        println!("Egenhofer relations are based on the 4-intersection model:");
        println!("  • Interior(A) ∩ Interior(B)");
        println!("  • Interior(A) ∩ Boundary(B)");
        println!("  • Boundary(A) ∩ Interior(B)");
        println!("  • Boundary(A) ∩ Boundary(B)\n");

        println!("Different combinations of empty/non-empty intersections");
        println!("define the 8 topological relations.\n");

        println!("=== COMPARISON WITH SIMPLE FEATURES ===\n");
        println!("Simple Features (DE-9IM):      Egenhofer (4-intersection):");
        println!("  • sf_equals                    • eh_equals");
        println!("  • sf_disjoint                  • eh_disjoint");
        println!("  • sf_intersects                • (no direct equivalent)");
        println!("  • sf_touches                   • eh_meet");
        println!("  • sf_crosses                   • (different model)");
        println!("  • sf_within                    • eh_inside");
        println!("  • sf_contains                  • eh_contains");
        println!("  • sf_overlaps                  • eh_overlap\n");

        println!("=== USE CASES ===\n");
        println!("1. **Spatial Databases**: Qualitative spatial queries");
        println!("   - Find all regions that meet a boundary");
        println!("   - Identify contained features\n");

        println!("2. **GIS Analysis**: Topological relationships");
        println!("   - Administrative boundary analysis");
        println!("   - Land parcel adjacency\n");

        println!("3. **Cartography**: Map overlay operations");
        println!("   - Identifying overlapping map features");
        println!("   - Boundary alignment verification\n");

        println!("4. **Urban Planning**: Spatial constraints");
        println!("   - Zoning compliance (inside/outside checks)");
        println!("   - Building setback validation\n");

        println!("=== Example completed successfully! ===");
    }

    #[cfg(not(feature = "geos-backend"))]
    {
        println!("❌ This example requires the 'geos-backend' feature to be enabled.\n");
        println!("Please run with:");
        println!("   cargo run --example egenhofer_relations --features geos-backend\n");
        println!("Note: This requires GEOS library to be installed:");
        println!("  • macOS: brew install geos");
        println!("  • Ubuntu: sudo apt-get install libgeos-dev");
        println!("  • Fedora: sudo dnf install geos-devel");
    }

    Ok(())
}
