//! RCC8 (Region Connection Calculus) Relations Example
//!
//! This example demonstrates the 8 RCC8 topological relations for
//! qualitative spatial reasoning about regions.
//!
//! Run with: cargo run --example rcc8_relations --features geos-backend

use oxirs_geosparql::error::Result;

#[cfg(feature = "geos-backend")]
use oxirs_geosparql::geometry::Geometry;

#[cfg(feature = "geos-backend")]
use oxirs_geosparql::functions::rcc8::{
    rcc8_dc, rcc8_ec, rcc8_eq, rcc8_ntpp, rcc8_ntppi, rcc8_po, rcc8_tpp, rcc8_tppi,
};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL RCC8 Relations Example ===\n");

    #[cfg(feature = "geos-backend")]
    {
        // Example 1: rcc8_eq - Equal regions
        println!("1. RCC8-EQ (Equal):");
        println!("   Two regions are identical\n");

        let region1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
        let region2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;

        let eq = rcc8_eq(&region1, &region2)?;
        println!("   Region A = Region B: {}", eq);
        assert!(eq);
        println!("   ✅ Identical regions\n");

        // Example 2: rcc8_dc - Disconnected (no connection)
        println!("2. RCC8-DC (Disconnected):");
        println!("   Regions are completely separate\n");

        let region1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")?;
        let region2 = Geometry::from_wkt("POLYGON((4 4, 6 4, 6 6, 4 6, 4 4))")?;

        let dc = rcc8_dc(&region1, &region2)?;
        println!("   Region A disconnected from Region B: {}", dc);
        assert!(dc);
        println!("   ✅ No spatial connection\n");

        // Example 3: rcc8_ec - Externally connected (touching boundaries)
        println!("3. RCC8-EC (Externally Connected):");
        println!("   Regions touch at boundaries only\n");

        let region1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")?;
        let region2 = Geometry::from_wkt("POLYGON((2 0, 4 0, 4 2, 2 2, 2 0))")?;

        let ec = rcc8_ec(&region1, &region2)?;
        println!("   Region A externally connected to Region B: {}", ec);
        println!("   (Sharing edge at x=2)");
        println!("   ✅ Boundary connection\n");

        // Example 4: rcc8_po - Partially overlapping
        println!("4. RCC8-PO (Partially Overlapping):");
        println!("   Regions overlap but neither contains the other\n");

        let region1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))")?;
        let region2 = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))")?;

        let po = rcc8_po(&region1, &region2)?;
        println!("   Region A partially overlaps Region B: {}", po);
        assert!(po);
        println!("   ✅ Partial intersection\n");

        // Example 5: rcc8_tpp - Tangential Proper Part
        println!("5. RCC8-TPP (Tangential Proper Part):");
        println!("   A is inside B and boundaries touch\n");

        let outer = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
        let inner = Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))")?;

        let tpp = rcc8_tpp(&inner, &outer)?;
        println!("   Inner region is TPP of outer region: {}", tpp);
        println!("   (Sharing corner and edges)");
        println!("   ✅ Tangential containment\n");

        // Example 6: rcc8_tppi - TPP Inverse (inverse of TPP)
        println!("6. RCC8-TPPi (Tangential Proper Part Inverse):");
        println!("   B is inside A and boundaries touch (inverse)\n");

        let tppi = rcc8_tppi(&outer, &inner)?;
        println!("   Outer region is TPPi of inner region: {}", tppi);
        println!("   ✅ Inverse relationship\n");

        // Example 7: rcc8_ntpp - Non-Tangential Proper Part
        println!("7. RCC8-NTPP (Non-Tangential Proper Part):");
        println!("   A is completely inside B, boundaries don't touch\n");

        let outer = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
        let inner = Geometry::from_wkt("POLYGON((3 3, 7 3, 7 7, 3 7, 3 3))")?;

        let ntpp = rcc8_ntpp(&inner, &outer)?;
        println!("   Inner region is NTPP of outer region: {}", ntpp);
        assert!(ntpp);
        println!("   ✅ Complete interior containment\n");

        // Example 8: rcc8_ntppi - NTPP Inverse (inverse of NTPP)
        println!("8. RCC8-NTPPi (Non-Tangential Proper Part Inverse):");
        println!("   B is completely inside A, boundaries don't touch\n");

        let ntppi = rcc8_ntppi(&outer, &inner)?;
        println!("   Outer region is NTPPi of inner region: {}", ntppi);
        assert!(ntppi);
        println!("   ✅ Inverse relationship\n");

        // Real-world example: Geographic containment
        println!("=== REAL-WORLD EXAMPLE: Geographic Regions ===\n");

        let ocean = Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))")?;
        let island = Geometry::from_wkt("POLYGON((40 40, 60 40, 60 60, 40 60, 40 40))")?;
        let lake_in_island = Geometry::from_wkt("POLYGON((45 45, 55 45, 55 55, 45 55, 45 45))")?;

        println!("Island is NTPP of Ocean: {}", rcc8_ntpp(&island, &ocean)?);
        println!(
            "Lake is NTPP of Island: {}",
            rcc8_ntpp(&lake_in_island, &island)?
        );
        println!(
            "Ocean contains Island (NTPPi): {}",
            rcc8_ntppi(&ocean, &island)?
        );
        println!("✅ Nested geographic containment\n");

        // Real-world example: Building and rooms
        println!("=== REAL-WORLD EXAMPLE: Building Layout ===\n");

        let building = Geometry::from_wkt("POLYGON((0 0, 20 0, 20 20, 0 20, 0 0))")?;
        let room1 = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
        let room2 = Geometry::from_wkt("POLYGON((10 0, 20 0, 20 10, 10 10, 10 0))")?;

        println!(
            "Room 1 is TPP of Building: {}",
            rcc8_tpp(&room1, &building)?
        );
        println!(
            "Room 2 is TPP of Building: {}",
            rcc8_tpp(&room2, &building)?
        );
        println!("Room 1 and Room 2 are EC: {}", rcc8_ec(&room1, &room2)?);
        println!("✅ Adjacent rooms in building\n");

        // Real-world example: Countries and borders
        println!("=== REAL-WORLD EXAMPLE: Country Borders ===\n");

        let country_a = Geometry::from_wkt("POLYGON((0 0, 5 0, 5 5, 0 5, 0 0))")?;
        let country_b = Geometry::from_wkt("POLYGON((5 0, 10 0, 10 5, 5 5, 5 0))")?;
        let country_c = Geometry::from_wkt("POLYGON((10 5, 15 5, 15 10, 10 10, 10 5))")?;

        println!(
            "Country A and B share border (EC): {}",
            rcc8_ec(&country_a, &country_b)?
        );
        println!(
            "Country B and C are disconnected (DC): {}",
            rcc8_dc(&country_b, &country_c)?
        );
        println!("✅ Border relationships\n");

        // Summary table
        println!("=== RCC8 RELATIONS SUMMARY ===\n");
        println!("┌──────────┬────────────────────────────────────────────────────────┐");
        println!("│ Relation │ Description                                            │");
        println!("├──────────┼────────────────────────────────────────────────────────┤");
        println!("│ EQ       │ Equal (identical regions)                              │");
        println!("│ DC       │ Disconnected (no connection)                           │");
        println!("│ EC       │ Externally Connected (touching boundaries)             │");
        println!("│ PO       │ Partially Overlapping (intersecting)                   │");
        println!("│ TPP      │ Tangential Proper Part (inside, boundary touches)     │");
        println!("│ TPPi     │ TPP Inverse (contains, boundary touches)               │");
        println!("│ NTPP     │ Non-Tangential Proper Part (inside, no boundary touch)│");
        println!("│ NTPPi    │ NTPP Inverse (contains, no boundary touch)             │");
        println!("└──────────┴────────────────────────────────────────────────────────┘\n");

        println!("=== RELATION HIERARCHY ===\n");
        println!("Connection strength (weak to strong):");
        println!("  DC → EC → PO → TPP → NTPP");
        println!("         ↘  ↑  ↗");
        println!("           EQ\n");

        println!("Inverse pairs:");
        println!("  • TPP ↔ TPPi");
        println!("  • NTPP ↔ NTPPi\n");

        println!("=== COMPARISON WITH OTHER MODELS ===\n");
        println!("RCC8 vs Simple Features:");
        println!("  • RCC8-EQ     ≈ sf_equals");
        println!("  • RCC8-DC     ≈ sf_disjoint");
        println!("  • RCC8-EC     ≈ sf_touches");
        println!("  • RCC8-PO     ≈ sf_overlaps (partial)");
        println!("  • RCC8-NTPP   ≈ sf_within (strict)");
        println!("  • RCC8-NTPPi  ≈ sf_contains (strict)\n");

        println!("RCC8 vs Egenhofer:");
        println!("  • RCC8-EQ     = eh_equals");
        println!("  • RCC8-DC     = eh_disjoint");
        println!("  • RCC8-EC     = eh_meet");
        println!("  • RCC8-PO     ≈ eh_overlap");
        println!("  • RCC8-NTPP   = eh_inside");
        println!("  • RCC8-NTPPi  = eh_contains\n");

        println!("=== USE CASES ===\n");
        println!("1. **Qualitative Spatial Reasoning**: AI and robotics");
        println!("   - Spatial planning without precise coordinates");
        println!("   - High-level spatial descriptions\n");

        println!("2. **Geographic Information Systems**: Region analysis");
        println!("   - Administrative boundary relationships");
        println!("   - Containment hierarchies\n");

        println!("3. **Constraint Satisfaction**: Spatial constraints");
        println!("   - Room layout planning");
        println!("   - Architectural design validation\n");

        println!("4. **Semantic Web**: RDF spatial reasoning");
        println!("   - Ontology-based spatial queries");
        println!("   - Knowledge graph spatial relations\n");

        println!("=== MATHEMATICAL PROPERTIES ===\n");
        println!("RCC8 relations are:");
        println!("  • Jointly Exhaustive: Every pair satisfies exactly one relation");
        println!("  • Mutually Exclusive: No pair satisfies multiple relations");
        println!(
            "  • Composition Table: Relations can be composed (A rel1 B, B rel2 C → A rel3 C)\n"
        );

        println!("=== Example completed successfully! ===");
    }

    #[cfg(not(feature = "geos-backend"))]
    {
        println!("❌ This example requires the 'geos-backend' feature to be enabled.\n");
        println!("Please run with:");
        println!("   cargo run --example rcc8_relations --features geos-backend\n");
        println!("Note: This requires GEOS library to be installed:");
        println!("  • macOS: brew install geos");
        println!("  • Ubuntu: sudo apt-get install libgeos-dev");
        println!("  • Fedora: sudo dnf install geos-devel");
    }

    Ok(())
}
