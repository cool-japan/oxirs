//! 3D Buffer Operations Example
//!
//! This example demonstrates the advanced 3D buffering capabilities of oxirs-geosparql,
//! including uniform and anisotropic buffers, different cap styles, and Z interpolation strategies.

use oxirs_geosparql::functions::buffer_3d::buffer_3d;
#[cfg(feature = "geos-backend")]
use oxirs_geosparql::functions::buffer_3d::{
    buffer_3d_with_params, BufferParams3D, CapStyle3D, JoinStyle3D, ZInterpolationStrategy,
};
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 3D Buffer Operations Example ===\n");

    // Section 1: Basic 3D Buffer with Uniform Distance
    println!("## Section 1: Basic 3D Buffer with Uniform Distance");
    println!("Creating a 3D buffer around a point with uniform distance...\n");

    let point_3d = Geometry::from_wkt("POINT Z (100 200 50)")?;
    println!("Original point: {}", point_3d.to_wkt());

    #[cfg(feature = "geos-backend")]
    {
        let buffered_uniform = buffer_3d(&point_3d, 10.0)?;
        println!(
            "Buffered (uniform 10m): {}",
            buffered_uniform.geometry_type()
        );
        println!("Has Z coordinates: {}", buffered_uniform.is_3d());
        println!();
    }

    // Section 2: Anisotropic 3D Buffer (Different Horizontal and Vertical Distances)
    println!("## Section 2: Anisotropic 3D Buffer");
    println!("Creating a 3D buffer with different horizontal and vertical distances...\n");

    #[cfg(feature = "geos-backend")]
    {
        let params_aniso = BufferParams3D::anisotropic(15.0, 5.0); // 15m horizontal, 5m vertical
        let buffered_aniso = buffer_3d_with_params(&point_3d, &params_aniso)?;
        println!("Horizontal distance: 15m, Vertical distance: 5m");
        println!("Buffered geometry type: {}", buffered_aniso.geometry_type());
        println!();
    }

    // Section 3: Different Cap Styles
    println!("## Section 3: Different Cap Styles");
    println!("Demonstrating different 3D cap styles...\n");

    #[cfg(feature = "geos-backend")]
    {
        // Spherical cap (default - rounded in all directions)
        let params_spherical = BufferParams3D::uniform(10.0).with_cap_style(CapStyle3D::Spherical);
        let buffered_spherical = buffer_3d_with_params(&point_3d, &params_spherical)?;
        println!("Spherical cap: {}", buffered_spherical.geometry_type());

        // Cylindrical cap (rounded horizontally, flat vertically)
        let params_cylindrical =
            BufferParams3D::uniform(10.0).with_cap_style(CapStyle3D::Cylindrical);
        let buffered_cylindrical = buffer_3d_with_params(&point_3d, &params_cylindrical)?;
        println!("Cylindrical cap: {}", buffered_cylindrical.geometry_type());

        // Flat cap (no rounding)
        let params_flat = BufferParams3D::uniform(10.0).with_cap_style(CapStyle3D::Flat);
        let buffered_flat = buffer_3d_with_params(&point_3d, &params_flat)?;
        println!("Flat cap: {}", buffered_flat.geometry_type());
        println!();
    }

    // Section 4: Different Join Styles for 3D Edges
    println!("## Section 4: Different Join Styles");
    println!("Demonstrating different 3D join styles for edges...\n");

    let linestring_3d = Geometry::from_wkt("LINESTRING Z (0 0 10, 10 0 20, 10 10 15)")?;
    println!("Original linestring: {}", linestring_3d.to_wkt());

    #[cfg(feature = "geos-backend")]
    {
        // Round join (default - smooth curves)
        let params_round = BufferParams3D::uniform(5.0).with_join_style(JoinStyle3D::Round);
        let buffered_round = buffer_3d_with_params(&linestring_3d, &params_round)?;
        println!("Round join: {}", buffered_round.geometry_type());

        // Bevel join (straight connections)
        let params_bevel = BufferParams3D::uniform(5.0).with_join_style(JoinStyle3D::Bevel);
        let buffered_bevel = buffer_3d_with_params(&linestring_3d, &params_bevel)?;
        println!("Bevel join: {}", buffered_bevel.geometry_type());

        // Mitre join (sharp corners)
        let params_mitre = BufferParams3D::uniform(5.0).with_join_style(JoinStyle3D::Mitre);
        let buffered_mitre = buffer_3d_with_params(&linestring_3d, &params_mitre)?;
        println!("Mitre join: {}", buffered_mitre.geometry_type());
        println!();
    }

    // Section 5: Z-Coordinate Interpolation Strategies
    println!("## Section 5: Z-Coordinate Interpolation Strategies");
    println!("Demonstrating different Z interpolation methods...\n");

    let polygon_3d = Geometry::from_wkt("POLYGON Z ((0 0 5, 10 0 10, 10 10 8, 0 10 6, 0 0 5))")?;
    println!("Original polygon: {}", polygon_3d.to_wkt());

    #[cfg(feature = "geos-backend")]
    {
        // Average interpolation
        let params_avg =
            BufferParams3D::uniform(3.0).with_z_interpolation(ZInterpolationStrategy::Average);
        let _buffered_avg = buffer_3d_with_params(&polygon_3d, &params_avg)?;
        println!("Average Z interpolation: Done");

        // Preserve interpolation
        let params_preserve =
            BufferParams3D::uniform(3.0).with_z_interpolation(ZInterpolationStrategy::Preserve);
        let _buffered_preserve = buffer_3d_with_params(&polygon_3d, &params_preserve)?;
        println!("Preserve Z interpolation: Done");

        // Linear interpolation
        let params_linear =
            BufferParams3D::uniform(3.0).with_z_interpolation(ZInterpolationStrategy::Linear);
        let _buffered_linear = buffer_3d_with_params(&polygon_3d, &params_linear)?;
        println!("Linear Z interpolation: Done");

        // Smooth interpolation
        let params_smooth =
            BufferParams3D::uniform(3.0).with_z_interpolation(ZInterpolationStrategy::Smooth);
        let _buffered_smooth = buffer_3d_with_params(&polygon_3d, &params_smooth)?;
        println!("Smooth Z interpolation: Done");
        println!();
    }

    // Section 6: Complex 3D Buffer with Multiple Parameters
    println!("## Section 6: Complex 3D Buffer Configuration");
    println!("Creating a 3D buffer with multiple custom parameters...\n");

    #[cfg(feature = "geos-backend")]
    {
        let complex_params = BufferParams3D::anisotropic(20.0, 10.0)
            .with_cap_style(CapStyle3D::Spherical)
            .with_join_style(JoinStyle3D::Round)
            .with_quadrant_segments(16) // Higher resolution
            .with_z_interpolation(ZInterpolationStrategy::Smooth);

        let buffered_complex = buffer_3d_with_params(&linestring_3d, &complex_params)?;
        println!("Complex buffer created with:");
        println!("  - Horizontal distance: 20m");
        println!("  - Vertical distance: 10m");
        println!("  - Cap style: Spherical");
        println!("  - Join style: Round");
        println!("  - Quadrant segments: 16");
        println!("  - Z interpolation: Smooth");
        println!("Result: {}", buffered_complex.geometry_type());
        println!();
    }

    // Section 7: 3D Building Footprint Buffer
    println!("## Section 7: Real-World Example - 3D Building Footprint");
    println!("Creating a safety buffer around a building...\n");

    // Building footprint with height information
    #[cfg(feature = "geos-backend")]
    let building = Geometry::from_wkt(
        "POLYGON Z ((0 0 0, 30 0 0, 30 20 0, 0 20 0, 0 0 0), \
                     (0 0 30, 30 0 30, 30 20 30, 0 20 30, 0 0 30))",
    )?;

    #[cfg(feature = "geos-backend")]
    {
        // Create a 5m safety buffer around the building (horizontal)
        // and 2m clearance above/below (vertical)
        let safety_params = BufferParams3D::anisotropic(5.0, 2.0)
            .with_cap_style(CapStyle3D::Cylindrical) // Flat top/bottom, rounded sides
            .with_join_style(JoinStyle3D::Round)
            .with_z_interpolation(ZInterpolationStrategy::Preserve);

        let safety_zone = buffer_3d_with_params(&building, &safety_params)?;
        println!("Safety zone created:");
        println!("  - 5m horizontal clearance");
        println!("  - 2m vertical clearance");
        println!("  - Geometry type: {}", safety_zone.geometry_type());
        println!();
    }

    // Section 8: 3D Underground Utility Buffer
    println!("## Section 8: Underground Utility Protection Zone");
    println!("Creating a protection zone around underground pipes...\n");

    let utility_pipe = Geometry::from_wkt("LINESTRING Z (0 0 -3, 50 0 -3, 50 50 -2.5)")?;
    println!("Utility pipe: {}", utility_pipe.to_wkt());

    #[cfg(feature = "geos-backend")]
    {
        // Create a protection zone with larger vertical buffer for excavation safety
        let protection_params = BufferParams3D::anisotropic(2.0, 1.5)
            .with_cap_style(CapStyle3D::Spherical)
            .with_join_style(JoinStyle3D::Round)
            .with_z_interpolation(ZInterpolationStrategy::Linear);

        let protection_zone = buffer_3d_with_params(&utility_pipe, &protection_params)?;
        println!("Protection zone created:");
        println!("  - 2m horizontal buffer");
        println!("  - 1.5m vertical buffer");
        println!("  - Result: {}", protection_zone.geometry_type());
        println!();
    }

    // Section 9: Error Handling
    println!("## Section 9: Error Handling");
    println!("Attempting to create 3D buffer on 2D geometry...\n");

    let point_2d = Geometry::from_wkt("POINT (100 200)")?;
    let result = buffer_3d(&point_2d, 10.0);

    match result {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Expected error: {}", e),
    }
    println!();

    // Section 10: Performance Considerations
    println!("## Section 10: Performance Tips");
    println!("Recommendations for 3D buffer operations:\n");
    println!("1. Use anisotropic buffers when horizontal and vertical scales differ");
    println!("2. Choose cap/join styles based on your use case:");
    println!("   - Spherical: Most accurate but slower");
    println!("   - Cylindrical: Good for buildings and structures");
    println!("   - Flat: Fastest, good for simple cases");
    println!("3. Z interpolation strategies:");
    println!("   - Average: Fastest, uniform Z");
    println!("   - Preserve: Good for maintaining original elevations");
    println!("   - Linear: Balanced approach");
    println!("   - Smooth: Best quality, slowest");
    println!("4. Adjust quadrant_segments for quality vs performance tradeoff");
    println!();

    println!("=== Example Complete ===");
    Ok(())
}
