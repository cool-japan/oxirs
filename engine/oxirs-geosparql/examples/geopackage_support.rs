//! GeoPackage Support Example
//!
//! This example demonstrates how to use GeoPackage format for reading and writing
//! spatial data. GeoPackage is an OGC standard SQLite-based format widely used
//! in GIS applications.
//!
//! Run with: cargo run --example geopackage_support --features geopackage

use oxirs_geosparql::geometry::geopackage::GeoPackage;
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GeoPackage Support Example ===\n");

    // 1. Create an in-memory GeoPackage database
    println!("1. Creating in-memory GeoPackage database...");
    let mut gpkg = GeoPackage::create_memory()?;
    println!("   ✓ Database created\n");

    // 2. Create a feature table for points
    println!("2. Creating feature table 'cities'...");
    gpkg.create_feature_table(
        "cities", // Table name
        "geom",   // Geometry column name
        "POINT",  // Geometry type
        4326,     // SRID (WGS 84)
        false,    // Has Z coordinates
        false,    // Has M coordinates
    )?;
    println!("   ✓ Feature table created\n");

    // 3. Insert some city locations
    println!("3. Inserting city locations...");
    let cities = vec![
        ("Tokyo", "POINT(139.6917 35.6895)"),
        ("London", "POINT(-0.1278 51.5074)"),
        ("New York", "POINT(-74.0060 40.7128)"),
        ("Paris", "POINT(2.3522 48.8566)"),
        ("Sydney", "POINT(151.2093 -33.8688)"),
    ];

    for (name, wkt) in &cities {
        let point = Geometry::from_wkt(wkt)?;
        let id = gpkg.insert_geometry("cities", "geom", &point)?;
        println!("   ✓ Inserted {} (ID: {})", name, id);
    }
    println!();

    // 4. Query all geometries back
    println!("4. Querying all cities from database...");
    let geometries = gpkg.query_geometries("cities", "geom")?;
    println!("   ✓ Retrieved {} geometries", geometries.len());
    for (i, geom) in geometries.iter().enumerate() {
        let wkt = geom.to_wkt();
        println!("   {} - {}", i + 1, wkt);
    }
    println!();

    // 5. Create a file-based GeoPackage
    println!("5. Creating file-based GeoPackage...");
    let temp_dir = std::env::temp_dir();
    let gpkg_path = temp_dir.join("example_cities.gpkg");
    let mut file_gpkg = GeoPackage::open(&gpkg_path)?;
    println!("   ✓ Created: {}", gpkg_path.display());

    // Create table and insert data
    file_gpkg.create_feature_table("cities", "geom", "POINT", 4326, false, false)?;
    for (_name, wkt) in &cities {
        let point = Geometry::from_wkt(wkt)?;
        file_gpkg.insert_geometry("cities", "geom", &point)?;
    }
    println!("   ✓ Inserted {} cities into file", cities.len());
    println!();

    // 6. List all feature tables
    println!("6. Listing all feature tables...");
    let tables = gpkg.get_feature_tables()?;
    println!("   Found {} feature table(s):", tables.len());
    for table in &tables {
        println!("   - {}", table);
    }
    println!();

    // 7. Create table with line geometries
    println!("7. Creating table for roads (LineString geometries)...");
    gpkg.create_feature_table("roads", "geom", "LINESTRING", 4326, false, false)?;

    let roads = vec![
        "LINESTRING(0 0, 1 1, 2 0)",
        "LINESTRING(2 0, 3 1, 4 0, 5 1)",
        "LINESTRING(-1 -1, -2 0, -1 1)",
    ];

    for road_wkt in &roads {
        let road = Geometry::from_wkt(road_wkt)?;
        gpkg.insert_geometry("roads", "geom", &road)?;
    }
    println!("   ✓ Inserted {} roads", roads.len());
    println!();

    // 8. Create table with polygon geometries
    println!("8. Creating table for parcels (Polygon geometries)...");
    gpkg.create_feature_table("parcels", "geom", "POLYGON", 4326, false, false)?;

    let parcels = vec![
        "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))",
        "POLYGON((20 20, 30 20, 30 30, 20 30, 20 20))",
        "POLYGON((15 5, 25 5, 25 15, 15 15, 15 5), (18 8, 22 8, 22 12, 18 12, 18 8))", // With hole
    ];

    for parcel_wkt in &parcels {
        let parcel = Geometry::from_wkt(parcel_wkt)?;
        gpkg.insert_geometry("parcels", "geom", &parcel)?;
    }
    println!(
        "   ✓ Inserted {} parcels (including one with a hole)",
        parcels.len()
    );
    println!();

    // 9. Query specific table
    println!("9. Querying roads from database...");
    let road_geoms = gpkg.query_geometries("roads", "geom")?;
    println!("   Retrieved {} road geometries:", road_geoms.len());
    for (i, road) in road_geoms.iter().enumerate() {
        println!("   Road {}: {}", i + 1, road.to_wkt());
    }
    println!();

    // 10. Working with CRS information
    println!("10. CRS Information...");
    let point = Geometry::from_wkt("POINT(10 20)")?;
    println!("    Default CRS: {}", point.crs.uri);
    println!("    EPSG Code: {:?}", point.crs.epsg_code());
    println!();

    // 11. GeoPackage metadata
    println!("11. GeoPackage Metadata...");
    println!("    Application ID: 0x47503130 (GP10)");
    println!("    User Version: 10300 (GeoPackage 1.3.0)");
    println!("    Format: SQLite-based OGC standard");
    println!("    Mandatory tables:");
    println!("      - gpkg_contents (content metadata)");
    println!("      - gpkg_spatial_ref_sys (coordinate systems)");
    println!("      - gpkg_geometry_columns (geometry column metadata)");
    println!();

    // 12. Best practices
    println!("12. GeoPackage Best Practices...");
    println!("    ✓ Use standard SRS IDs (4326 for WGS 84, 3857 for Web Mercator)");
    println!("    ✓ Always specify geometry type when creating tables");
    println!("    ✓ Use Z/M flags appropriately for 3D/measured data");
    println!("    ✓ GeoPackage is self-contained (single file deployment)");
    println!("    ✓ Supports transactions for data integrity");
    println!("    ✓ Can include both vector and raster data");
    println!("    ✓ Widely supported by GIS software (QGIS, ArcGIS, etc.)");
    println!();

    // 13. Use cases
    println!("13. Common GeoPackage Use Cases...");
    println!("    • Mobile/offline GIS applications");
    println!("    • Data exchange between different GIS platforms");
    println!("    • Field data collection");
    println!("    • Embedded spatial databases");
    println!("    • Web map tile caching");
    println!("    • Spatial data archiving");
    println!();

    // 14. Performance tips
    println!("14. Performance Tips...");
    println!("    • Use transactions for bulk inserts");
    println!("    • Create spatial indexes for large datasets");
    println!("    • Batch geometry operations when possible");
    println!("    • Use appropriate precision for coordinates");
    println!("    • Consider file size limits (~140TB maximum)");
    println!();

    // Cleanup
    println!("15. Cleanup...");
    std::fs::remove_file(&gpkg_path)?;
    println!("    ✓ Removed temporary file: {}", gpkg_path.display());
    println!();

    println!("=== Example Complete ===");
    println!("\nGeoPackage provides a modern, efficient way to store and");
    println!("exchange geospatial data. It's an excellent choice for:");
    println!("  - Cross-platform compatibility");
    println!("  - Mobile applications");
    println!("  - Single-file deployment");
    println!("  - OGC standards compliance");

    Ok(())
}
