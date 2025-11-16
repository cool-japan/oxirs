//! Shapefile Format Support Example
//!
//! This example demonstrates the shapefile reading capabilities
//! of the oxirs-geosparql crate. Shapefiles are the most widely used vector
//! data format in GIS applications.
//!
//! Run with: cargo run --example shapefile_support --features shapefile-support
//!
//! Note: This example demonstrates reading shapefiles. Writing support is planned for future releases.

#[cfg(feature = "shapefile-support")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OxiRS GeoSPARQL: Shapefile Support Demo ===\n");

    // ========================================
    // 1. What is a Shapefile?
    // ========================================
    println!("1. Shapefile Format Overview");
    println!("   Shapefiles are the de facto standard for GIS vector data.");
    println!("   They consist of multiple files:");
    println!("   • .shp (geometry) - Required");
    println!("   • .shx (index) - Required");
    println!("   • .dbf (attributes) - Required");
    println!("   • .prj (CRS) - Optional but recommended");
    println!("   • .cpg (encoding) - Optional");
    println!();

    // ========================================
    // 2. Reading Shapefiles
    // ========================================
    println!("2. Reading Shapefiles");
    println!("   To read a shapefile, simply provide the path to the .shp file.");
    println!("   The other associated files (.shx, .dbf, .prj) will be");
    println!("   automatically located in the same directory.");
    println!();
    println!("   Example code:");
    println!("   ```rust");
    println!("   use oxirs_geosparql::geometry::Geometry;");
    println!("   ");
    println!("   // Read all geometries from a shapefile");
    println!("   let geometries = Geometry::from_shapefile(\"cities.shp\")?;");
    println!("   ");
    println!("   for (i, geom) in geometries.iter().enumerate() {{");
    println!("       println!(\"Geometry {{}}: {{}}\", i+1, geom.to_wkt());");
    println!("   }}");
    println!("   ```");
    println!();

    // ========================================
    // 3. Supported Geometry Types
    // ========================================
    println!("3. Supported Geometry Types");
    println!("   The shapefile parser supports all common geometry types:");
    println!("   ");
    println!("   • Point / PointM / PointZ");
    println!("   • MultiPoint / MultiPointM / MultiPointZ");
    println!("   • PolyLine / PolyLineM / PolyLineZ (converted to LineString)");
    println!("   • Polygon / PolygonM / PolygonZ (including holes)");
    println!("   ");
    println!("   Note: M and Z coordinates are currently dropped during conversion");
    println!("         as geo-types doesn't natively support 3D coordinates.");
    println!();

    // ========================================
    // 4. CRS Handling
    // ========================================
    println!("4. Coordinate Reference System (CRS) Handling");
    println!("   CRS information is read from the .prj file (if present).");
    println!("   The .prj file contains the CRS in WKT (Well-Known Text) format.");
    println!("   ");
    println!("   Supported CRS detection:");
    println!("   • EPSG codes from AUTHORITY fields");
    println!("   • Common CRS names (WGS 84, Web Mercator)");
    println!("   • Defaults to WGS 84 (EPSG:4326) if no .prj file exists");
    println!();

    // ========================================
    // 5. Example Usage Patterns
    // ========================================
    println!("5. Common Usage Patterns");
    println!();
    println!("   Pattern 1: Read and convert to another format");
    println!("   ```rust");
    println!("   let geometries = Geometry::from_shapefile(\"input.shp\")?;");
    println!("   for geom in geometries {{");
    println!("       let geojson = geom.to_geojson()?;  // Convert to GeoJSON");
    println!("       let wkt = geom.to_wkt();            // Convert to WKT");
    println!("       let kml = geom.to_kml()?;           // Convert to KML");
    println!("   }}");
    println!("   ```");
    println!();
    println!("   Pattern 2: Spatial analysis on shapefile data");
    println!("   ```rust");
    println!("   let geometries = Geometry::from_shapefile(\"parcels.shp\")?;");
    println!("   for geom in geometries {{");
    println!("       let area = geom.area();             // Calculate area");
    println!("       let centroid = geom.centroid();     // Find centroid");
    println!("       let buffer = geom.buffer(100.0)?;   // Create buffer");
    println!("   }}");
    println!("   ```");
    println!();
    println!("   Pattern 3: Filter and process");
    println!("   ```rust");
    println!("   let geometries = Geometry::from_shapefile(\"cities.shp\")?;");
    println!("   let large_cities: Vec<_> = geometries");
    println!("       .into_iter()");
    println!("       .filter(|g| g.area() > 1000.0)  // Filter by area");
    println!("       .collect();");
    println!("   ```");
    println!();

    // ========================================
    // 6. Integration with Other Formats
    // ========================================
    println!("6. Integration with Other Formats");
    println!("   Shapefile geometries can be converted to/from other formats:");
    println!("   ");
    println!("   Shapefile → WKT:      geom.to_wkt()");
    #[cfg(feature = "geojson-support")]
    println!("   Shapefile → GeoJSON:  geom.to_geojson()");
    #[cfg(feature = "kml-support")]
    println!("   Shapefile → KML:      geom.to_kml()");
    #[cfg(feature = "gml-support")]
    println!("   Shapefile → GML:      geom.to_gml()");
    println!();

    // ========================================
    // 7. Performance Considerations
    // ========================================
    println!("7. Performance Considerations");
    println!("   • Shapefiles are read sequentially from disk");
    println!("   • The .shx index file is used for faster access when available");
    println!("   • Large shapefiles (>1GB) may take time to load completely");
    println!("   • Consider streaming processing for very large files");
    println!("   • CRS transformation can be performed after loading if needed");
    println!();

    // ========================================
    // 8. Common Use Cases
    // ========================================
    println!("8. Common Use Cases");
    println!();
    println!("   Use Case 1: GIS Data Import");
    println!("   Import shapefile data from GIS software (ArcGIS, QGIS) for analysis");
    println!("   in Rust applications.");
    println!();
    println!("   Use Case 2: Geospatial ETL");
    println!("   Read shapefiles, transform geometries, and write to database");
    println!("   (PostGIS, SpatiaLite) or other formats (GeoJSON, GeoPackage).");
    println!();
    println!("   Use Case 3: Legacy Data Migration");
    println!("   Convert legacy shapefile datasets to modern formats like");
    println!("   GeoJSON or GeoPackage for web applications.");
    println!();
    println!("   Use Case 4: Spatial Analysis Pipelines");
    println!("   Read shapefile data, perform spatial operations (buffer, union,");
    println!("   intersection), and export results.");
    println!();

    // ========================================
    // 9. Limitations
    // ========================================
    println!("9. Current Limitations");
    println!("   • Writing shapefiles is not yet implemented (planned for future)");
    println!("   • Attribute data (.dbf) is not exposed in the current API");
    println!("   • 3D coordinates (Z) are dropped during conversion");
    println!("   • Measured coordinates (M) are dropped during conversion");
    println!("   • Multipatch geometries are not supported");
    println!();

    // ========================================
    // 10. Future Enhancements
    // ========================================
    println!("10. Future Enhancements");
    println!("   Planned features for upcoming releases:");
    println!("   • Writing shapefile support");
    println!("   • Attribute data access (.dbf reading/writing)");
    println!("   • 3D geometry support (PointZ, PolyLineZ, PolygonZ)");
    println!("   • Measured coordinate support (PointM, PolyLineM, PolygonM)");
    println!("   • Streaming API for large files");
    println!("   • Direct shapefile-to-database import");
    println!();

    // ========================================
    // Summary
    // ========================================
    println!("=== Summary ===");
    println!("Shapefiles are the most widely used GIS vector data format.");
    println!();
    println!("Key Features:");
    println!("  • Industry standard format (ESRI)");
    println!("  • Supported by all major GIS software");
    println!("  • Multi-file format (.shp, .shx, .dbf, .prj)");
    println!("  • Includes geometry and attribute data");
    println!("  • CRS information via .prj file");
    println!();
    println!("Usage in OxiRS (Current v0.1.0):");
    println!("  • Enable with feature: --features shapefile-support");
    println!("  • Read: Geometry::from_shapefile(path)");
    println!("  • Write: Coming in future release");
    println!("  • Attributes: Coming in future release");
    println!();
    println!("Typical Applications:");
    println!("  • GIS data import/export");
    println!("  • Geospatial data pipelines");
    println!("  • Government open data processing");
    println!("  • Legacy system integration");
    println!("  • Desktop GIS software integration");
    println!();
    println!("Alternative Formats:");
    println!("  • GeoJSON - Web-friendly, single file");
    println!("  • GeoPackage - Modern, SQLite-based");
    println!("  • FlatGeobuf - Cloud-optimized");
    println!("  • GML - XML-based standard");

    Ok(())
}

#[cfg(not(feature = "shapefile-support"))]
fn main() {
    eprintln!("This example requires the 'shapefile-support' feature.");
    eprintln!("Run with: cargo run --example shapefile_support --features shapefile-support");
    std::process::exit(1);
}
