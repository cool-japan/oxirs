//! MVT (Mapbox Vector Tiles) support example for oxirs-geosparql
//!
//! This example demonstrates:
//! - Creating MVT tiles with geometries
//! - Working with tile coordinates (z/x/y)
//! - Adding multiple layers to a tile
//! - Adding feature properties
//! - Encoding tiles to binary MVT format
//! - Use cases for web mapping applications
//!
//! Run with: cargo run --example mvt_support --features mvt-support

use oxirs_geosparql::error::Result;

#[cfg(feature = "mvt-support")]
use oxirs_geosparql::geometry::{mvt_parser::MvtTile, Geometry};

#[cfg(feature = "mvt-support")]
use geo_types::{Coord, Geometry as GeoGeometry, LineString, Point, Polygon};

fn main() -> Result<()> {
    println!("=== OxiRS GeoSPARQL MVT (Mapbox Vector Tiles) Support Example ===\n");

    #[cfg(feature = "mvt-support")]
    {
        use std::collections::HashMap;
        use std::env;
        use std::fs::File;
        use std::io::Write;

        let temp_dir = env::temp_dir();

        // 1. Creating a simple MVT tile for San Francisco
        println!("1. CREATING MVT TILE - San Francisco Area:");
        println!("   Zoom: 10, X: 163, Y: 395 (San Francisco tile)");

        let mut tile = MvtTile::new(10, 163, 395);
        let (min_lon, min_lat, max_lon, max_lat) = tile.get_bounds();
        println!(
            "   Bounds: ({:.4}, {:.4}) to ({:.4}, {:.4})",
            min_lon, min_lat, max_lon, max_lat
        );

        // 2. Adding point features (cities/places)
        println!("\n2. ADDING POINT FEATURES - Cities:");

        let sf_point = Geometry::new(GeoGeometry::Point(Point::new(-122.4194, 37.7749)));
        let mut sf_props = HashMap::new();
        sf_props.insert("name".to_string(), "San Francisco".to_string());
        sf_props.insert("population".to_string(), "870000".to_string());
        sf_props.insert("type".to_string(), "city".to_string());

        tile.add_feature("places", sf_point, Some(sf_props))?;
        println!("   ✓ Added San Francisco");

        let oakland_point = Geometry::new(GeoGeometry::Point(Point::new(-122.2711, 37.8044)));
        let mut oakland_props = HashMap::new();
        oakland_props.insert("name".to_string(), "Oakland".to_string());
        oakland_props.insert("population".to_string(), "430000".to_string());
        oakland_props.insert("type".to_string(), "city".to_string());

        tile.add_feature("places", oakland_point, Some(oakland_props))?;
        println!("   ✓ Added Oakland");

        // 3. Adding linestring features (roads/highways)
        println!("\n3. ADDING LINESTRING FEATURES - Major Roads:");

        let highway_101 = Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord {
                x: -122.4194,
                y: 37.7749,
            },
            Coord {
                x: -122.3900,
                y: 37.7900,
            },
            Coord {
                x: -122.3500,
                y: 37.8100,
            },
        ])));

        let mut road_props = HashMap::new();
        road_props.insert("name".to_string(), "Highway 101".to_string());
        road_props.insert("type".to_string(), "highway".to_string());
        road_props.insert("lanes".to_string(), "6".to_string());

        tile.add_feature("roads", highway_101, Some(road_props))?;
        println!("   ✓ Added Highway 101");

        // 4. Adding polygon features (buildings/areas)
        println!("\n4. ADDING POLYGON FEATURES - Parks:");

        let golden_gate_park = Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord {
                    x: -122.5100,
                    y: 37.7694,
                },
                Coord {
                    x: -122.4545,
                    y: 37.7694,
                },
                Coord {
                    x: -122.4545,
                    y: 37.7752,
                },
                Coord {
                    x: -122.5100,
                    y: 37.7752,
                },
                Coord {
                    x: -122.5100,
                    y: 37.7694,
                },
            ]),
            vec![],
        )));

        let mut park_props = HashMap::new();
        park_props.insert("name".to_string(), "Golden Gate Park".to_string());
        park_props.insert("type".to_string(), "park".to_string());
        park_props.insert("area".to_string(), "4.1 km²".to_string());

        tile.add_feature("landuse", golden_gate_park, Some(park_props))?;
        println!("   ✓ Added Golden Gate Park");

        // 5. Encoding the tile
        println!("\n5. ENCODING MVT TILE:");
        let mvt_bytes = tile.encode()?;
        println!("   Encoded size: {} bytes", mvt_bytes.len());
        println!("   Number of layers: {}", tile.layers.len());
        println!("   Layers:");
        for layer in &tile.layers {
            println!("     - {}: {} features", layer.name, layer.features.len());
        }

        // 6. Writing to file
        println!("\n6. WRITING MVT FILE:");
        let mvt_path = temp_dir.join("tile_10_163_395.mvt");
        let mut file = File::create(&mvt_path)?;
        file.write_all(&mvt_bytes)?;
        println!("   ✓ Written to: {}", mvt_path.display());
        println!("   File size: {} bytes", mvt_bytes.len());

        // 7. Creating a tile pyramid (multiple zoom levels)
        println!("\n7. CREATING TILE PYRAMID:");

        // Base tile coordinates at zoom 10
        let base_zoom = 10_u8;
        let base_x = 163_u32;
        let base_y = 395_u32;

        for zoom in 8..=12 {
            let (x, y) = if zoom < base_zoom {
                // Lower zoom - divide by 2^(base_zoom - zoom)
                let shift = base_zoom - zoom;
                (base_x >> shift, base_y >> shift)
            } else {
                // Higher zoom - multiply by 2^(zoom - base_zoom)
                let shift = zoom - base_zoom;
                (base_x << shift, base_y << shift)
            };

            let mut pyramid_tile = MvtTile::new(zoom, x, y);

            let point = Geometry::new(GeoGeometry::Point(Point::new(-122.4194, 37.7749)));
            pyramid_tile.add_feature("places", point, None)?;

            let bytes = pyramid_tile.encode()?;
            println!(
                "   Zoom {}: tile {}/{}/{} ({} bytes)",
                zoom,
                zoom,
                x,
                y,
                bytes.len()
            );
        }

        // 8. Multiple tiles for tiled map
        println!("\n8. CREATING NEIGHBORING TILES:");

        let tiles = [
            (163, 395), // Center
            (164, 395), // East
            (163, 396), // South
            (164, 396), // Southeast
        ];

        for (x, y) in &tiles {
            let mut neighbor = MvtTile::new(10, *x, *y);
            let (min_lon, min_lat, max_lon, max_lat) = neighbor.get_bounds();

            let point = Geometry::new(GeoGeometry::Point(Point::new(
                (min_lon + max_lon) / 2.0,
                (min_lat + max_lat) / 2.0,
            )));
            neighbor.add_feature("grid", point, None)?;

            let bytes = neighbor.encode()?;
            println!("   Tile 10/{}/{}: {} bytes", x, y, bytes.len());
        }

        // 9. MVT advantages and use cases
        println!("\n9. MVT ADVANTAGES:");
        println!("   ✓ Efficient binary format (Protocol Buffers)");
        println!("   ✓ Tile-based system perfect for web maps");
        println!("   ✓ Multiple zoom levels for progressive loading");
        println!("   ✓ Layer-based organization");
        println!("   ✓ Feature properties support");
        println!("   ✓ Wide browser/library support");

        println!("\n10. USE CASES:");
        println!("   • Web mapping applications (Mapbox, MapLibre)");
        println!("   • Tile servers (Martin, Tegola, T-Rex)");
        println!("   • Vector basemaps");
        println!("   • Real-time data visualization");
        println!("   • Mobile mapping apps");
        println!("   • Offline map caching");
        println!("   • Custom map styles and themes");

        println!("\n11. INTEGRATION WITH WEB MAPS:");
        println!("   JavaScript example:");
        println!(r#"   map.addSource('my-tiles', {{"#);
        println!(r#"     type: 'vector',"#);
        println!(r#"     tiles: ['http://localhost:8080/tiles/{{z}}/{{x}}/{{y}}.mvt']"#);
        println!(r#"   }});"#);

        println!("\n12. TYPICAL TILE URL PATTERNS:");
        println!("   • TMS:  /tiles/{{z}}/{{x}}/{{y}}.mvt");
        println!("   • XYZ:  /tiles/{{z}}/{{x}}/{{y}}.pbf");
        println!("   • Named: /{{style}}/{{z}}/{{x}}/{{y}}.vector.pbf");

        // Cleanup
        let _ = std::fs::remove_file(&mvt_path);

        println!("\n✓ All MVT operations completed successfully!");
    }

    #[cfg(not(feature = "mvt-support"))]
    {
        println!("MVT support is not enabled!");
        println!("Run with: cargo run --example mvt_support --features mvt-support");
    }

    Ok(())
}
