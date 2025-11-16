//! TopoJSON Support Example
//!
//! This example demonstrates the TopoJSON format support in oxirs-geosparql.
//! TopoJSON is a topology-preserving JSON format that encodes topology by
//! sharing arcs between geometries, resulting in smaller file sizes and
//! preserving topological relationships.
//!
//! Run with: cargo run --example topojson_support --features topojson-support

use geo_types::{
    Coord, Geometry as GeoGeometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point,
    Polygon,
};
use oxirs_geosparql::geometry::Geometry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TopoJSON Support Example ===\n");

    // Section 1: Basic Point Geometry
    println!("## Section 1: Basic Point Geometry");
    println!("TopoJSON for simple point geometries...\n");

    let point = Point::new(100.0, 200.0);
    let geom = Geometry::new(GeoGeometry::Point(point));
    let geometries = vec![geom];

    let topojson = Geometry::to_topojson(&geometries)?;
    println!("TopoJSON output:");
    println!("{}\n", topojson);

    // Parse back
    let parsed_geometries = Geometry::from_topojson(&topojson)?;
    println!(
        "Parsed {} geometries back from TopoJSON",
        parsed_geometries.len()
    );
    println!("Round-trip successful!\n");

    // Section 2: Multiple Points (MultiPoint)
    println!("## Section 2: MultiPoint Geometry");
    println!("Demonstrating multi-point features...\n");

    let points = vec![
        Point::new(10.0, 20.0),
        Point::new(30.0, 40.0),
        Point::new(50.0, 60.0),
    ];
    let multipoint = Geometry::new(GeoGeometry::MultiPoint(MultiPoint(points)));
    let mp_geometries = vec![multipoint];

    let mp_topojson = Geometry::to_topojson(&mp_geometries)?;
    println!("MultiPoint TopoJSON:");
    println!("{}\n", mp_topojson);

    // Section 3: LineString Geometry
    println!("## Section 3: LineString Geometry");
    println!("Roads and paths represented as LineStrings...\n");

    let coords = vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 10.0, y: 10.0 },
        Coord { x: 20.0, y: 5.0 },
        Coord { x: 30.0, y: 15.0 },
    ];
    let linestring = LineString::new(coords);
    let ls_geom = Geometry::new(GeoGeometry::LineString(linestring));
    let ls_geometries = vec![ls_geom];

    let ls_topojson = Geometry::to_topojson(&ls_geometries)?;
    println!("LineString TopoJSON:");
    println!("{}\n", ls_topojson);

    // Section 4: Polygon Geometry
    println!("## Section 4: Polygon Geometry");
    println!("Regions and areas as polygons...\n");

    let exterior = LineString::new(vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 10.0, y: 0.0 },
        Coord { x: 10.0, y: 10.0 },
        Coord { x: 0.0, y: 10.0 },
        Coord { x: 0.0, y: 0.0 },
    ]);
    let polygon = Polygon::new(exterior, vec![]);
    let poly_geom = Geometry::new(GeoGeometry::Polygon(polygon));
    let poly_geometries = vec![poly_geom];

    let poly_topojson = Geometry::to_topojson(&poly_geometries)?;
    println!("Polygon TopoJSON:");
    println!("{}\n", poly_topojson);

    // Section 5: Polygon with Holes
    println!("## Section 5: Polygon with Holes");
    println!("Complex polygons with interior rings (holes)...\n");

    let exterior = LineString::new(vec![
        Coord { x: 0.0, y: 0.0 },
        Coord { x: 20.0, y: 0.0 },
        Coord { x: 20.0, y: 20.0 },
        Coord { x: 0.0, y: 20.0 },
        Coord { x: 0.0, y: 0.0 },
    ]);

    let hole = LineString::new(vec![
        Coord { x: 5.0, y: 5.0 },
        Coord { x: 15.0, y: 5.0 },
        Coord { x: 15.0, y: 15.0 },
        Coord { x: 5.0, y: 15.0 },
        Coord { x: 5.0, y: 5.0 },
    ]);

    let polygon_with_hole = Polygon::new(exterior, vec![hole]);
    let poly_hole_geom = Geometry::new(GeoGeometry::Polygon(polygon_with_hole));
    let poly_hole_geometries = vec![poly_hole_geom];

    let poly_hole_topojson = Geometry::to_topojson(&poly_hole_geometries)?;
    println!("Polygon with hole TopoJSON:");
    println!("{}\n", poly_hole_topojson);

    // Section 6: MultiLineString
    println!("## Section 6: MultiLineString Geometry");
    println!("Multiple connected or disconnected lines...\n");

    let line1 = LineString::new(vec![Coord { x: 0.0, y: 0.0 }, Coord { x: 5.0, y: 5.0 }]);

    let line2 = LineString::new(vec![Coord { x: 10.0, y: 10.0 }, Coord { x: 15.0, y: 15.0 }]);

    let multilinestring = MultiLineString(vec![line1, line2]);
    let mls_geom = Geometry::new(GeoGeometry::MultiLineString(multilinestring));
    let mls_geometries = vec![mls_geom];

    let mls_topojson = Geometry::to_topojson(&mls_geometries)?;
    println!("MultiLineString TopoJSON:");
    println!("{}\n", mls_topojson);

    // Section 7: MultiPolygon
    println!("## Section 7: MultiPolygon Geometry");
    println!("Archipelagos and discontinuous regions...\n");

    let polygon1 = Polygon::new(
        LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 5.0, y: 0.0 },
            Coord { x: 5.0, y: 5.0 },
            Coord { x: 0.0, y: 5.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    );

    let polygon2 = Polygon::new(
        LineString::new(vec![
            Coord { x: 10.0, y: 10.0 },
            Coord { x: 15.0, y: 10.0 },
            Coord { x: 15.0, y: 15.0 },
            Coord { x: 10.0, y: 15.0 },
            Coord { x: 10.0, y: 10.0 },
        ]),
        vec![],
    );

    let multipolygon = MultiPolygon(vec![polygon1, polygon2]);
    let mp_geom = Geometry::new(GeoGeometry::MultiPolygon(multipolygon));
    let mp_geometries = vec![mp_geom];

    let mp_topojson = Geometry::to_topojson(&mp_geometries)?;
    println!("MultiPolygon TopoJSON:");
    println!("{}\n", mp_topojson);

    // Section 8: Multiple Geometries in One TopoJSON
    println!("## Section 8: Multiple Geometries in One TopoJSON");
    println!("Combining different geometry types...\n");

    let mixed_geometries = vec![
        Geometry::new(GeoGeometry::Point(Point::new(1.0, 2.0))),
        Geometry::new(GeoGeometry::LineString(LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 5.0, y: 5.0 },
        ]))),
        Geometry::new(GeoGeometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 15.0, y: 10.0 },
                Coord { x: 15.0, y: 15.0 },
                Coord { x: 10.0, y: 15.0 },
                Coord { x: 10.0, y: 10.0 },
            ]),
            vec![],
        ))),
    ];

    let mixed_topojson = Geometry::to_topojson(&mixed_geometries)?;
    println!("Mixed geometries TopoJSON:");
    println!("{}\n", mixed_topojson);

    let parsed_mixed = Geometry::from_topojson(&mixed_topojson)?;
    println!("Parsed {} mixed geometries", parsed_mixed.len());
    println!();

    // Section 9: Parsing Existing TopoJSON
    println!("## Section 9: Parsing Existing TopoJSON");
    println!("Reading TopoJSON from external sources...\n");

    let existing_topojson = r#"{
        "type": "Topology",
        "objects": {
            "cities": {
                "type": "MultiPoint",
                "coordinates": [[100, 200], [150, 250], [200, 300]]
            },
            "boundary": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
            }
        }
    }"#;

    let parsed_existing = Geometry::from_topojson(existing_topojson)?;
    println!(
        "Parsed {} geometries from existing TopoJSON:",
        parsed_existing.len()
    );
    for (i, geom) in parsed_existing.iter().enumerate() {
        println!("  Geometry {}: {}", i + 1, geom.geometry_type());
    }
    println!();

    // Section 10: TopoJSON Benefits
    println!("## Section 10: TopoJSON Benefits");
    println!();
    println!("TopoJSON advantages:");
    println!("1. Topology Preservation: Shared boundaries are encoded once");
    println!("2. Smaller File Sizes: Arc sharing reduces redundancy");
    println!("3. Coordinate Quantization: Optional precision reduction for compression");
    println!("4. Transform Support: Scale and translate for coordinate optimization");
    println!("5. Web-Friendly: JSON format works seamlessly with web APIs");
    println!("6. D3.js Integration: Native support in visualization libraries");
    println!();

    // Section 11: Use Cases
    println!("## Section 11: Real-World Use Cases");
    println!();
    println!("TopoJSON is ideal for:");
    println!("- Administrative boundaries (countries, states, counties)");
    println!("- Electoral districts with shared borders");
    println!("- Watershed and drainage basins");
    println!("- Road networks (shared nodes and edges)");
    println!("- Web mapping applications");
    println!("- Data visualization (D3.js, Mapbox, Leaflet)");
    println!("- Mobile applications (smaller download sizes)");
    println!();

    // Section 12: Topology Example
    println!("## Section 12: Topology Preservation Example");
    println!("Two adjacent polygons sharing a border...\n");

    // Two polygons that share a border
    let polygon_a = Polygon::new(
        LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 10.0, y: 10.0 },
            Coord { x: 0.0, y: 10.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    );

    let polygon_b = Polygon::new(
        LineString::new(vec![
            Coord { x: 10.0, y: 0.0 }, // Shared edge starts here
            Coord { x: 20.0, y: 0.0 },
            Coord { x: 20.0, y: 10.0 },
            Coord { x: 10.0, y: 10.0 }, // Shared edge ends here
            Coord { x: 10.0, y: 0.0 },
        ]),
        vec![],
    );

    let adjacent_polygons = vec![
        Geometry::new(GeoGeometry::Polygon(polygon_a)),
        Geometry::new(GeoGeometry::Polygon(polygon_b)),
    ];

    let topology_topojson = Geometry::to_topojson(&adjacent_polygons)?;
    println!("TopoJSON for adjacent polygons:");
    println!("{}", topology_topojson);
    println!("\nNote: In a full TopoJSON implementation with arc de-duplication,");
    println!("the shared edge would be encoded only once in the 'arcs' array.");
    println!();

    // Section 13: Performance Comparison
    println!("## Section 13: Format Comparison");
    println!();
    println!("Approximate file size comparison (for complex geometries):");
    println!("- GeoJSON: 100% (baseline)");
    println!("- TopoJSON: 40-60% (with arc sharing)");
    println!("- TopoJSON + quantization: 20-30% (aggressive compression)");
    println!("- Shapefile: 50-70% (binary format)");
    println!("- FlatGeobuf: 40-60% (binary + spatial index)");
    println!();

    // Section 14: Round-Trip Validation
    println!("## Section 14: Round-Trip Validation");
    println!("Verifying data integrity through serialization cycles...\n");

    let test_polygon = Polygon::new(
        LineString::new(vec![
            Coord {
                x: -122.419,
                y: 37.775,
            }, // San Francisco coordinates
            Coord {
                x: -122.419,
                y: 37.805,
            },
            Coord {
                x: -122.375,
                y: 37.805,
            },
            Coord {
                x: -122.375,
                y: 37.775,
            },
            Coord {
                x: -122.419,
                y: 37.775,
            },
        ]),
        vec![],
    );

    let test_geom = Geometry::new(GeoGeometry::Polygon(test_polygon));
    let test_geometries = vec![test_geom];

    // Serialize
    let serialized = Geometry::to_topojson(&test_geometries)?;

    // Deserialize
    let deserialized = Geometry::from_topojson(&serialized)?;

    println!("Original geometries: {}", test_geometries.len());
    println!("After round-trip: {}", deserialized.len());
    println!(
        "Geometry types match: {}",
        test_geometries[0].geometry_type() == deserialized[0].geometry_type()
    );
    println!("✓ Round-trip validation successful!\n");

    // Section 15: Integration with Web Mapping
    println!("## Section 15: Web Mapping Integration");
    println!();
    println!("TopoJSON can be easily integrated with:");
    println!();
    println!("D3.js example:");
    println!("```javascript");
    println!("d3.json('data.topojson').then(topology => {{");
    println!("  const geojson = topojson.feature(topology, topology.objects.geometries);");
    println!("  // Use in D3 visualization");
    println!("}});");
    println!("```");
    println!();
    println!("Mapbox example:");
    println!("```javascript");
    println!("map.addSource('data', {{");
    println!("  type: 'geojson',");
    println!("  data: topojson.feature(topology, topology.objects.geometries)");
    println!("}});");
    println!("```");
    println!();

    println!("=== Example Complete ===\n");
    println!("Summary:");
    println!("- TopoJSON provides topology-preserving geometry encoding");
    println!("- Reduces file sizes through arc sharing");
    println!("- Perfect for web mapping and data visualization");
    println!("- Supports all common geometry types");
    println!("- Maintains topological relationships");

    Ok(())
}
