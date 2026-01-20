# OxiRS GeoSPARQL Cookbook

*Last Updated: January 2026*

## Overview

This cookbook provides ready-to-use code recipes for common GeoSPARQL tasks. Copy, paste, and adapt these examples for your projects.

## Table of Contents

1. [Basic Geometry Operations](#basic-geometry-operations)
2. [Spatial Queries](#spatial-queries)
3. [Coordinate Transformations](#coordinate-transformations)
4. [Spatial Analysis](#spatial-analysis)
5. [Data Import/Export](#data-importexport)
6. [Performance Optimization](#performance-optimization)
7. [Real-World Scenarios](#real-world-scenarios)

---

## Basic Geometry Operations

### Recipe 1.1: Create Geometries from WKT

```rust
use oxirs_geosparql::geometry::Geometry;

// Point
let point = Geometry::from_wkt("POINT(10.5 20.3)")?;

// LineString
let line = Geometry::from_wkt("LINESTRING(0 0, 1 1, 2 2, 3 1)")?;

// Polygon (with hole)
let polygon = Geometry::from_wkt(
    "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2))"
)?;

// Multi-geometries
let multipoint = Geometry::from_wkt("MULTIPOINT((0 0), (5 5), (10 10))")?;
let multipolygon = Geometry::from_wkt(
    "MULTIPOLYGON(((0 0, 5 0, 5 5, 0 5, 0 0)), ((10 10, 15 10, 15 15, 10 15, 10 10)))"
)?;
```

### Recipe 1.2: Create 3D Geometries

```rust
// Point with Z coordinate
let point_z = Geometry::from_wkt("POINT Z(10.5 20.3 100.0)")?;

// LineString with Z coordinates
let line_z = Geometry::from_wkt("LINESTRING Z(0 0 10, 1 1 20, 2 2 30)")?;

// Point with measured (M) coordinate
let point_m = Geometry::from_wkt("POINT M(10.5 20.3 42.0)")?;

// Point with Z and M coordinates
let point_zm = Geometry::from_wkt("POINT ZM(10.5 20.3 100.0 42.0)")?;

// Check if geometry is 3D
if point_z.is_3d() {
    println!("This is a 3D geometry!");
}
```

### Recipe 1.3: Create Geometries Programmatically

```rust
use geo_types::{Point, LineString, Polygon, Coord, Geometry as GeoGeometry};
use oxirs_geosparql::geometry::Geometry;

// Point
let point = Geometry::new(GeoGeometry::Point(Point::new(10.0, 20.0)));

// LineString
let coords = vec![
    Coord { x: 0.0, y: 0.0 },
    Coord { x: 1.0, y: 1.0 },
    Coord { x: 2.0, y: 2.0 },
];
let line = Geometry::new(GeoGeometry::LineString(LineString::new(coords)));

// Polygon
let exterior = LineString::new(vec![
    Coord { x: 0.0, y: 0.0 },
    Coord { x: 10.0, y: 0.0 },
    Coord { x: 10.0, y: 10.0 },
    Coord { x: 0.0, y: 10.0 },
    Coord { x: 0.0, y: 0.0 },
]);
let polygon = Geometry::new(GeoGeometry::Polygon(Polygon::new(exterior, vec![])));
```

### Recipe 1.4: Geometric Properties

```rust
use oxirs_geosparql::functions::geometric_properties::*;

let polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;

// Area
let area = area(&polygon)?;
println!("Area: {}", area);  // 100.0

// Perimeter
let perimeter = length(&polygon)?;
println!("Perimeter: {}", perimeter);  // 40.0

// Centroid
let centroid = centroid(&polygon)?;
println!("Centroid: {}", centroid.to_wkt());  // POINT(5 5)

// Bounding box
let bbox = envelope(&polygon)?;
println!("Envelope: {}", bbox.to_wkt());

// Convex hull
let hull = convex_hull(&polygon)?;
```

### Recipe 1.5: Buffer Operations

```rust
use oxirs_geosparql::functions::geometric_operations::buffer;

let point = Geometry::from_wkt("POINT(0 0)")?;

// Create 10-unit buffer around point (creates circle/polygon)
let buffered = buffer(&point, 10.0)?;
println!("Buffer: {}", buffered.to_wkt());

// Negative buffer (shrink polygon)
let polygon = Geometry::from_wkt("POLYGON((0 0, 20 0, 20 20, 0 20, 0 0))")?;
let shrunk = buffer(&polygon, -2.0)?;  // Inset by 2 units
```

---

## Spatial Queries

### Recipe 2.1: Topological Relations

```rust
use oxirs_geosparql::functions::simple_features::*;

let point = Geometry::from_wkt("POINT(5 5)")?;
let polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;

// Check if point is within polygon
if sf_within(&point, &polygon)? {
    println!("Point is inside polygon");
}

// Check if geometries intersect
if sf_intersects(&point, &polygon)? {
    println!("Geometries intersect");
}

// Check if geometries are disjoint
let other_point = Geometry::from_wkt("POINT(20 20)")?;
if sf_disjoint(&other_point, &polygon)? {
    println!("Point is outside polygon");
}

// Contains relationship
if sf_contains(&polygon, &point)? {
    println!("Polygon contains point");
}

// Touches (boundary contact)
let boundary_point = Geometry::from_wkt("POINT(0 5)")?;
if sf_touches(&boundary_point, &polygon)? {
    println!("Point touches polygon boundary");
}
```

### Recipe 2.2: Distance Calculations

```rust
use oxirs_geosparql::functions::geometric_operations::distance;

let p1 = Geometry::from_wkt("POINT(0 0)")?;
let p2 = Geometry::from_wkt("POINT(3 4)")?;

// Calculate distance
let dist = distance(&p1, &p2)?;
println!("Distance: {}", dist);  // 5.0 (Pythagorean theorem)

// 3D distance
let p1_3d = Geometry::from_wkt("POINT Z(0 0 0)")?;
let p2_3d = Geometry::from_wkt("POINT Z(3 4 12)")?;
let dist_3d = oxirs_geosparql::functions::geometric_operations::distance_3d(&p1_3d, &p2_3d)?;
println!("3D Distance: {}", dist_3d);  // 13.0
```

### Recipe 2.3: Spatial Index Queries

```rust
use oxirs_geosparql::index::{SpatialIndex, SpatialIndexTrait};
use oxirs_geosparql::geometry::Geometry;

// Create index
let mut index = SpatialIndex::new();

// Insert geometries
let points = vec![
    Geometry::from_wkt("POINT(0 0)")?,
    Geometry::from_wkt("POINT(1 1)")?,
    Geometry::from_wkt("POINT(2 2)")?,
    Geometry::from_wkt("POINT(10 10)")?,
];

for point in points {
    index.insert(point)?;
}

// Query by bounding box
let bbox = Geometry::from_wkt("POLYGON((0 0, 3 0, 3 3, 0 3, 0 0))")?;
let results = index.query_bbox(&bbox)?;
println!("Found {} geometries in bbox", results.len());

// Query by distance (within radius)
let center = Geometry::from_wkt("POINT(0 0)")?;
let within_distance = index.query_distance(&center, 2.0)?;
println!("Found {} geometries within distance", within_distance.len());

// K-nearest neighbors
let k_nearest = index.query_k_nearest(&center, 3)?;
println!("Found {} nearest neighbors", k_nearest.len());
```

### Recipe 2.4: Advanced Spatial Index (R*-tree)

```rust
use oxirs_geosparql::index::{RStarTree, SpatialIndexTrait};

// Create R*-tree (20-40% faster queries than R-tree)
let mut index = RStarTree::new();

// Bulk load (5-10x faster than individual inserts)
let geometries = load_large_dataset();  // Your data
index.bulk_load(geometries)?;

// Query (benefits from optimized structure)
let bbox = Geometry::from_wkt("POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))")?;
let results = index.query_bbox(&bbox)?;
```

---

## Coordinate Transformations

### Recipe 3.1: Transform WGS84 to Web Mercator

```rust
use oxirs_geosparql::geometry::{Geometry, Crs};

let mut geom = Geometry::from_wkt("POINT(10.0 50.0)")?;  // WGS84 (lon, lat)

// Transform to Web Mercator (EPSG:3857)
let web_mercator = Crs::from_epsg(3857)?;
geom.transform(&web_mercator)?;

println!("Transformed: {}", geom.to_wkt());
```

### Recipe 3.2: Batch Transformation (10x Faster)

```rust
use oxirs_geosparql::functions::coordinate_transformation::transform_batch;
use oxirs_geosparql::geometry::Crs;

let mut geometries = vec![
    Geometry::from_wkt("POINT(10.0 50.0)")?,
    Geometry::from_wkt("POINT(11.0 51.0)")?,
    Geometry::from_wkt("POINT(12.0 52.0)")?,
    // ... thousands more
];

// Batch transform (reuses PROJ context = 10x faster)
let target_crs = Crs::from_epsg(3857)?;
transform_batch(&mut geometries, &target_crs)?;
```

### Recipe 3.3: Parallel Transformation (50x Faster)

```rust
use oxirs_geosparql::functions::coordinate_transformation::transform_batch_parallel;
use oxirs_geosparql::geometry::Crs;

let mut geometries = load_large_dataset();  // 10K+ geometries

// Parallel batch transformation (uses all CPU cores)
let target_crs = Crs::from_epsg(3857)?;
transform_batch_parallel(&mut geometries, &target_crs)?;

// 50x faster than individual transforms on 8-core CPU
```

### Recipe 3.4: Custom CRS

```rust
use oxirs_geosparql::geometry::Crs;

// From EPSG code
let wgs84 = Crs::from_epsg(4326)?;
let utm_zone_33n = Crs::from_epsg(32633)?;

// From PROJ string
let custom_crs = Crs::from_proj_string(
    "+proj=lcc +lat_1=49 +lat_2=46 +lat_0=47.5 +lon_0=13.33333333333333 +x_0=400000 +y_0=400000"
)?;

// From URI
let crs_uri = Crs::from_uri("http://www.opengis.net/def/crs/EPSG/0/4326")?;
```

---

## Spatial Analysis

### Recipe 4.1: Clustering (DBSCAN)

```rust
use oxirs_geosparql::analysis::clustering::{dbscan_clustering, ClusteringConfig};

let points = vec![
    Geometry::from_wkt("POINT(0 0)")?,
    Geometry::from_wkt("POINT(0.5 0.5)")?,
    Geometry::from_wkt("POINT(10 10)")?,
    Geometry::from_wkt("POINT(10.5 10.5)")?,
];

// DBSCAN clustering (epsilon=1.0, min_points=2)
let config = ClusteringConfig {
    epsilon: 1.0,
    min_points: 2,
};
let result = dbscan_clustering(&points, &config)?;

println!("Found {} clusters", result.num_clusters);
for (i, cluster_id) in result.cluster_assignments.iter().enumerate() {
    println!("Point {} -> Cluster {}", i, cluster_id);
}
```

### Recipe 4.2: K-Means Clustering

```rust
use oxirs_geosparql::analysis::clustering::{kmeans_clustering, ClusteringConfig};

let points = load_points();  // Your data

// K-means with 5 clusters
let config = ClusteringConfig {
    epsilon: 0.0,
    min_points: 5,  // Number of clusters
};
let result = kmeans_clustering(&points, &config)?;

println!("Cluster centroids:");
for centroid in result.cluster_centroids {
    println!("  {}", centroid.to_wkt());
}
```

### Recipe 4.3: Voronoi Diagrams

```rust
use oxirs_geosparql::analysis::voronoi::voronoi_diagram;

let points = vec![
    Geometry::from_wkt("POINT(0 0)")?,
    Geometry::from_wkt("POINT(10 0)")?,
    Geometry::from_wkt("POINT(5 8.66)")?,
];

// Create Voronoi diagram
let voronoi_cells = voronoi_diagram(&points)?;

for (i, cell) in voronoi_cells.iter().enumerate() {
    println!("Cell {}: {} vertices", i, cell.vertices.len());
}
```

### Recipe 4.4: Delaunay Triangulation

```rust
use oxirs_geosparql::analysis::triangulation::delaunay_triangulation;

let points = vec![
    Geometry::from_wkt("POINT(0 0)")?,
    Geometry::from_wkt("POINT(10 0)")?,
    Geometry::from_wkt("POINT(5 8.66)")?,
    Geometry::from_wkt("POINT(5 2.89)")?,
];

// Create Delaunay triangulation
let triangles = delaunay_triangulation(&points)?;

for (i, triangle) in triangles.iter().enumerate() {
    println!("Triangle {}: {:?}", i, triangle.vertices);
}
```

### Recipe 4.5: Heatmap Generation

```rust
use oxirs_geosparql::analysis::heatmap::{generate_heatmap, HeatmapConfig, KernelFunction};

let points = vec![
    Geometry::from_wkt("POINT(0 0)")?,
    Geometry::from_wkt("POINT(1 1)")?,
    Geometry::from_wkt("POINT(2 2)")?,
];

// Configure heatmap
let config = HeatmapConfig {
    grid_width: 100,
    grid_height: 100,
    radius: 5.0,
    kernel: KernelFunction::Gaussian,
    normalize: true,
};

// Generate heatmap
let heatmap = generate_heatmap(&points, &config)?;

println!("Heatmap: {}x{} grid", heatmap.width, heatmap.height);
println!("Max intensity: {}", heatmap.max_value);
```

### Recipe 4.6: Spatial Interpolation (IDW)

```rust
use oxirs_geosparql::analysis::interpolation::{
    idw_interpolation, InterpolationConfig, InterpolationMethod
};

// Known points with values
let known_points = vec![
    (Geometry::from_wkt("POINT(0 0)")?, 10.0),
    (Geometry::from_wkt("POINT(10 0)")?, 20.0),
    (Geometry::from_wkt("POINT(5 8.66)")?, 15.0),
];

// Points to interpolate
let query_points = vec![
    Geometry::from_wkt("POINT(5 0)")?,
    Geometry::from_wkt("POINT(5 5)")?,
];

// IDW interpolation
let config = InterpolationConfig {
    method: InterpolationMethod::IDW { power: 2.0 },
};
let interpolated = idw_interpolation(&known_points, &query_points, &config)?;

for (point, value) in query_points.iter().zip(interpolated.iter()) {
    println!("{} -> {}", point.to_wkt(), value);
}
```

---

## Data Import/Export

### Recipe 5.1: Read/Write GeoJSON

```rust
#[cfg(feature = "geojson-support")]
{
    use oxirs_geosparql::geometry::Geometry;

    // Parse GeoJSON
    let geojson_str = r#"{
        "type": "Point",
        "coordinates": [10.0, 20.0]
    }"#;
    let geometry = Geometry::from_geojson(geojson_str)?;

    // Serialize to GeoJSON
    let geojson = geometry.to_geojson()?;
    println!("{}", geojson);
}
```

### Recipe 5.2: Read/Write Shapefile

```rust
#[cfg(feature = "shapefile-support")]
{
    use oxirs_geosparql::geometry::shapefile_parser::{read_shapefile, write_shapefile};

    // Read shapefile
    let geometries = read_shapefile("data/countries.shp")?;
    println!("Loaded {} geometries", geometries.len());

    // Write shapefile
    write_shapefile(&geometries, "output/exported.shp")?;
}
```

### Recipe 5.3: Read/Write GeoPackage

```rust
#[cfg(feature = "geopackage")]
{
    use oxirs_geosparql::geometry::geopackage::GeoPackage;

    // Create GeoPackage
    let mut gpkg = GeoPackage::create_memory()?;
    gpkg.create_feature_table("places", "POINT", 4326)?;

    // Insert geometries
    let point = Geometry::from_wkt("POINT(10 20)")?;
    gpkg.insert_geometry("places", &point, None)?;

    // Query geometries
    let results = gpkg.query_geometries("places")?;
    println!("Found {} geometries", results.len());
}
```

### Recipe 5.4: Read/Write FlatGeobuf

```rust
#[cfg(feature = "flatgeobuf-support")]
{
    use oxirs_geosparql::geometry::flatgeobuf_parser::{
        parse_flatgeobuf_bytes, write_flatgeobuf_to_file
    };
    use std::fs;

    // Read FlatGeobuf
    let data = fs::read("data.fgb")?;
    let geometries = parse_flatgeobuf_bytes(&data)?;

    // Write FlatGeobuf
    write_flatgeobuf_to_file(&geometries, "output.fgb")?;
}
```

### Recipe 5.5: PostGIS EWKB/EWKT

```rust
use oxirs_geosparql::geometry::Geometry;

// Parse EWKB (Extended Well-Known Binary) from PostGIS
let ewkb_bytes = vec![/* EWKB bytes from database */];
let geometry = Geometry::from_ewkb(&ewkb_bytes)?;

// Serialize to EWKB for PostGIS
let ewkb = geometry.to_ewkb()?;

// Parse EWKT (Extended Well-Known Text)
let ewkt = "SRID=4326;POINT(10 20)";
let geometry = Geometry::from_ewkt(ewkt)?;

// Serialize to EWKT
let ewkt_str = geometry.to_ewkt();
```

---

## Performance Optimization

### Recipe 6.1: Use Memory Pool for High-Throughput

```rust
use oxirs_geosparql::geometry::memory_pool::GeometryPool;

// Create pool (pre-allocates memory)
let pool = GeometryPool::with_capacity(10000);

// Allocate from pool (fast, no heap allocation)
let mut point = pool.alloc_point()?;
point.set_x_y(10.0, 20.0);

// Use geometry...

// Return to pool for reuse
pool.return_point(point)?;

// 30% faster for high-throughput workloads
```

### Recipe 6.2: SIMD Distance Calculations

```rust
use oxirs_geosparql::performance::simd::simd_distance_batch;

let target = Geometry::from_wkt("POINT(0 0)")?;
let geometries = load_large_dataset();  // 1000+ geometries

// SIMD-accelerated batch distance (4x faster)
let distances = simd_distance_batch(&target, &geometries)?;
```

### Recipe 6.3: Parallel Processing

```rust
use oxirs_geosparql::performance::parallel::parallel_distance_matrix;

let geometries = load_large_dataset();  // 1000+ geometries

// Parallel distance matrix (8x faster on 8 cores)
let matrix = parallel_distance_matrix(&geometries)?;
```

### Recipe 6.4: GPU Acceleration

```rust
#[cfg(feature = "gpu")]
{
    use oxirs_geosparql::performance::gpu::GpuGeometryContext;

    let mut gpu_ctx = GpuGeometryContext::new()?;
    let geometries = load_massive_dataset();  // 100K+ geometries

    // GPU-accelerated pairwise distances (50x faster)
    let distances = gpu_ctx.pairwise_distance_matrix(&geometries)?;
}
```

### Recipe 6.5: Zero-Copy WKT Parsing

```rust
use oxirs_geosparql::geometry::zero_copy_wkt::{WktArena, ZeroCopyWktParser};

let arena = WktArena::new();
let mut parser = ZeroCopyWktParser::new(&arena);

// Parse without allocations (20% memory reduction)
let geometry = parser.parse("POINT(1 2)")?;
```

---

## Real-World Scenarios

### Scenario 1: Find Nearby Restaurants

```rust
use oxirs_geosparql::index::{SpatialIndex, SpatialIndexTrait};
use oxirs_geosparql::geometry::Geometry;

// Load restaurant locations
let mut index = SpatialIndex::new();
let restaurants = load_restaurants();  // Vec<(String, Geometry)>

for (name, location) in &restaurants {
    index.insert(location.clone())?;
}

// User location
let user_location = Geometry::from_wkt("POINT(10.0 50.0)")?;

// Find restaurants within 500m
let nearby = index.query_distance(&user_location, 500.0)?;

println!("Found {} restaurants within 500m", nearby.len());
```

### Scenario 2: Tile-Based Map Rendering

```rust
#[cfg(feature = "mvt-support")]
{
    use oxirs_geosparql::geometry::mvt_parser::{MvtTile, MvtLayer};
    use oxirs_geosparql::index::{RStarTree, SpatialIndexTrait};

    // Index buildings
    let mut index = RStarTree::new();
    let buildings = load_buildings();
    index.bulk_load(buildings)?;

    // Get tile bounds (z=14, x=8456, y=5467)
    let tile_bbox = calculate_tile_bounds(14, 8456, 5467);

    // Query buildings in tile
    let visible = index.query_bbox(&tile_bbox)?;

    // Create MVT tile
    let mut tile = MvtTile::new(14, 8456, 5467);
    let mut layer = MvtLayer::new("buildings");

    for building in visible {
        layer.add_feature(building, &[])?;
    }

    tile.add_layer(layer);
    let mvt_bytes = tile.encode()?;

    // Serve MVT to client
    serve_tile_response(mvt_bytes)?;
}
```

### Scenario 3: Geocoding with Spatial Join

```rust
use oxirs_geosparql::index::{SpatialIndex, SpatialIndexTrait};
use oxirs_geosparql::functions::simple_features::sf_within;

// Load administrative boundaries (countries, states, cities)
let mut boundary_index = SpatialIndex::new();
let boundaries = load_boundaries();  // Vec<(String, Geometry)>

for (name, boundary) in &boundaries {
    boundary_index.insert(boundary.clone())?;
}

// Geocode a point
let point = Geometry::from_wkt("POINT(10.0 50.0)")?;  // Somewhere in Germany

// Find containing boundaries
let candidates = boundary_index.query_bbox(&point.envelope()?)?;

for candidate in candidates {
    if sf_within(&point, &candidate)? {
        println!("Point is in: {}", get_name(&candidate));
    }
}
```

### Scenario 4: GPS Track Simplification

```rust
use oxirs_geosparql::validation::{simplify_geometry, simplify_geometry_vw};

// Load GPS track (many points)
let gps_track = Geometry::from_wkt(
    "LINESTRING(0 0, 0.01 0.01, 0.02 0.02, ..., 10 10)"  // 1000s of points
)?;

// Douglas-Peucker simplification (tolerance = 0.1)
let simplified = simplify_geometry(&gps_track, 0.1)?;
println!("Reduced from {} to {} points",
    count_coords(&gps_track),
    count_coords(&simplified)
);

// Visvalingam-Whyatt simplification (keep 100 points)
let simplified_vw = simplify_geometry_vw(&gps_track, 100)?;
```

### Scenario 5: Elevation Profile

```rust
use oxirs_geosparql::geometry::Geometry;

// Load GPS track with Z coordinates
let track = Geometry::from_wkt(
    "LINESTRING Z(0 0 100, 1 1 110, 2 2 95, 3 3 120)"
)?;

// Extract elevation profile
if track.is_3d() {
    for i in 0..count_coords(&track) {
        if let Some(z) = track.coord3d.z_at(i) {
            let (x, y) = get_xy_at(&track, i)?;
            println!("Distance: {:.2}m, Elevation: {:.1}m", distance_along_track(i), z);
        }
    }
}
```

### Scenario 6: Batch Coordinate Transformation for Web Map

```rust
use oxirs_geosparql::functions::coordinate_transformation::transform_batch_parallel;
use oxirs_geosparql::geometry::Crs;

// Load features in WGS84
let mut features = load_features_wgs84();  // 10,000 features

// Transform to Web Mercator for web map display
let web_mercator = Crs::from_epsg(3857)?;
transform_batch_parallel(&mut features, &web_mercator)?;

// Serialize to GeoJSON for web client
let geojson = serialize_features_to_geojson(&features)?;
send_to_client(geojson)?;
```

### Scenario 7: Spatial Analysis Pipeline

```rust
use oxirs_geosparql::analysis::clustering::dbscan_clustering;
use oxirs_geosparql::analysis::interpolation::idw_interpolation;
use oxirs_geosparql::geometry::Geometry;

// Step 1: Load sensor readings (temperature)
let sensors = vec![
    (Geometry::from_wkt("POINT(0 0)")?, 20.0),
    (Geometry::from_wkt("POINT(10 0)")?, 25.0),
    (Geometry::from_wkt("POINT(5 8.66)")?, 22.0),
];

// Step 2: Cluster sensors
let sensor_locations: Vec<_> = sensors.iter().map(|(g, _)| g.clone()).collect();
let clusters = dbscan_clustering(&sensor_locations, &config)?;

// Step 3: Interpolate temperature across grid
let grid_points = generate_grid(0.0, 0.0, 10.0, 10.0, 1.0);
let interpolated = idw_interpolation(&sensors, &grid_points, &config)?;

// Step 4: Generate heatmap
let heatmap_config = HeatmapConfig { /* ... */ };
let heatmap = generate_heatmap(&grid_points, &heatmap_config)?;

// Visualize or export results
export_heatmap_as_geotiff(&heatmap, "temperature_map.tif")?;
```

---

## Error Handling Best Practices

```rust
use oxirs_geosparql::error::{GeoSparqlError, Result};

fn process_geometry(wkt: &str) -> Result<f64> {
    // Parse geometry (may fail)
    let geom = Geometry::from_wkt(wkt)
        .map_err(|e| GeoSparqlError::ParseError(format!("Invalid WKT: {}", e)))?;

    // Validate geometry
    let validation = oxirs_geosparql::validation::validate_geometry(&geom)?;
    if !validation.is_valid {
        return Err(GeoSparqlError::GeometryOperationFailed(
            format!("Invalid geometry: {:?}", validation.errors)
        ));
    }

    // Calculate area
    let area = oxirs_geosparql::functions::geometric_properties::area(&geom)?;

    Ok(area)
}

// Usage
match process_geometry("POINT(10 20)") {
    Ok(area) => println!("Area: {}", area),
    Err(GeoSparqlError::ParseError(msg)) => eprintln!("Parse error: {}", msg),
    Err(e) => eprintln!("Error: {}", e),
}
```

---

## Testing Recipes

### Property-Based Testing

```rust
#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use oxirs_geosparql::geometry::Geometry;
    use oxirs_geosparql::functions::geometric_operations::distance;

    proptest! {
        #[test]
        fn distance_is_symmetric(x1 in -180.0..180.0, y1 in -90.0..90.0,
                                  x2 in -180.0..180.0, y2 in -90.0..90.0) {
            let p1 = Geometry::from_wkt(&format!("POINT({} {})", x1, y1)).unwrap();
            let p2 = Geometry::from_wkt(&format!("POINT({} {})", x2, y2)).unwrap();

            let d12 = distance(&p1, &p2).unwrap();
            let d21 = distance(&p2, &p1).unwrap();

            prop_assert!((d12 - d21).abs() < 1e-10, "Distance should be symmetric");
        }
    }
}
```

---

## Further Reading

- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)
- [API Documentation](https://docs.rs/oxirs-geosparql)
- [GeoSPARQL Specification](http://www.opengis.net/doc/IS/geosparql/1.1)

---

*This cookbook is maintained by the OxiRS team. Contributions and recipe suggestions welcome!*
