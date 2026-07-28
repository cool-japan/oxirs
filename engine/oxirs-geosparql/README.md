# oxirs-geosparql

[![Version](https://img.shields.io/badge/version-0.4.2-blue)](https://github.com/cool-japan/oxirs/releases)

GeoSPARQL implementation for spatial data and queries in RDF/SPARQL.

**Status**: v0.4.1 - Released 2026-07-26

✅ **Production Ready**: APIs are stable and ready for production deployments. 1,967 tests passing.

## Overview

`oxirs-geosparql` provides a complete implementation of the OGC GeoSPARQL 1.0/1.1 specification for the OxiRS semantic web platform. It enables spatial queries and operations on RDF knowledge graphs.

## Features

- **GeoSPARQL Vocabulary**: Full support for GeoSPARQL ontology and datatypes
- **WKT/GML Support**: Parse and serialize Well-Known Text (WKT) and GML geometries
- **Simple Features Relations**: All 8 topological predicates (sfEquals, sfDisjoint, sfIntersects, sfTouches, sfCrosses, sfWithin, sfContains, sfOverlaps)
- **Egenhofer Relations**: All 8 relations based on the 4-intersection model, all in pure Rust
- **RCC8 Relations**: All 8 Region Connection Calculus relations, all in pure Rust
- **Geometric Operations**: Distance, buffer, convex hull, intersection, union, etc.
- **Geometric Properties**: Dimension, SRID, isEmpty, isSimple, etc.
- **Spatial Indexing**: R-tree based spatial index for efficient queries
- **CRS Support**: Coordinate Reference System handling with EPSG codes and transformations
- **GeoPackage**: Read/write OGC GeoPackage files on a 100% Pure-Rust SQLite-compatible backend (no `libsqlite3`) — see [GeoPackage Support](#geopackage-support) below

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxirs-geosparql = "0.4.2"
```

## Usage

### Basic Geometry Operations

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::simple_features;

// Parse WKT geometries
let point = Geometry::from_wkt("POINT(1.0 2.0)").unwrap();
let polygon = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();

// Test spatial relations
let contains = simple_features::sf_contains(&polygon, &point).unwrap();
assert!(contains);

// Convert back to WKT
let wkt = point.to_wkt();
println!("WKT: {}", wkt); // Output: POINT(1 2)
```

### Topological Relations

#### Egenhofer Relations (4-Intersection Model)

All eight Egenhofer relations run in pure Rust:

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::egenhofer;

let polygon1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
let polygon2 = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))").expect("should succeed");

// Runs in pure Rust — no boundary calculation required
let overlap = egenhofer::eh_overlap(&polygon1, &polygon2).expect("should succeed");
assert!(overlap); // Polygons overlap

// Boundary-dependent relations work too — these used to require a GEOS build
let touching = Geometry::from_wkt("POLYGON((4 0, 8 0, 8 4, 4 4, 4 0))").expect("should succeed");
let meets = egenhofer::eh_meet(&polygon1, &touching).expect("should succeed");
assert!(meets); // The two polygons share only the edge x=4
```

#### RCC8 Relations (Region Connection Calculus)

All eight RCC8 relations run in pure Rust:

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::rcc8;

let polygon1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").expect("should succeed");
let polygon2 = Geometry::from_wkt("POLYGON((5 5, 7 5, 7 7, 5 7, 5 5))").expect("should succeed");

// Runs in pure Rust — no boundary calculation required
let dc = rcc8::rcc8_dc(&polygon1, &polygon2).expect("should succeed");
assert!(dc); // The two regions are disconnected

// Boundary-dependent relations work too — these used to require a GEOS build
let outer = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").expect("should succeed");
let tpp = rcc8::rcc8_tpp(&polygon1, &outer).expect("should succeed");
assert!(tpp); // polygon1 is a proper part of outer, touching at the boundary
```

**Boundary-dependent Egenhofer & RCC8 relations:**

`eh_meet`, `eh_inside`, `eh_contains`, `rcc8_ec`, `rcc8_tpp`, `rcc8_tppi`, `rcc8_ntpp`
and `rcc8_ntppi` need a computed geometric boundary. That used to mean GEOS, whose C FFI
was quarantined into a companion `publish = false` crate, leaving these eight relations
returning `UnsupportedOperation` in the published build. They are Pure Rust now — the
boundary comes from `geo` — so the adapter crate is gone and all eight work out of the
box. Their SPARQL filter functions (`geof:ehMeet`, `geof:ehInside`, `geof:ehContains`,
`geof:rcc8ec`, `geof:rcc8tpp`, `geof:rcc8tppi`, `geof:rcc8ntpp`, `geof:rcc8ntppi`) are
registered by `sparql_integration::get_all_geosparql_functions()` accordingly.

### Spatial Indexing

```rust
use oxirs_geosparql::{Geometry, SpatialIndex};
use geo_types::{Geometry as GeoGeometry, Point};

// Create a spatial index
let index = SpatialIndex::new();

// Insert geometries
let point1 = Geometry::new(GeoGeometry::Point(Point::new(1.0, 1.0)));
let point2 = Geometry::new(GeoGeometry::Point(Point::new(5.0, 5.0)));

index.insert(point1).unwrap();
index.insert(point2).unwrap();

// Query by bounding box
let results = index.query_bbox(0.0, 0.0, 2.0, 2.0);
assert_eq!(results.len(), 1);

// Find nearest geometry
let (nearest, distance) = index.nearest(0.0, 0.0).unwrap();
```

### Coordinate Reference Systems

```rust
use oxirs_geosparql::geometry::{Crs, Geometry};
use geo_types::{Geometry as GeoGeometry, Point};

// Create geometry with EPSG:4326 (WGS84)
let point = Geometry::with_crs(
    GeoGeometry::Point(Point::new(139.7, 35.7)), // Tokyo
    Crs::epsg(4326),
);

// Parse WKT with CRS
let point_with_crs = Geometry::from_wkt(
    "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(139.7 35.7)"
).unwrap();
```

### CRS Transformation

Transform geometries between different coordinate reference systems (requires `proj-support` feature):

```rust
use oxirs_geosparql::geometry::{Crs, Geometry};
use oxirs_geosparql::functions::coordinate_transformation::transform;

// Tokyo in WGS84 (EPSG:4326)
let tokyo_wgs84 = Geometry::from_wkt(
    "<http://www.opengis.net/def/crs/EPSG/0/4326> POINT(139.7 35.7)"
)?;

// Transform to Web Mercator (EPSG:3857) for web mapping
let tokyo_mercator = transform(&tokyo_wgs84, &Crs::epsg(3857))?;

println!("WGS84: {}", tokyo_wgs84.to_wkt());
println!("Web Mercator: {}", tokyo_mercator.to_wkt());
```

**Common transformations:**
- WGS84 → Web Mercator (EPSG:4326 → EPSG:3857): For web mapping
- WGS84 → UTM (EPSG:4326 → EPSG:326XX/327XX): For local surveying
- Any EPSG → Any EPSG: Cross-border analysis

**Requirements:**
- Enable `proj-support` feature in `Cargo.toml`
- Install PROJ library: `brew install proj` (macOS) or `apt-get install libproj-dev` (Ubuntu)

## GeoSPARQL Compliance

This implementation supports:

### Core Features
- ✅ GeoSPARQL vocabulary (`geo:`, `geof:`)
- ✅ Geometry classes (Point, LineString, Polygon, etc.)
- ✅ WKT literals (`geo:wktLiteral`)
- ✅ GML literals (`geo:gmlLiteral`) - with `gml-support` feature

### Topology Vocabulary
- ✅ Simple Features relations (DE-9IM)
  - `geof:sfEquals`, `geof:sfDisjoint`, `geof:sfIntersects`
  - `geof:sfTouches`, `geof:sfCrosses`, `geof:sfWithin`
  - `geof:sfContains`, `geof:sfOverlaps`
- ✅ Egenhofer relations (4-intersection model) — all 8, pure Rust
  - `geof:ehEquals`, `geof:ehDisjoint`, `geof:ehOverlap`, `geof:ehCovers`, `geof:ehCoveredBy`
  - `geof:ehMeet`, `geof:ehInside`, `geof:ehContains`
- ✅ RCC8 relations (Region Connection Calculus) — all 8, pure Rust
  - `geof:rcc8eq`, `geof:rcc8dc`, `geof:rcc8po`
  - `geof:rcc8ec`, `geof:rcc8tpp`, `geof:rcc8tppi`, `geof:rcc8ntpp`, `geof:rcc8ntppi`

### Geometry Extension
- ✅ WKT parsing and serialization
- ✅ GML parsing and serialization (with `gml-support`)
- ✅ Spatial properties (dimension, SRID, isEmpty, isSimple)
- ✅ Buffer operations (pure Rust, every geometry type, full OGC cap/join styles)
- ✅ Boundary operations
- ✅ Geometric set operations (intersection, union, difference, symmetric difference)
- ✅ Convex hull, envelope
- ✅ CRS transformation (with `proj-support`)

### Query Rewrite Extension
- ✅ R-tree spatial indexing
- ⏳ Query optimization - Planned

## SPARQL Integration

GeoSPARQL functions can be used in SPARQL queries:

```sparql
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>

SELECT ?place ?name
WHERE {
  ?place geo:hasGeometry ?geom ;
         rdfs:label ?name .
  ?geom geo:asWKT ?wkt .

  FILTER(geof:sfWithin(?wkt, "POLYGON((...))"^^geo:wktLiteral))
}
```

## Feature Flags

- `wkt-support` (default): WKT parsing and serialization
- `gml-support`: GML (Geography Markup Language) parsing and serialization
- `geojson-support`: GeoJSON support
- `kml-support`: KML (Keyhole Markup Language) parsing and serialization
- `gpx-support`: GPX (GPS Exchange Format) parsing and serialization
- `shapefile-support`: ESRI Shapefile reading and writing (including interior rings/holes for `Polygon`/`MultiPolygon`)
- `geopackage`: OGC GeoPackage reading and writing on the Pure-Rust `oxisql-core`/`oxisql-sqlite-compat` backend (no `libsqlite3`) — see [GeoPackage Support](#geopackage-support)
- `flatgeobuf-support`: FlatGeobuf reading and writing (local files; remote HTTP-range reading is out of scope, see Cargo.toml notes)
- `mvt-support`: Mapbox Vector Tile encoding
- `topojson-support`: TopoJSON support
- `proj-support`: Coordinate transformation support
- `parallel`: Parallel processing for large datasets
- `gpu` / `cuda` / `metal` / `wgpu_backend` / `performance`: GPU-acceleration abstraction flags (Pure Rust; see `Cargo.toml` for current status)

> **GEOS**: not used, and not needed. The operations that once required it — the
> boundary-dependent Egenhofer/RCC8 relations, and buffering for non-polygon types or
> custom cap/join styles — are Pure Rust here, so the crate has a 100% Pure-Rust
> dependency surface with no opt-in. The `geos-backend` feature and the
> `oxirs-geosparql-adapter-geos` quarantine crate that succeeded it are both removed.

📖 **See [BUFFER_GUIDE.md](BUFFER_GUIDE.md) for comprehensive buffer operations documentation**

### Buffer Operations

Buffering is Pure Rust and unconditional — no feature flag, no C++ library. It runs on
`geo::algorithm::buffer`, backed by the `i_overlay` offsetting engine:

```rust
use oxirs_geosparql::functions::geometric_operations::buffer;

let poly = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").expect("should succeed");
let buffered = buffer(&poly, 2.0).expect("should succeed");
```

Every geometry type is supported (Point, LineString, the Multi\* forms,
GeometryCollection), positive distances expand and negative distances erode, and
`buffer_with_params` exposes the full OGC cap/join styles:

```rust
use oxirs_geosparql::functions::geometric_operations::{
    buffer_with_params, BufferParams, CapStyle, JoinStyle,
};

let params = BufferParams {
    cap_style: CapStyle::Square,
    join_style: JoinStyle::Mitre,
    ..BufferParams::default()
};
let line = Geometry::from_wkt("LINESTRING(0 0, 5 0, 5 5)").expect("should succeed");
let buffered = buffer_with_params(&line, 1.0, &params).expect("should succeed");
```

Earlier versions split this across a Polygon-only `rust-buffer` feature and a GEOS
C-library backend (latterly the `oxirs-geosparql-adapter-geos` quarantine crate). Both
are gone, along with the `buffer_rust` function — `buffer` covers all of it.

## GeoPackage Support

Enable the `geopackage` feature to read and write [OGC GeoPackage](http://www.geopackage.org/spec/)
files — SQLite-based vector feature storage. The backend is the Pure-Rust
`oxisql-core`/`oxisql-sqlite-compat` engine (COOLJAPAN Pure Rust Policy v2), so there is
**no `libsqlite3` / C dependency**:

```toml
[dependencies]
oxirs-geosparql = { version = "0.4.2", features = ["geopackage"] }
```

```rust,ignore
use oxirs_geosparql::geometry::geopackage::GeoPackage;

// Open (or create) a GeoPackage file; the required tables and default SRS rows
// are initialized automatically.
let gpkg = GeoPackage::open("data.gpkg")?;

// An in-memory GeoPackage is also available (useful for tests):
let gpkg = GeoPackage::create_memory()?;

// The engine writes in WAL mode. Call `checkpoint()` before handing a
// file-backed GeoPackage to an external GIS reader — this flushes the WAL so the
// GPKG application-id magic bytes land in the main database file.
gpkg.checkpoint()?;
```

`GeoPackage` drives the underlying async `oxisql` engine through an owned
current-thread Tokio runtime, so its public API stays synchronous.

## Architecture

```
oxirs-geosparql/
├── vocabulary/     # GeoSPARQL URIs and terms
├── geometry/       # Geometry types and WKT parser
│   └── geopackage.rs   # OGC GeoPackage read/write (oxisql-core / oxisql-sqlite-compat)
├── functions/      # Topological and geometric functions
│   ├── simple_features/       # DE-9IM relations
│   ├── geometric_operations/  # Buffer, hull, etc.
│   └── geometric_properties/  # Dimension, SRID, etc.
├── index/          # R-tree spatial indexing
└── error/          # Error types
```

No GEOS, and no C dependency of any kind: the boundary-dependent Egenhofer/RCC8
relations and buffering for non-polygon types are all Pure Rust in this crate.

## References

- [OGC GeoSPARQL 1.1 Specification](https://www.ogc.org/standards/geosparql)
- [Simple Features Specification](https://www.ogc.org/standards/sfa)
- [Apache Jena GeoSPARQL](https://jena.apache.org/documentation/geosparql/)

## License

Licensed under Apache-2.0.
