# oxirs-geosparql

[![Version](https://img.shields.io/badge/version-0.1.0--alpha.3-orange)](https://github.com/cool-japan/oxirs/releases)

GeoSPARQL implementation for spatial data and queries in RDF/SPARQL.

**Status**: Alpha Release (v0.1.0-alpha.3) - Released October 12, 2025

⚠️ **Alpha Software**: APIs may change without notice. Suitable for production alpha testing and internal deployments.

## Overview

`oxirs-geosparql` provides a complete implementation of the OGC GeoSPARQL 1.0/1.1 specification for the OxiRS semantic web platform. It enables spatial queries and operations on RDF knowledge graphs.

## Features

- **GeoSPARQL Vocabulary**: Full support for GeoSPARQL ontology and datatypes
- **WKT/GML Support**: Parse and serialize Well-Known Text (WKT) and GML geometries
- **Simple Features Relations**: All 8 topological predicates (sfEquals, sfDisjoint, sfIntersects, sfTouches, sfCrosses, sfWithin, sfContains, sfOverlaps)
- **Egenhofer Relations**: All 8 relations based on 4-intersection model
- **RCC8 Relations**: All 8 Region Connection Calculus relations
- **Geometric Operations**: Distance, buffer, convex hull, intersection, union, etc.
- **Geometric Properties**: Dimension, SRID, isEmpty, isSimple, etc.
- **Spatial Indexing**: R-tree based spatial index for efficient queries
- **CRS Support**: Coordinate Reference System handling with EPSG codes and transformations

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxirs-geosparql = "0.1.0-beta.1"
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

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::egenhofer;

let polygon1 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
let polygon2 = Geometry::from_wkt("POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))").unwrap();

// Test various Egenhofer relations
let overlap = egenhofer::eh_overlap(&polygon1, &polygon2).unwrap();
assert!(overlap); // Polygons overlap

let meets = egenhofer::eh_meet(&polygon1, &polygon2).unwrap();
// Note: Egenhofer relations require GEOS backend for boundary calculations
```

#### RCC8 Relations (Region Connection Calculus)

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::rcc8;

let polygon1 = Geometry::from_wkt("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))").unwrap();
let polygon2 = Geometry::from_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();

// Test RCC8 relations
let tpp = rcc8::rcc8_tpp(&polygon1, &polygon2).unwrap();
assert!(tpp); // polygon1 is a tangential proper part of polygon2
```

**Requirements for Egenhofer & RCC8:**
- Enable `geos-backend` feature for boundary calculations
- Install GEOS library: `brew install geos` (macOS) or `apt-get install libgeos-dev` (Ubuntu)

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
- ✅ Egenhofer relations (4-intersection model)
  - `geof:ehEquals`, `geof:ehDisjoint`, `geof:ehMeet`, `geof:ehOverlap`
  - `geof:ehCovers`, `geof:ehCoveredBy`, `geof:ehInside`, `geof:ehContains`
- ✅ RCC8 relations (Region Connection Calculus)
  - `geof:rcc8eq`, `geof:rcc8dc`, `geof:rcc8ec`, `geof:rcc8po`
  - `geof:rcc8tpp`, `geof:rcc8tppi`, `geof:rcc8ntpp`, `geof:rcc8ntppi`

### Geometry Extension
- ✅ WKT parsing and serialization
- ✅ GML parsing and serialization (with `gml-support`)
- ✅ Spatial properties (dimension, SRID, isEmpty, isSimple)
- ✅ Buffer operations (pure Rust + GEOS backends)
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
- `geos-backend`: Use GEOS C++ library for advanced operations (requires GEOS installation)
- `rust-buffer`: Pure Rust buffer operations for Polygon/MultiPolygon (no C++ dependencies)
- `proj-support`: Coordinate transformation support
- `parallel`: Parallel processing for large datasets

📖 **See [BUFFER_GUIDE.md](BUFFER_GUIDE.md) for comprehensive buffer operations documentation**

### Buffer Operations

oxirs-geosparql provides **two buffer implementations**:

**1. Pure Rust Buffer** (`rust-buffer` feature):
```toml
[dependencies]
oxirs-geosparql = { version = "0.1.0-beta.1"", features = ["rust-buffer"] }
```

```rust
use oxirs_geosparql::functions::geometric_operations::buffer;

let poly = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").unwrap();
let buffered = buffer(&poly, 2.0).unwrap(); // Pure Rust!
```

**Advantages:**
- ✅ No C++ dependencies
- ✅ Easy cross-compilation
- ✅ Smaller binary size
- ✅ Supports Polygon and MultiPolygon

**Limitations:**
- ⚠️ Only Polygon/MultiPolygon (Point/LineString require GEOS)

**2. GEOS Backend** (`geos-backend` feature):
```toml
[dependencies]
oxirs-geosparql = { version = "0.1.0-beta.1"", features = ["geos-backend"] }
```

**Advantages:**
- ✅ Supports all geometry types
- ✅ Industry-standard quality
- ✅ Advanced cap/join styles

**Requirements:**
- Requires GEOS C++ library installation
  - macOS: `brew install geos`
  - Ubuntu: `sudo apt-get install libgeos-dev`

**Hybrid Strategy:**
The `buffer()` function automatically uses the best available backend:
- Polygon/MultiPolygon → `rust-buffer` (if enabled), else `geos-backend`
- Point/LineString → `geos-backend` (required)

## Architecture

```
oxirs-geosparql/
├── vocabulary/     # GeoSPARQL URIs and terms
├── geometry/       # Geometry types and WKT parser
├── functions/      # Topological and geometric functions
│   ├── simple_features/       # DE-9IM relations
│   ├── geometric_operations/  # Buffer, hull, etc.
│   └── geometric_properties/  # Dimension, SRID, etc.
├── index/          # R-tree spatial indexing
└── error/          # Error types
```

## References

- [OGC GeoSPARQL 1.1 Specification](https://www.ogc.org/standards/geosparql)
- [Simple Features Specification](https://www.ogc.org/standards/sfa)
- [Apache Jena GeoSPARQL](https://jena.apache.org/documentation/geosparql/)

## License

Dual-licensed under MIT OR Apache-2.0.
