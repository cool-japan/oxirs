# OxiRS GeoSPARQL Architecture

*Last Updated: January 2026*

## Overview

This document describes the architecture and design of `oxirs-geosparql`, a high-performance Rust implementation of the OGC GeoSPARQL 1.1 standard. The crate provides spatial data structures, topological operations, and advanced spatial analysis capabilities with a focus on performance and correctness.

## Table of Contents

1. [Design Principles](#design-principles)
2. [Module Organization](#module-organization)
3. [Data Structures](#data-structures)
4. [Algorithms and Operations](#algorithms-and-operations)
5. [Performance Architecture](#performance-architecture)
6. [Integration Points](#integration-points)
7. [Error Handling](#error-handling)
8. [Testing Strategy](#testing-strategy)

---

## Design Principles

### 1. Zero-Cost Abstractions

oxirs-geosparql is designed to provide high-level abstractions without runtime overhead:

- **Trait-based polymorphism** with monomorphization (no vtables)
- **Generic programming** with compile-time specialization
- **Zero-copy parsing** where possible
- **Memory pooling** for frequently allocated objects

### 2. Correctness First

Spatial operations are mathematically complex. We prioritize correctness:

- **Property-based testing** verifies mathematical invariants
- **OGC compliance testing** ensures standard conformance
- **Floating-point precision** handling with configurable tolerances
- **Validation** at API boundaries

### 3. Performance by Default

Performance optimizations are built-in, not opt-in:

- **SIMD acceleration** for distance calculations (auto-vectorization)
- **Parallel processing** with rayon for batch operations
- **Spatial indexing** for O(log n) spatial queries
- **Memory efficiency** through arena allocation and zero-copy

### 4. Modular Design

Each component is independently useful:

- **Geometry operations** work without spatial indexing
- **Serialization formats** are feature-gated
- **Performance features** can be selectively enabled
- **Integration with SciRS2** for scientific computing

---

## Module Organization

```
oxirs-geosparql/
├── src/
│   ├── lib.rs                    # Public API surface
│   ├── error.rs                  # Error types and Result alias
│   ├── vocabulary.rs             # GeoSPARQL URIs and constants
│   │
│   ├── geometry/                 # Geometry data structures
│   │   ├── mod.rs               # Core Geometry struct
│   │   ├── coord3d.rs           # 3D coordinate storage (Z/M)
│   │   ├── wkt_parser.rs        # WKT parsing and serialization
│   │   ├── geojson_parser.rs    # GeoJSON support
│   │   ├── gml_parser.rs        # GML 3.1.1/3.2.1 support
│   │   ├── kml_parser.rs        # KML (Google Earth) support
│   │   ├── gpx_parser.rs        # GPX (GPS Exchange) support
│   │   ├── shapefile_parser.rs  # ESRI Shapefile support
│   │   ├── ewkb_parser.rs       # PostGIS EWKB support
│   │   ├── ewkt_parser.rs       # PostGIS EWKT support
│   │   ├── geopackage.rs        # OGC GeoPackage support
│   │   ├── flatgeobuf_parser.rs # FlatGeobuf cloud-native format
│   │   ├── mvt_parser.rs        # Mapbox Vector Tiles
│   │   ├── topojson_parser.rs   # TopoJSON support
│   │   ├── zero_copy_wkt.rs     # Zero-copy WKT parsing
│   │   ├── rdf_serialization.rs # RDF (Turtle, N-Triples) support
│   │   └── memory_pool.rs       # Geometry memory pooling
│   │
│   ├── functions/               # GeoSPARQL functions
│   │   ├── mod.rs               # Function registry
│   │   ├── simple_features.rs   # 8 SF topological relations
│   │   ├── egenhofer.rs         # 8 Egenhofer relations
│   │   ├── rcc8.rs              # 8 RCC8 relations
│   │   ├── geometric_properties.rs  # Area, length, centroid, etc.
│   │   ├── geometric_operations.rs  # Distance, buffer, set ops
│   │   ├── coordinate_transformation.rs  # CRS transformations
│   │   ├── topological_3d.rs    # 3D topological relations
│   │   ├── buffer_3d.rs         # 3D buffer operations
│   │   ├── transformation_cache.rs  # CRS transformation caching
│   │   └── bbox_utils.rs        # Bounding box utilities
│   │
│   ├── index/                   # Spatial indexing
│   │   ├── mod.rs               # Default R-tree implementation
│   │   ├── spatial_index_trait.rs   # Common index interface
│   │   ├── r_star_tree.rs       # R*-tree (20-40% faster queries)
│   │   ├── hilbert_rtree.rs     # Hilbert R-tree (bulk loading)
│   │   ├── spatial_hash.rs      # Spatial hash index (O(1) insert)
│   │   ├── grid_index.rs        # Regular grid partitioning
│   │   ├── quadtree.rs          # Quadtree for 2D data
│   │   └── kdtree.rs            # K-d tree for point clouds
│   │
│   ├── analysis/                # Spatial analysis algorithms
│   │   ├── mod.rs               # Analysis module exports
│   │   ├── clustering.rs        # DBSCAN, K-means clustering
│   │   ├── voronoi.rs           # Voronoi diagrams
│   │   ├── triangulation.rs     # Delaunay triangulation
│   │   ├── interpolation.rs     # IDW, Kriging interpolation
│   │   ├── statistics.rs        # Moran's I, Getis-Ord
│   │   ├── heatmap.rs           # Kernel density estimation
│   │   └── network.rs           # Network analysis (Dijkstra, A*)
│   │
│   ├── performance/             # Performance optimizations
│   │   ├── mod.rs               # Performance module exports
│   │   ├── simd.rs              # SIMD-accelerated operations
│   │   ├── parallel.rs          # Parallel batch processing
│   │   ├── batch.rs             # Auto-optimizing batch processor
│   │   └── gpu.rs               # GPU acceleration (CUDA/Metal)
│   │
│   ├── validation.rs            # Geometry validation and repair
│   ├── sparql_integration.rs   # SPARQL function metadata
│   │
│   └── bin/
│       └── compare_benchmarks.rs  # Benchmark comparison tool
│
├── tests/                       # Integration tests
│   ├── spatial_queries.rs       # Spatial query scenarios
│   ├── property_tests.rs        # Property-based tests
│   ├── stress_tests.rs          # Large dataset tests
│   ├── ogc_conformance.rs       # OGC GeoSPARQL conformance
│   └── real_world_datasets.rs   # OpenStreetMap scenarios
│
├── benches/                     # Performance benchmarks
│   ├── spatial_operations.rs    # Geometric operation benchmarks
│   ├── spatial_index.rs         # Indexing benchmarks
│   ├── buffer_performance.rs    # Buffer operation benchmarks
│   ├── performance_comparison.rs # Cross-feature comparison
│   └── benchmark_tracking.rs    # Historical tracking
│
├── examples/                    # Example programs
│   └── *.rs                     # 20+ example programs
│
└── docs/                        # Documentation
    ├── PERFORMANCE_TUNING.md    # Optimization guide
    ├── COOKBOOK.md              # Recipe collection
    ├── MIGRATION_FROM_JENA.md   # Jena migration guide
    └── ARCHITECTURE.md          # This document
```

---

## Data Structures

### Core Geometry Structure

```rust
pub struct Geometry {
    /// Underlying 2D geometry from geo_types
    pub geom: geo_types::Geometry<f64>,

    /// 3D coordinates (Z and M values stored separately)
    pub coord3d: Coord3D,

    /// Coordinate Reference System
    pub crs: Crs,
}
```

**Design Rationale:**
- `geo_types::Geometry` provides battle-tested 2D geometry operations
- `Coord3D` extends to 3D without modifying geo_types (maintains compatibility)
- `Crs` enables multi-CRS operations with validation

### 3D Coordinate Storage

```rust
pub struct Coord3D {
    /// Dimension of coordinates
    pub dimension: CoordDim,

    /// Z-coordinate values (elevation/altitude)
    pub z_coords: Option<ZCoords>,

    /// M-coordinate values (measured/time)
    pub m_coords: Option<MCoords>,
}

pub enum CoordDim {
    XY,    // 2D
    XYZ,   // 3D with Z
    XYM,   // 2D with M
    XYZM,  // 3D with Z and M
}
```

**Design Rationale:**
- Separates Z/M from XY to avoid modifying geo_types
- `Option<>` makes 3D optional (zero overhead for 2D geometries)
- Supports all coordinate dimension combinations

### Spatial Index Trait

```rust
pub trait SpatialIndexTrait {
    fn insert(&mut self, geometry: Geometry) -> Result<()>;
    fn bulk_load(&mut self, geometries: Vec<Geometry>) -> Result<()>;
    fn remove(&mut self, geometry: &Geometry) -> Result<bool>;

    fn query_bbox(&self, bbox: &Geometry) -> Result<Vec<Geometry>>;
    fn query_distance(&self, center: &Geometry, radius: f64) -> Result<Vec<Geometry>>;
    fn query_k_nearest(&self, point: &Geometry, k: usize) -> Result<Vec<Geometry>>;
}
```

**Design Rationale:**
- Common interface for all spatial indexes
- Enables runtime index selection
- Bulk loading optimization for faster initialization

---

## Algorithms and Operations

### Topological Relations

Three relation families are implemented:

#### 1. Simple Features (OGC SF)

Based on 9-intersection model (DE-9IM):

```
         Interior  Boundary  Exterior
         --------------------------------
Other   | I ∩ I'  | I ∩ B'  | I ∩ E'
Geom    | B ∩ I'  | B ∩ B'  | B ∩ E'
         | E ∩ I'  | E ∩ B'  | E ∩ E'
```

**8 Relations:**
- `sfEquals`: Geometries are spatially equal
- `sfDisjoint`: No intersection
- `sfIntersects`: Any intersection
- `sfTouches`: Boundary contact only
- `sfCrosses`: Intersect but don't contain
- `sfWithin`: Fully contained
- `sfContains`: Fully contains
- `sfOverlaps`: Partial overlap

#### 2. Egenhofer (8 relations)

Refinement of DE-9IM for qualitative reasoning:
- `ehEquals`, `ehDisjoint`, `ehMeet`, `ehOverlap`
- `ehCovers`, `ehCoveredBy`, `ehInside`, `ehContains`

#### 3. RCC8 (Region Connection Calculus)

Topology for region-based reasoning:
- `rcc8eq` (equal), `rcc8dc` (disconnected), `rcc8ec` (externally connected)
- `rcc8po` (partially overlapping)
- `rcc8tpp`/`rcc8tppi` (tangential proper part)
- `rcc8ntpp`/`rcc8ntppi` (non-tangential proper part)

### Geometric Operations

#### Distance Calculation

```
Algorithm: Minimum Euclidean Distance
Complexity: O(n×m) for n and m vertices
Optimization: SIMD vectorization for point clouds
```

**Implementation:**
1. Point-to-Point: Direct Euclidean formula
2. Point-to-LineString: Perpendicular distance to segments
3. Point-to-Polygon: Distance to boundary or 0 if inside
4. Complex: Decompose into primitive operations

#### Buffer Operation

```
Algorithm: Parallel offset curves with rounding
Complexity: O(n log n) for n vertices
Optimization: Dual backend (GEOS + pure Rust)
```

**Implementation:**
1. Compute offset curves at distance d
2. Round corners based on cap/join style
3. Union resulting polygons
4. Handle degenerate cases (negative buffer)

### Spatial Indexing

#### R*-tree Structure

```
Tree Node:
├── Min Bounding Rectangle (MBR)
├── Children (nodes or geometries)
└── Level (0 = leaf, >0 = internal)

Insertion Algorithm:
1. ChooseSubtree: Minimize MBR enlargement
2. SplitNode: R*-tree quadratic split
3. AdjustTree: Propagate changes upward
```

**Query Complexity:**
- Insert: O(log n)
- Query: O(log n + k) where k = results
- Delete: O(log n)

#### Hilbert R-tree

Uses space-filling Hilbert curve for better locality:

```
1. Map geometries to Hilbert curve positions
2. Sort by Hilbert value
3. Build tree bottom-up (bulk loading)
4. Better cache locality = faster queries
```

---

## Performance Architecture

### Optimization Hierarchy

```
Level 1: Algorithm Selection
├── Choose optimal algorithm for data characteristics
└── Example: K-d tree for point clouds, R*-tree for mixed

Level 2: Vectorization (SIMD)
├── Auto-vectorization with AVX2/SSE4.2
├── 4x speedup for distance calculations
└── Uses scirs2-core SIMD primitives

Level 3: Parallelization
├── Data parallelism with rayon
├── 8x speedup on 8-core CPU
└── Automatic work stealing

Level 4: GPU Acceleration
├── CUDA/Metal/WGPU backends
├── 50-100x speedup for large datasets
└── Automatic CPU fallback

Level 5: Memory Optimization
├── Memory pooling for allocations
├── Zero-copy parsing
└── Arena allocation for temporary objects
```

### Performance Monitoring

Built-in profiling with SciRS2:

```rust
use scirs2_core::profiling::Profiler;
use scirs2_core::metrics::{Counter, Timer};

// Profiling
let profiler = Profiler::new();
profiler.start("spatial_query");
// ... operation ...
profiler.stop("spatial_query");

// Metrics (Prometheus-compatible)
let counter = Counter::new("geosparql_queries_total");
counter.inc();

let timer = Timer::new("geosparql_query_duration_seconds");
let _guard = timer.start();
// Timer automatically records on drop
```

---

## Integration Points

### 1. OxiRS Ecosystem

```
oxirs-core (RDF store)
    ↓
oxirs-arq (SPARQL engine)
    ↓ (registers GeoSPARQL functions)
oxirs-geosparql
    ↓ (provides geometry operations)
oxirs-fuseki (HTTP endpoint)
```

**Function Registration:**

```rust
// In oxirs-arq
use oxirs_geosparql::sparql_integration::get_all_geosparql_functions;

let functions = get_all_geosparql_functions();
for func in functions {
    sparql_engine.register_function(func.uri, func.implementation);
}
```

### 2. SciRS2 Integration

```
scirs2-core: Array operations, SIMD, parallel, GPU
    ↓
scirs2-linalg: Matrix operations for embeddings
    ↓
scirs2-graph: Graph algorithms for network analysis
    ↓
scirs2-stats: Statistical spatial analysis
```

**Usage:**

```rust
use scirs2_core::ndarray_ext::Array2;
use scirs2_core::simd::simd_dot_f32_ultra;
use scirs2_graph::shortest_path::dijkstra;
```

### 3. External Libraries

- **geo_types:** Core 2D geometry types
- **geo:** Geometric algorithms (area, length, etc.)
- **rstar:** Spatial indexing
- **spade:** Delaunay/Voronoi
- **proj:** CRS transformations
- **geos (optional):** Robust geometric operations

---

## Error Handling

### Error Type Hierarchy

```rust
pub enum GeoSparqlError {
    /// WKT/WKB parsing errors
    ParseError(String),

    /// Invalid geometry (self-intersection, etc.)
    InvalidGeometry(String),

    /// CRS incompatibility
    CrsIncompatible(String),

    /// Unsupported operation
    UnsupportedOperation(String),

    /// Geometric operation failed
    GeometryOperationFailed(String),

    /// I/O errors
    IoError(std::io::Error),
}

pub type Result<T> = std::result::Result<T, GeoSparqlError>;
```

### Error Propagation

```rust
// At API boundaries: Validate and convert errors
pub fn from_wkt(wkt: &str) -> Result<Geometry> {
    if wkt.is_empty() {
        return Err(GeoSparqlError::ParseError("Empty WKT".to_string()));
    }

    let parsed = wkt::Wkt::from_str(wkt)
        .map_err(|e| GeoSparqlError::ParseError(format!("Invalid WKT: {}", e)))?;

    convert_wkt_to_geometry(&parsed)
}
```

---

## Testing Strategy

### Test Pyramid

```
                    ┌─────────┐
                    │ E2E (11)│ Real-world scenarios
                    └─────────┘
                  ┌───────────────┐
                  │Integration(80)│ OGC conformance
                  └───────────────┘
              ┌───────────────────────┐
              │  Property-based (17)  │ Mathematical invariants
              └───────────────────────┘
          ┌─────────────────────────────────┐
          │       Unit Tests (310+)         │ Individual functions
          └─────────────────────────────────┘
      ┌───────────────────────────────────────────┐
      │        Stress Tests (12)                  │ Large datasets
      └───────────────────────────────────────────┘
```

**Total:** 542 tests (430+ passing, 1 skipped conditionally)

### Test Categories

1. **Unit Tests:** Test individual functions in isolation
2. **Integration Tests:** Test multi-component workflows
3. **Property Tests:** Verify mathematical properties (symmetry, etc.)
4. **Stress Tests:** Validate with large datasets (50K+ geometries)
5. **OGC Conformance:** Ensure GeoSPARQL 1.1 compliance
6. **Real-World:** OpenStreetMap-based scenarios

### Continuous Integration

GitHub Actions workflow tests:
- ✅ Linux (Ubuntu), macOS, Windows
- ✅ Rust stable and beta
- ✅ All features enabled
- ✅ Zero warnings policy
- ✅ Code coverage >90% target
- ✅ Performance regression detection

---

## Design Decisions

### Why Separate Z/M Coordinates?

**Decision:** Store Z/M separately from geo_types coordinates.

**Rationale:**
- `geo_types` only supports 2D (X, Y)
- Modifying geo_types would break ecosystem compatibility
- Separate storage allows optional 3D (zero overhead for 2D)
- Can migrate to geo_types 3D support when available

### Why Multiple Spatial Indexes?

**Decision:** Provide 7 spatial index implementations.

**Rationale:**
- Different data characteristics favor different indexes
- R*-tree: 20-40% faster for general queries
- Hilbert R-tree: Best for bulk loading and large datasets
- K-d tree: Optimal for point clouds
- Spatial hash: O(1) insert for uniform data

### Why Dual Buffer Backend?

**Decision:** Support both GEOS and pure Rust buffer implementations.

**Rationale:**
- GEOS: Robust, handles all geometry types, battle-tested
- Pure Rust: No C++ dependency, better error handling
- Hybrid: Automatic selection based on geometry type

### Why SciRS2 Integration?

**Decision:** Use SciRS2 for all scientific computing.

**Rationale:**
- Unified scientific computing ecosystem
- SIMD, parallel, GPU acceleration built-in
- Consistent API across OxiRS modules
- Better performance than direct ndarray/rand usage

---

## Future Architecture

### Planned Enhancements

1. **Distributed Spatial Indexing**
   - Partition spatial index across multiple nodes
   - Coordinate queries with spatial routing

2. **Streaming Spatial Operations**
   - Process geometries as streams
   - Constant memory usage for large datasets

3. **Advanced GPU Algorithms**
   - GPU-accelerated topological relations
   - Parallel spatial joins on GPU

4. **Quantum-Inspired Optimization**
   - Use quantum annealing for spatial optimization
   - SciRS2 provides quantum computing primitives

---

## References

- [OGC GeoSPARQL 1.1 Specification](http://www.opengis.net/doc/IS/geosparql/1.1)
- [OGC Simple Features Specification](https://www.ogc.org/standards/sfa)
- [R-tree: A Dynamic Index Structure](http://www-db.deis.unibo.it/courses/SI-LS/papers/Gut84.pdf)
- [The R*-tree: An Efficient and Robust Access Method](https://infolab.usc.edu/csci599/Fall2001/paper/rstar-tree.pdf)
- [SciRS2 Documentation](https://github.com/cool-japan/scirs)

---

*This architecture document is maintained by the OxiRS team. Last updated: January 2026*
