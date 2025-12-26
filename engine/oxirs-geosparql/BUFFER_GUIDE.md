# Buffer Operations Guide for oxirs-geosparql

## Overview

oxirs-geosparql provides **two buffer implementations** with **automatic backend selection**:

1. **Pure Rust Buffer** (`rust-buffer` feature) - No C++ dependencies
2. **GEOS Backend** (`geos-backend` feature) - Industry-standard C++ library

The `buffer()` function **automatically chooses** the optimal backend based on geometry type.

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
# Recommended: Both backends for maximum compatibility
oxirs-geosparql = { version = "0.1.0-rc.1", features = ["rust-buffer", "geos-backend"] }

# Or choose only one:
# oxirs-geosparql = { version = "0.1.0-rc.1", features = ["rust-buffer"] }  # Pure Rust only
# oxirs-geosparql = { version = "0.1.0-rc.1", features = ["geos-backend"] }  # GEOS only
```

### System Requirements

#### For `rust-buffer` feature
- ✅ No system dependencies
- ✅ Works out of the box

#### For `geos-backend` feature
- ⚠️ Requires GEOS C++ library installation

**macOS**:
```bash
brew install geos
```

**Ubuntu/Debian**:
```bash
sudo apt-get install libgeos-dev
```

**Fedora/RHEL**:
```bash
sudo dnf install geos-devel
```

## Usage

### Basic Buffer Operation

The `buffer()` function works transparently with **automatic backend selection**:

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::buffer;

// Positive buffer (expansion)
let square = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
let expanded = buffer(&square, 2.0)?;  // Expands 2 units

// Negative buffer (erosion/shrinking)
let shrunk = buffer(&square, -1.0)?;  // Shrinks 1 unit
```

### Automatic Backend Selection

The `buffer()` function automatically selects the optimal backend:

| Geometry Type | With `rust-buffer` | With `geos-backend` | With Both Features |
|--------------|-------------------|--------------------|--------------------|
| **Polygon** | ✅ Pure Rust | ✅ GEOS | ✅ Pure Rust (preferred) |
| **MultiPolygon** | ✅ Pure Rust | ✅ GEOS | ✅ Pure Rust (preferred) |
| **Point** | ❌ Error | ✅ GEOS | ✅ GEOS (required) |
| **LineString** | ❌ Error | ✅ GEOS | ✅ GEOS (required) |
| **MultiPoint** | ❌ Error | ✅ GEOS | ✅ GEOS (required) |
| **MultiLineString** | ❌ Error | ✅ GEOS | ✅ GEOS (required) |

### Explicit Backend Selection

If you want to **explicitly** use a specific backend:

```rust
use oxirs_geosparql::functions::geometric_operations::{buffer_rust, buffer_with_params, BufferParams};

// Force Pure Rust (only for Polygon/MultiPolygon)
#[cfg(feature = "rust-buffer")]
{
    let result = buffer_rust(&polygon, 2.0)?;
}

// Force GEOS with custom parameters
#[cfg(feature = "geos-backend")]
{
    let params = BufferParams {
        cap_style: CapStyle::Square,
        join_style: JoinStyle::Mitre,
        quadrant_segments: 16,
        mitre_limit: 5.0,
    };
    let result = buffer_with_params(&geometry, 2.0, &params)?;
}
```

## Backend Comparison

### 📦 Pure Rust Buffer (`rust-buffer`)

**Powered by**: [geo-buffer](https://crates.io/crates/geo-buffer) crate (Straight Skeleton algorithm)

**Supported Geometries**:
- ✅ Polygon (simple, non-convex, with holes)
- ✅ MultiPolygon

**Advantages**:
- ✅ **No C++ dependencies** - Works on any platform without external libraries
- ✅ **Easy cross-compilation** - Compile to WebAssembly, mobile, embedded systems
- ✅ **Smaller binary size** - No GEOS library bundled
- ✅ **Pure Rust safety** - Memory-safe, no FFI overhead
- ✅ **Fast for polygons** - Optimized for 2D polygon operations

**Limitations**:
- ⚠️ **Polygon/MultiPolygon only** - Cannot buffer Point or LineString
- ⚠️ **Different algorithm** - Results may differ slightly from PostGIS/QGIS (which use GEOS)
- ⚠️ **No advanced parameters** - Cannot customize cap/join styles

**Use Cases**:
- Web applications (WASM deployment)
- Embedded systems
- Mobile applications
- Projects avoiding C++ dependencies
- Urban planning / building setbacks
- Cartography / map buffering

### 🔧 GEOS Backend (`geos-backend`)

**Powered by**: [GEOS C++ library](https://libgeos.org) (Modified straight skeleton + others)

**Supported Geometries**:
- ✅ All geometry types (Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)

**Advantages**:
- ✅ **All geometry types** - Buffers any geometry
- ✅ **Industry-standard** - Used by PostGIS, QGIS, GDAL, Shapely
- ✅ **20+ years of development** - Battle-tested, highly optimized
- ✅ **Advanced parameters** - Custom cap styles (round/flat/square), join styles (round/mitre/bevel)
- ✅ **Identical to PostGIS** - Same results as PostGIS `ST_Buffer()`

**Limitations**:
- ⚠️ **Requires GEOS C++ library** - Must install system dependency
- ⚠️ **Larger binary size** - GEOS library adds ~5MB
- ⚠️ **Cross-compilation complexity** - GEOS must be available on target platform
- ⚠️ **FFI overhead** - Small performance cost for crossing FFI boundary

**Use Cases**:
- Server-side applications
- Desktop GIS applications
- Scientific computing
- PostGIS compatibility required
- Need to buffer Point/LineString
- Advanced buffer styling required

## Examples

### Example 1: Simple Polygon Buffer

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::buffer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let building = Geometry::from_wkt("POLYGON((0 0, 20 0, 20 15, 0 15, 0 0))")?;
    println!("Building footprint: {}", building.to_wkt());

    // Create 5-meter buffer zone around building
    let buffer_zone = buffer(&building, 5.0)?;
    println!("5m buffer zone: {}", buffer_zone.to_wkt());

    // Result: MULTIPOLYGON(((-5 -5, 25 -5, 25 20, -5 20, -5 -5)))
    // ✅ Uses: Pure Rust (if available), else GEOS

    Ok(())
}
```

### Example 2: Urban Planning - Building Setback

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::buffer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // City block with exclusion zone
    let city_block = Geometry::from_wkt(
        "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0), (30 30, 70 30, 70 70, 30 70, 30 70))"
    )?;

    // Apply 10-meter setback from all boundaries
    let buildable_area = buffer(&city_block, -10.0)?;
    println!("Buildable area: {}", buildable_area.to_wkt());

    // ✅ Pure Rust handles polygon with hole perfectly

    Ok(())
}
```

### Example 3: Point Buffer (Requires GEOS)

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::buffer;

#[cfg(feature = "geos-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // GPS location
    let location = Geometry::from_wkt("POINT(139.7 35.7)")?;  // Tokyo

    // Create 1km radius circle around point
    let search_area = buffer(&location, 0.01)?;  // ~1km in degrees
    println!("Search area: {}", search_area.to_wkt());

    // Result: POLYGON with ~32 points forming a circle
    // ✅ Uses: GEOS (required for Point)

    Ok(())
}

#[cfg(not(feature = "geos-backend"))]
fn main() {
    eprintln!("❌ Point buffer requires geos-backend feature");
}
```

### Example 4: LineString Buffer (Road Corridor)

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::buffer;

#[cfg(feature = "geos-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Road centerline
    let road = Geometry::from_wkt("LINESTRING(0 0, 100 50, 200 30)")?;

    // Create 10-meter road corridor (5m each side)
    let corridor = buffer(&road, 10.0)?;
    println!("Road corridor: {}", corridor.to_wkt());

    // Result: POLYGON with rounded ends
    // ✅ Uses: GEOS (required for LineString)

    Ok(())
}

#[cfg(not(feature = "geos-backend"))]
fn main() {
    eprintln!("❌ LineString buffer requires geos-backend feature");
}
```

### Example 5: Advanced GEOS Parameters

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::{buffer_with_params, BufferParams, CapStyle, JoinStyle};

#[cfg(feature = "geos-backend")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let line = Geometry::from_wkt("LINESTRING(0 0, 100 0)")?;

    // Custom buffer with square caps and mitre joins
    let params = BufferParams {
        cap_style: CapStyle::Square,     // Extend beyond endpoints
        join_style: JoinStyle::Mitre,    // Sharp corners
        quadrant_segments: 16,           // More segments for smoother curves
        mitre_limit: 10.0,               // Allow sharp spikes
    };

    let buffered = buffer_with_params(&line, 5.0, &params)?;
    println!("Custom buffer: {}", buffered.to_wkt());

    Ok(())
}

#[cfg(not(feature = "geos-backend"))]
fn main() {
    eprintln!("❌ Advanced parameters require geos-backend feature");
}
```

## Performance Considerations

### Benchmark Results (Approximate)

| Operation | Pure Rust | GEOS | Winner |
|-----------|-----------|------|--------|
| Small Polygon (10x10) | ~5 µs | ~15 µs | 🏆 Pure Rust (3x faster) |
| Medium Polygon (100x100) | ~8 µs | ~20 µs | 🏆 Pure Rust (2.5x faster) |
| Large Polygon (1000x1000) | ~12 µs | ~30 µs | 🏆 Pure Rust (2.5x faster) |
| Polygon with Hole | ~15 µs | ~35 µs | 🏆 Pure Rust (2.3x faster) |
| Point Buffer | N/A | ~20 µs | 🏆 GEOS (only option) |
| LineString Buffer | N/A | ~25 µs | 🏆 GEOS (only option) |

**Run benchmarks yourself**:
```bash
cargo bench --bench buffer_performance --features rust-buffer,geos-backend
```

### Performance Tips

1. **For Polygon/MultiPolygon**: Pure Rust is **2-3x faster** than GEOS
2. **For Point/LineString**: GEOS is the **only option**
3. **Enable both features** for best performance (automatic selection)
4. **WKT parsing** can be slow for large geometries - cache parsed `Geometry` objects

## Troubleshooting

### Error: "Buffer operation requires 'rust-buffer' or 'geos-backend' feature"

**Solution**: Enable at least one buffer feature in `Cargo.toml`:
```toml
oxirs-geosparql = { version = "0.1.0-rc.1", features = ["rust-buffer"] }
```

### Error: "Pure Rust buffer only supports Polygon and MultiPolygon"

**Solution**: Enable `geos-backend` feature for Point/LineString support:
```toml
oxirs-geosparql = { version = "0.1.0-rc.1", features = ["rust-buffer", "geos-backend"] }
```

### Error: "failed to run custom build command for `geos-sys`"

**Cause**: GEOS C++ library not installed

**Solution**:
```bash
# macOS
brew install geos

# Ubuntu/Debian
sudo apt-get install libgeos-dev

# Fedora/RHEL
sudo dnf install geos-devel
```

### Difference in Results Between Pure Rust and GEOS

**Explanation**: Pure Rust uses **Straight Skeleton** algorithm, while GEOS uses a **Modified Straight Skeleton** with additional optimizations. Results may differ slightly:

- Pure Rust: More mathematically pure, simpler implementation
- GEOS: Optimized for cartographic use, handles edge cases better

For **PostGIS compatibility**, use `geos-backend`. For **pure Rust simplicity**, use `rust-buffer`.

## Migration from PostGIS

If you're migrating from PostGIS `ST_Buffer()`:

```sql
-- PostGIS
SELECT ST_AsText(ST_Buffer(geom, 10.0)) FROM buildings;
```

```rust
// oxirs-geosparql with GEOS backend (identical results)
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::buffer;

let geom = Geometry::from_wkt(&wkt_from_postgis)?;
let buffered = buffer(&geom, 10.0)?;
let result_wkt = buffered.to_wkt();

// ✅ Enable geos-backend for PostGIS-identical results
```

## Feature Flag Summary

| Configuration | Polygon/MultiPolygon | Point/LineString | Binary Size | Dependencies |
|--------------|---------------------|------------------|-------------|--------------|
| **No features** | ❌ Error | ❌ Error | Smallest | None |
| **rust-buffer only** | ✅ Pure Rust | ❌ Error | Small | None |
| **geos-backend only** | ✅ GEOS | ✅ GEOS | Large | GEOS C++ |
| **Both (recommended)** | ✅ Pure Rust | ✅ GEOS | Large | GEOS C++ |

## Further Reading

- [GeoSPARQL 1.1 Specification](https://www.ogc.org/standards/geosparql)
- [GEOS Project](https://libgeos.org)
- [geo-buffer crate](https://crates.io/crates/geo-buffer)
- [Simple Features Specification](https://www.ogc.org/standards/sfa)

## Running Examples

See working examples in the `examples/` directory:

```bash
# Pure Rust buffer examples
cargo run --example pure_rust_buffer --features rust-buffer

# Backend comparison
cargo run --example buffer_comparison --features rust-buffer,geos-backend
```

## Contributing

Found a bug or have a feature request? Please open an issue on [GitHub](https://github.com/cool-japan/oxirs).

---

**Last Updated**: 2025-12-25
**oxirs-geosparql Version**: 0.1.0-rc.1
