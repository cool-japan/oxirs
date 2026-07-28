# Buffer Operations Guide for oxirs-geosparql

## Overview

`oxirs-geosparql` has **one** buffer implementation: `geo::algorithm::buffer`, which is
Pure Rust (backed by the `i_overlay` offsetting engine). There is nothing to enable, no
backend to choose, and no C++ library to install.

This replaced two narrower paths that earlier versions shipped:

- a `rust-buffer` feature backed by `geo-buffer`'s straight-skeleton algorithm, which
  handled Polygon and MultiPolygon only, and
- a `geos-backend` feature (later a `publish = false` adapter crate) that linked the
  GEOS C library for every other geometry type and for cap/join styles.

Both are gone. If you were passing `--features rust-buffer` or depending on
`oxirs-geosparql-adapter-geos`, drop them; `buffer()` now does strictly more than either
did. The removed `buffer_rust()` function has no replacement because `buffer()` covers
its cases.

## Quick Start

```toml
[dependencies]
oxirs-geosparql = "0.4.1"
```

No system dependencies, no feature flags.

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::buffer;

let polygon = Geometry::from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")?;
let buffered = buffer(&polygon, 2.0)?;
println!("{}", buffered.to_wkt());
```

`buffer()` always returns a `MultiPolygon`, whatever the input type.

## Geometry Type Support

Every geometry type buffers:

| Input | Result |
|---|---|
| `Point` | Disc around the point |
| `MultiPoint` | One disc per point (merged where they overlap) |
| `LineString` | Corridor around the line |
| `MultiLineString` | Corridor around each line |
| `Polygon` | Expanded (or eroded) polygon |
| `MultiPolygon` | Expanded (or eroded) parts |
| `GeometryCollection` | Union of the members' buffers |

Positive distances expand, negative distances erode. Eroding a polygon by more than its
inradius yields an empty result rather than an error.

## Cap and Join Styles

`buffer_with_params` exposes the OGC cap/join model:

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::{
    buffer_with_params, BufferParams, CapStyle, JoinStyle,
};

let line = Geometry::from_wkt("LINESTRING(0 0, 5 0, 5 5)")?;

let params = BufferParams {
    cap_style: CapStyle::Square,
    join_style: JoinStyle::Mitre,
    quadrant_segments: 16,
    mitre_limit: 5.0,
};

let buffered = buffer_with_params(&line, 1.0, &params)?;
```

| `BufferParams` field | Meaning | Default |
|---|---|---|
| `cap_style` | `Round`, `Flat` (butt), or `Square` ends on linear geometry | `Round` |
| `join_style` | `Round`, `Mitre`, or `Bevel` corners | `Round` |
| `quadrant_segments` | Segments per quarter circle — higher is smoother and slower | `8` |
| `mitre_limit` | Max ratio of miter extension to buffer distance before falling back to a bevel | `5.0` |

`quadrant_segments` must be at least 1 and `mitre_limit` at least 1.0; anything else is
rejected with `GeoSparqlError::InvalidParameter`.

### A note on numeric fidelity

`quadrant_segments` and `mitre_limit` are the OGC/JTS spelling of curve resolution.
`geo` models the same two knobs as angles, so the values are converted rather than passed
straight through: `quadrant_segments` becomes an angle per segment of
`(π/2) / quadrant_segments` radians, and `mitre_limit` becomes the sharpest corner angle
that still gets a miter, `2·asin(1/mitre_limit)`. The JTS default of 8 segments maps to
0.196 rad, which is what `geo`'s own default of 0.20 approximates — so results are very
close to JTS/GEOS/PostGIS but not bit-identical. If you need agreement with a PostGIS
result set, compare with a tolerance rather than an equality check.

## Boundary

`boundary()` is likewise Pure Rust now (it used to be GEOS-only), following the OGC
Simple Features definition: empty for points, the endpoints of a LineString, the rings of
a Polygon, and the union of component boundaries for the Multi\* types.

## Examples

Two runnable examples ship with the crate:

```bash
cargo run --example buffer_comparison   # every geometry type, all cap/join styles
cargo run --example pure_rust_buffer    # polygon-focused walkthrough
```

## Performance

```bash
cargo bench --bench buffer_performance
```

The benchmark covers polygon buffering (positive and negative), non-polygon buffering,
and a WKT round-trip.

Tips:

- Lower `quadrant_segments` when you do not need smooth curves; it is the main cost knob
  for round caps and joins.
- `CapStyle::Flat` / `JoinStyle::Bevel` avoid arc generation entirely.
- Buffer once and reuse the result rather than re-buffering inside a loop.

## Troubleshooting

**"quadrant_segments must be at least 1" / "mitre_limit must be at least 1.0"**

`BufferParams` was constructed with an out-of-range value. Start from
`BufferParams::default()` and override only what you need.

**An eroded polygon came back empty**

A negative distance larger than the polygon's inradius removes it entirely. That is
correct behaviour, not an error — check `is_empty()` on the result.

**Results differ slightly from PostGIS/GEOS**

Expected; see the numeric fidelity note above.

## Migration from PostGIS

```sql
-- PostGIS
SELECT ST_AsText(ST_Buffer(geom, 10.0)) FROM buildings;
```

```rust
use oxirs_geosparql::geometry::Geometry;
use oxirs_geosparql::functions::geometric_operations::buffer;

let geom = Geometry::from_wkt(&wkt_from_postgis)?;
let buffered = buffer(&geom, 10.0)?;
let result_wkt = buffered.to_wkt();
```

`ST_Buffer`'s `quad_segs`, `endcap` and `join` styling parameters map onto
`BufferParams` as described above.

## Further Reading

- [GeoSPARQL 1.1 Specification](https://www.ogc.org/standards/geosparql)
- [Simple Features Specification](https://www.ogc.org/standards/sfa)
- [`geo::algorithm::buffer`](https://docs.rs/geo/latest/geo/algorithm/buffer/index.html)
- [`i_overlay`](https://crates.io/crates/i_overlay)

## Contributing

Found a bug or have a feature request? Please open an issue on [GitHub](https://github.com/cool-japan/oxirs).

---

**Last Updated**: 2026-07-27
**oxirs-geosparql Version**: 0.4.1
