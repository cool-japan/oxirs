# OxiRS GeoSPARQL - TODO

*Version: 0.3.2 | Last Updated: July 12, 2026*

## Current Status

OxiRS GeoSPARQL v0.3.2 is production-ready, providing complete OGC GeoSPARQL 1.0/1.1 implementation for spatial data and queries in RDF/SPARQL.

### Production Features
- ✅ **GeoSPARQL Vocabulary** - Full support for GeoSPARQL ontology and datatypes
- ✅ **WKT/GML Support** - Parse and serialize Well-Known Text and GML geometries; WKT parser accepts optional Z/M coordinate dimensions
- ✅ **Topological Predicates** - All 24 relations (Simple Features, Egenhofer, RCC8); 16/24 run in pure Rust, the remaining 8 boundary-dependent Egenhofer/RCC8 relations require the companion `oxirs-geosparql-adapter-geos` crate (publish=false)
- ✅ **Geometric Operations** - Distance, buffer, convex hull, intersection, union
- ✅ **Spatial Indexing** - R-tree based spatial index for efficient queries
- ✅ **CRS Support** - Coordinate Reference System handling with EPSG codes
- ✅ **GeoPackage** - Read/write OGC GeoPackage files on the Pure-Rust `oxisql-core`/`oxisql-sqlite-compat` backend (no `libsqlite3`), with explicit `checkpoint()` for WAL flush
- ✅ **Shapefile** - Reader/writer now emits interior rings (holes) for `Polygon`/`MultiPolygon`
- ✅ **1967 tests passing** with comprehensive coverage

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ GeoSPARQL vocabulary, WKT/GML, topological predicates, spatial indexing
- ✅ 1713 tests passing

### v0.2.3 - Released (March 16, 2026)
- ✅ Additional coordinate systems
- ✅ 3D geometry support
- ✅ Enhanced spatial indexing
- ✅ Performance optimizations
- ✅ Geospatial reasoning
- ✅ Spatial aggregations
- ✅ Enhanced CRS transformations
- ✅ GeoJSON integration improvements

### v0.3.0 - Released (Q2 2026)
- [x] Full OGC GeoSPARQL 1.1 compliance (completed 2026-04-28)
  - [x] geof:relate — DE-9IM pattern matching via geo::algorithm::relate::Relate
  - [x] geof:simplify — geometry-level Douglas-Peucker (dispatches over geo_types variants)
  - [x] geof:boundary — pure-Rust implementation (OGC SFA §6.1.6.1 Mod-2 rule)
  - [x] geof:isValid — topological validity check
  - [x] geof:isRing — closed + simple LineString predicate
  - [x] geof:isClosed — endpoints equal predicate for LineString/MultiLineString
  - [x] Register all new functions in sparql_integration.rs
  - [x] tests/ogc_geosparql_1_1_conformance.rs conformance test suite
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Comprehensive benchmarks (completed 2026-04-29)

### v0.3.2 - Current Release (July 12, 2026)
- [x] `GeoPackage`'s SQLite backend migrated from `rusqlite` (bundled C `libsqlite3`) to Pure-Rust `oxisql-core`/`oxisql-sqlite-compat` (COOLJAPAN Pure-Rust Policy v2); new `GeoPackage::checkpoint()` for explicit WAL flush
- [x] GEOS (`geos`/`geos-sys` C FFI) quarantined out of this crate into the new companion `oxirs-geosparql-adapter-geos` crate (`publish = false`); the `geos-backend` Cargo feature was removed. `eh_meet`/`eh_inside`/`eh_contains` and `rcc8_ec`/`rcc8_tpp`/`rcc8_tppi`/`rcc8_ntpp`/`rcc8_ntppi` now return `GeoSparqlError::UnsupportedOperation` by default and point callers at the adapter crate; their SPARQL functions are excluded from `get_all_geosparql_functions()`
- [x] Shapefile writer emits interior rings (holes) for `Polygon`/`MultiPolygon`
- [x] WKT parser accepts optional Z/M coordinate dimensions (`POINT Z`, `POINT M`, `POINT ZM`)
- [x] `compressed_storage` round-trip fix: polygon holes and multi-part `MultiLineString`/`MultiPolygon` geometries no longer collapse to a single ring/part (new `ring_counts` field)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS GeoSPARQL v0.3.2 - Spatial data support for RDF*
