# OxiRS GeoSPARQL - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Current Status

OxiRS GeoSPARQL v0.3.1 is production-ready, providing complete OGC GeoSPARQL 1.0/1.1 implementation for spatial data and queries in RDF/SPARQL.

### Production Features
- ✅ **GeoSPARQL Vocabulary** - Full support for GeoSPARQL ontology and datatypes
- ✅ **WKT/GML Support** - Parse and serialize Well-Known Text and GML geometries
- ✅ **Topological Predicates** - All 24 relations (Simple Features, Egenhofer, RCC8)
- ✅ **Geometric Operations** - Distance, buffer, convex hull, intersection, union
- ✅ **Spatial Indexing** - R-tree based spatial index for efficient queries
- ✅ **CRS Support** - Coordinate Reference System handling with EPSG codes
- ✅ **1713 tests passing** with comprehensive coverage

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ GeoSPARQL vocabulary, WKT/GML, topological predicates, spatial indexing
- ✅ 1713 tests passing

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Additional coordinate systems
- ✅ 3D geometry support
- ✅ Enhanced spatial indexing
- ✅ Performance optimizations
- ✅ Geospatial reasoning
- ✅ Spatial aggregations
- ✅ Enhanced CRS transformations
- ✅ GeoJSON integration improvements

### v0.3.0 - Planned (Q2 2026)
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

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS GeoSPARQL v0.2.3 - Spatial data support for RDF*
