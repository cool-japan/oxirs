//! Triple and quad index structures (SPO, POS, OSP, GSPO, GPOS, GOSP)
//!
//! This module implements triple indexes for efficient SPARQL query execution,
//! quad indexes for named graph (RDF dataset) support, and spatial indexes
//! for GeoSPARQL geospatial queries.

pub mod bloom_filter;
pub mod gpu_accelerated_scan;
pub mod quad;
pub mod spatial;
pub mod triple;
pub mod triple_index;

pub use bloom_filter::{BloomFilter, BloomFilterConfig, BloomFilterStats, CountingBloomFilter};
pub use gpu_accelerated_scan::{
    GpuAccelerationConfig, GpuBackendType, GpuIndexScanner, GpuScanStats, JoinComponent,
    TriplePattern,
};
pub use quad::{GospKey, GposKey, GspoKey, Quad, QuadIndexes};
pub use spatial::{
    BoundingBox, Geometry, LineString, Point, Polygon, SpatialIndex, SpatialQuery,
    SpatialQueryResult, SpatialStats,
};
pub use triple::{EmptyValue, OspKey, PosKey, SpoKey, Triple};
pub use triple_index::{OspIndex, PosIndex, SpoIndex, TripleIndex, TripleIndexes};
