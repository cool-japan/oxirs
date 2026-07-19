//! Triple and quad index structures (SPO, POS, OSP, GSPO, GPOS, GOSP)
//!
//! This module implements triple indexes for efficient SPARQL query execution,
//! quad indexes for named graph (RDF dataset) support, spatial indexes
//! for GeoSPARQL geospatial queries, and SIMD-accelerated filtering.

// The core triple index (triple, triple_index) compiles strictly; the
// peripheral index engines keep a scoped dead-code allow (out of scope for the
// durability pass).
#[allow(dead_code, unused_imports, unused_variables)]
pub mod adaptive;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod bloom_filter;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod btree_index;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod gpu_accelerated_scan;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod quad;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod simd_triple_filter;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod spatial;
pub mod triple;
pub mod triple_index;

pub use adaptive::{
    AdaptiveIndexSelector, IndexSelectionThresholds, IndexStats, IndexType, QueryPattern,
    TieredStorage,
};
pub use bloom_filter::{BloomFilter, BloomFilterConfig, BloomFilterStats, CountingBloomFilter};
pub use btree_index::{BTreeTripleIndex, EncodedTriple, TripleIndexSet, TripleOrdering};
pub use gpu_accelerated_scan::{
    GpuAccelerationConfig, GpuBackendType, GpuIndexScanner, GpuScanStats, JoinComponent,
    TriplePattern,
};
pub use quad::{GospKey, GposKey, GspoKey, Quad, QuadIndexes, QuadScan};
pub use simd_triple_filter::{FilterStats, SimdTripleFilter, SimdTriplePattern};
pub use spatial::{
    BoundingBox, Geometry, LineString, Point, Polygon, SpatialIndex, SpatialQuery,
    SpatialQueryResult, SpatialStats,
};
pub use triple::{EmptyValue, OspKey, PosKey, SpoKey, Triple};
pub use triple_index::{OspIndex, PosIndex, SpoIndex, TripleIndex, TripleIndexes, TripleScan};
