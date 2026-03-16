# OxiRS TSDB - TODO

*Version: 0.2.2 | Last Updated: 2026-03-16*

## Status: Production Ready

**oxirs-tsdb** provides time-series database capabilities for IoT-scale workloads with RDF integration.

### Features

#### Storage Engine
- **Gorilla Compression** - XOR-based compression with 40:1 ratio target
- **Delta-of-Delta Timestamps** - Variable-length encoding (<2 bits per timestamp)
- **Time Chunks** - 2-hour duration chunks with metadata
- **Columnar Storage** - LRU cache, atomic writes, binary chunk format
- **Indexing** - SeriesIndex with time-based lookup and crash recovery

#### Query Engine
- **Range Queries** - Efficient chunk scanning with lazy decompression
- **Aggregations** - AVG, MIN, MAX, SUM, COUNT with incremental processing
- **Window Functions** - Moving averages, moving min/max, rolling std dev
- **Resampling** - Time bucketing with configurable aggregation
- **Interpolation** - Linear, forward fill, backward fill, spline

#### Write Path
- **Write-Ahead Log** - Crash recovery with fsync control
- **Write Buffer** - 100K points default with flush triggers
- **Compactor** - Background merging with compression optimization
- **Retention Enforcer** - Automatic downsampling and expiration

#### RDF Integration
- **Hybrid Store Adapter** - Transparent RDF/TSDB routing
- **RDF Bridge** - Confidence-based auto-detection
- **Subject-Series Mapping** - Bidirectional lookups

#### SPARQL Extensions
- **Temporal Functions** - ts:window, ts:resample, ts:interpolate
- **Query Router** - Automatic backend routing

### Test Coverage
- **1127 tests passing** with comprehensive coverage
- Integration, SPARQL, storage, query, and write path tests

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ Gorilla compression, delta-of-delta, columnar storage, WAL, hybrid RDF/TSDB, 128 tests

### v0.2.2 - Current Release (March 16, 2026)
- ✅ 1M+ writes/sec sustained performance
- ✅ Memory usage profiling and optimization
- ✅ Advanced compression (adaptive, dictionary, RLE)
- ✅ Anomaly detection algorithms
- ✅ Holt-Winters forecasting support
- ✅ Apache Arrow integration (zero-copy export)
- ✅ Parquet export for analytics
- ✅ Prometheus remote write API
- ✅ Rollup engine, downsampler, gap filler, retention policy
- ✅ Anomaly detector, series metadata, compression codec
- ✅ 1127 tests passing

### v0.3.0 - Planned (Q2 2026)
- [ ] GPU-accelerated aggregations
- [ ] Kalman filter integration
- [ ] DuckDB integration
- [ ] Raft replication for high availability
- [ ] Multi-region deployment

## Documentation

- 9 working examples included:
  - compression_demo.rs - Gorilla + delta-of-delta encoding
  - query_demo.rs - Range queries and aggregations
  - window_functions.rs - Moving averages and window operations
  - resampling_demo.rs - Time bucketing and downsampling
  - hybrid_storage.rs - Basic hybrid RDF + TSDB
  - benchmark_demo.rs - Performance measurement
  - hybrid_store_demo.rs - Advanced hybrid routing
  - auto_detection_demo.rs - Intelligent auto-detection
  - columnar_storage_demo.rs - Disk-backed storage

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS TSDB v0.2.2 - Time-series database for IoT-scale workloads*
