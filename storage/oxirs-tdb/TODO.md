# OxiRS TDB - TODO

*Last Updated: October 4, 2025*

## ✅ Current Status: v0.1.0-alpha.2 Released

**oxirs-tdb** provides high-performance RDF storage with MVCC and ACID transactions.

### Alpha.2 Release Status (October 4, 2025)
- **Comprehensive test suite** with persisted dataset coverage & zero warnings
- **MVCC + ACID transactions** powering disk-backed CLI workflows
- **B+ Tree indexing** optimized for streaming import/export pipelines
- **Federation-aware storage** cooperating with `SERVICE` queries and shards
- **Telemetry hooks** for Prometheus metrics and cache diagnostics
- **Released on crates.io**: `oxirs-tdb = "0.1.0-alpha.2"`

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Performance
- [ ] Index optimization
- [ ] Buffer pool tuning
- [ ] Compression improvements
- [ ] Write-ahead log optimization

#### Features
- [ ] Backup and restore utilities
- [ ] Database compaction
- [ ] Statistics collection
- [ ] Query hint support

#### Stability
- [ ] Crash recovery improvements
- [ ] Corruption detection and repair
- [ ] Transaction conflict resolution
- [ ] Deadlock detection

#### Monitoring
- [ ] Performance metrics
- [ ] Storage statistics
- [ ] Health checks
- [ ] Diagnostic tools

### v0.2.0 Targets (Q1 2026)
- [ ] Full TDB2 feature parity
- [ ] Advanced compression algorithms
- [ ] Distributed transaction support
- [ ] Hot backup capabilities