# OxiRS Cluster - TODO

*Last Updated: October 12, 2025*

## ✅ Current Status: v0.1.0-alpha.3 Released (Experimental)

**oxirs-cluster** provides distributed RDF storage with Raft consensus (experimental feature).

### Alpha.3 Release Status (October 12, 2025)
- **Comprehensive test suite** with persisted dataset scenarios & zero warnings
- **Raft consensus** integrated with durable storage checkpoints
- **Distributed RDF storage** synchronized with CLI persistence pipeline
- **High availability** plus Prometheus/SciRS2 metrics for cluster health
- **Federation awareness** enabling cross-cluster SPARQL `SERVICE` routing
- **Released on crates.io**: `oxirs-cluster = "0.1.0-alpha.3"` (experimental)

## 🎯 Post-Alpha Development Roadmap

### Beta Release Targets (v0.1.0-beta.1 - December 2025)

#### Core Clustering
- [ ] Raft consensus optimization
- [ ] Node discovery improvements
- [ ] Leader election tuning
- [ ] Log replication optimization

#### Data Distribution
- [ ] Partitioning strategies
- [ ] Data rebalancing
- [ ] Consistency guarantees
- [ ] Conflict resolution

#### Stability
- [ ] Network partition handling
- [ ] Crash recovery
- [ ] Data integrity verification
- [ ] Split-brain prevention

#### Monitoring
- [ ] Cluster health monitoring
- [ ] Performance metrics
- [ ] Node status tracking
- [ ] Replication lag monitoring

### v0.2.0 Targets (Q1 2026)
- [ ] Multi-datacenter support
- [ ] Read replicas
- [ ] Backup and restore
- [ ] Rolling upgrades