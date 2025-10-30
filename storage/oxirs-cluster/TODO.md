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
- **Released on crates.io**: `oxirs-cluster = "0.1.0-beta.1"` (experimental)

## 🎯 v0.1.0 Complete Feature Roadmap

### v0.1.0-beta.1 Target (December 2025) - ALL FEATURES

#### Core Clustering (Target: v0.1.0)
- [ ] Raft consensus optimization
- [ ] Node discovery improvements
- [ ] Leader election tuning
- [ ] Log replication optimization
- [ ] Byzantine fault tolerance
- [ ] Quorum-based operations
- [ ] Dynamic membership changes

#### Data Distribution (Target: v0.1.0)
- [ ] Partitioning strategies (hash, range, consistent hashing)
- [ ] Data rebalancing
- [ ] Consistency guarantees (strong, eventual, causal)
- [ ] Conflict resolution (CRDTs, vector clocks)
- [ ] Multi-datacenter support
- [ ] Geographic replication
- [ ] Sharding strategies

#### Stability (Target: v0.1.0)
- [ ] Network partition handling
- [ ] Crash recovery
- [ ] Data integrity verification
- [ ] Split-brain prevention
- [ ] Automatic failover
- [ ] Graceful degradation
- [ ] Circuit breakers

#### Monitoring (Target: v0.1.0)
- [ ] Cluster health monitoring
- [ ] Performance metrics
- [ ] Node status tracking
- [ ] Replication lag monitoring
- [ ] Distributed tracing
- [ ] Alerting system
- [ ] Visualization dashboard

#### Operations (Target: v0.1.0)
- [ ] Read replicas
- [ ] Backup and restore
- [ ] Rolling upgrades
- [ ] Zero-downtime migrations
- [ ] Disaster recovery
- [ ] Automated scaling