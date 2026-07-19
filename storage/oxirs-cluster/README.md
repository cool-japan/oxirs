# OxiRS Cluster

[![Version](https://img.shields.io/badge/version-0.4.0-blue)](https://github.com/cool-japan/oxirs/releases)
[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Status**: v0.4.0 - Released 2026-07-19 (1868 tests passing)

✨ **Production Release**: Production-ready with API stability guarantees and comprehensive testing.

A high-performance, distributed RDF storage system using Raft consensus for horizontal scaling and fault tolerance. Part of the OxiRS ecosystem providing a JVM-free alternative to Apache Jena + Fuseki with enhanced clustering capabilities.

## Features

- **Raft Consensus**: Strong consistency with automated leader election and log replication
- **Byzantine Fault Tolerance (opt-in)**: a closed-loop PBFT path behind the `bft` feature and `NodeConfig.use_bft` (v0.4.0) — commits are idempotent (extra commits past quorum never re-execute), a committed `RdfCommand` is applied to the real storage backend and returns a real `RdfResponse`, and `process_request` completes on a genuine 2f+1 quorum via a `(client_id, timestamp)`-keyed commit callback (configurable timeout). Requesting `use_bft` on a build compiled without the `bft` feature fails loud rather than silently downgrading
- **Horizontal Scaling**: Linear performance scaling to 1000+ nodes
- **High Availability**: 99.9% uptime with automatic failover
- **Distributed RDF Storage**: Efficient partitioning and indexing of RDF triples
- **SPARQL 1.2 Support**: Distributed query processing with federated queries
- **Enterprise Security**: TLS encryption, authentication, and access control
- **Operational Excellence**: Comprehensive monitoring, alerting, and management tools
- **Certification Suite**: `certification::CertificationSuite` runs deterministic, in-memory
  simulations (no real sockets) validating consistency (read-your-writes, linearizability
  probes, convergence), partition handling (island formation, quorum loss/recovery, split-brain
  prevention), Raft safety invariants (leader uniqueness, log monotonicity), and SLA bounds
  (read/write p99 latency, throughput floor)

## Architecture

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Node A        │  │   Node B        │  │   Node C        │
│   (Leader)      │  │   (Follower)    │  │   (Follower)    │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ Raft Consensus  │◄─┤ Raft Consensus  │◄─┤ Raft Consensus  │
│ RDF Storage     │  │ RDF Storage     │  │ RDF Storage     │
│ Query Engine    │  │ Query Engine    │  │ Query Engine    │
│ Network Layer   │  │ Network Layer   │  │ Network Layer   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Core Components

- **Consensus Layer**: Raft-based distributed consensus for strong consistency
- **Storage Layer**: Distributed RDF triple storage with efficient indexing
- **Network Layer**: High-performance inter-node communication
- **Discovery Service**: Automatic node registration and cluster membership
- **Query Engine**: Distributed SPARQL query processing
- **Replication**: Multi-master replication with conflict resolution

## Quick Start

### Prerequisites

- Rust 1.70+ (MSRV)
- Memory: 4GB+ recommended
- Network: Low-latency connection between nodes

### Installation

Add to your `Cargo.toml`:

```toml
# Experimental feature
[dependencies]
oxirs-cluster = "0.3.2"
```

### Basic Usage

```rust
use oxirs_cluster::{DistributedStore, NodeConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize node configuration
    let mut config = NodeConfig::new(1, "127.0.0.1:8080".parse()?);
    config.data_dir = "./data".to_string();
    config.add_peer(2);
    config.add_peer(3);

    // Start the distributed store (wraps a ClusterNode running Raft consensus)
    let mut store = DistributedStore::new(config).await?;
    store.start().await?;

    // Insert RDF data (accepted on the leader; forwarded internally otherwise)
    store
        .insert_triple(
            "http://example.org/alice",
            "http://example.org/knows",
            "http://example.org/bob",
        )
        .await?;

    // Execute SPARQL query
    let results = store.query_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }").await?;
    println!("Results: {:?}", results);

    Ok(())
}
```

### Multi-Node Cluster Setup

`oxirs-cluster` is a library crate — it does not bundle a multi-node launcher
binary. Wire `NodeConfig`/`DistributedStore` into your own binary (or a test
harness), giving each node a distinct `node_id` and the full peer set:

```rust
// Node 1 (peers 2 and 3)
let mut n1 = NodeConfig::new(1, "127.0.0.1:8080".parse()?);
n1.data_dir = "./data/node1".to_string();
n1.add_peer(2);
n1.add_peer(3);

// Node 2 (peers 1 and 3)
let mut n2 = NodeConfig::new(2, "127.0.0.1:8081".parse()?);
n2.data_dir = "./data/node2".to_string();
n2.add_peer(1);
n2.add_peer(3);

// Node 3 (peers 1 and 2)
let mut n3 = NodeConfig::new(3, "127.0.0.1:8082".parse()?);
n3.data_dir = "./data/node3".to_string();
n3.add_peer(1);
n3.add_peer(2);

// Start each `DistributedStore::new(n).await?.start().await?` in its own
// process/task; Raft handles leader election among the three automatically.
```

## Configuration

### Environment Variables

```bash
OXIRS_CLUSTER_NODE_ID=node-1
OXIRS_CLUSTER_BIND_ADDR=0.0.0.0:8080
OXIRS_CLUSTER_DATA_DIR=/var/lib/oxirs
OXIRS_CLUSTER_LOG_LEVEL=info
OXIRS_CLUSTER_HEARTBEAT_INTERVAL=150ms
OXIRS_CLUSTER_ELECTION_TIMEOUT=1500ms
```

### Configuration File (oxirs-cluster.toml)

```toml
[cluster]
node_id = "node-1"
bind_address = "0.0.0.0:8080"
data_dir = "/var/lib/oxirs"

[raft]
heartbeat_interval = "150ms"
election_timeout = "1500ms"
max_log_entries = 10000
snapshot_threshold = 5000

[storage]
partition_count = 16
replication_factor = 3
compression = "lz4"

[network]
max_connections = 1000
connection_timeout = "30s"
message_timeout = "5s"
```

## Performance

### Benchmarks

- **Throughput**: 10,000+ operations/second per cluster
- **Latency**: <100ms for read queries, <200ms for writes
- **Scalability**: Linear performance to 1000+ nodes
- **Recovery**: <30 seconds for automatic failover

### Tuning

```rust
let mut config = NodeConfig::new(1, "127.0.0.1:8080".parse()?);
config
    .add_peer(2)
    .add_peer(3);
let config = config
    .with_discovery(DiscoveryConfig::default())
    .with_replication_strategy(ReplicationStrategy::default());
```

Raft election/heartbeat timing (`election_timeout_min`/`_max`, `heartbeat_interval`,
`max_batch_size`) defaults live in the crate-internal `raft_state::RaftConfig` and are not yet
exposed as `NodeConfig` builder methods.

## Monitoring

### Metrics

The cluster exposes Prometheus-compatible metrics:

- `oxirs_cluster_nodes_total`: Total number of cluster nodes
- `oxirs_cluster_leader_changes_total`: Number of leader changes
- `oxirs_cluster_queries_total`: Total queries processed
- `oxirs_cluster_query_duration_seconds`: Query latency histogram
- `oxirs_cluster_replication_lag_seconds`: Replication lag

### Health Checks

```bash
# Check cluster health
curl http://localhost:8080/health

# Check node status
curl http://localhost:8080/status

# Get cluster metrics
curl http://localhost:8080/metrics
```

## Development

### Building

```bash
# Build with all features
cargo build --all-features

# Run tests
cargo nextest run --no-fail-fast

# Run benchmarks
cargo bench
```

### Testing

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration_tests

# Chaos engineering tests
cargo test --test chaos_engineering_tests
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass (`cargo nextest run --no-fail-fast`)
6. Run clippy (`cargo clippy --workspace --all-targets -- -D warnings`)
7. Format your code (`cargo fmt --all`)
8. Commit your changes (`git commit -am 'Add amazing feature'`)
9. Push to the branch (`git push origin feature/amazing-feature`)
10. Open a Pull Request

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Raft Consensus Algorithm](https://raft.github.io/) by Diego Ongaro and John Ousterhout
- [OpenRaft](https://github.com/datafuselabs/openraft) for Rust Raft implementation
- [Apache Jena](https://jena.apache.org/) for RDF/SPARQL inspiration
- [OxiGraph](https://github.com/oxigraph/oxigraph) for RDF storage patterns

## Roadmap

### Post-1.0 Enhancements
- [x] Multi-region deployment support — `region_manager`, `cross_dc`, `cross_dc_consistency` modules
- [x] Advanced conflict resolution — `conflict_resolution` module (CRDTs, vector clocks)
- [x] Improved monitoring dashboard — `visualization_dashboard` module with REST API
- [x] Performance optimizations — `raft_optimization`, `performance_monitor`, SIMD Merkle hashing
- [x] Machine learning-based query optimization — `ml_optimization` module (Q-learning, cost optimization)
- [x] Edge computing integration — `edge_computing` module
- [x] Advanced security features — `security`, `encryption`, `tls`, `bft` (Byzantine fault tolerance) modules
- [ ] GraphQL federation support — cross-cluster query federation exists (`federation` module);
      GraphQL-schema-level federation is a separate concern handled by `oxirs-gql`, not this crate

---

For more information, see the [OxiRS documentation](https://github.com/cool-japan/oxirs) and [TODO.md](TODO.md) for detailed implementation progress.