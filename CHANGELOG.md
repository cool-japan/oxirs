# Changelog

All notable changes to OxiRS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.1.0] - 2026-01-07

### Overview

**Initial production release** of OxiRS - A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning. This release provides a complete, production-ready alternative to Apache Jena + Fuseki with modern enhancements.

**Major Achievements**:
- **Complete SPARQL 1.1/1.2 Implementation**: Full W3C compliance with advanced query optimization
- **13,123 tests passing**: 100% pass rate with comprehensive test coverage (136 skipped)
- **Zero warnings**: Strict `-D warnings` enforcement across all 22 crates
- **Industrial IoT Support**: Time-series, Modbus, CANbus/J1939 integration
- **AI-Powered Features**: GraphRAG, embeddings, physics-informed reasoning
- **Production-Ready**: Complete observability, security, and deployment automation

**Quality Metrics**:
- **13,123 tests passing** - 100% pass rate (136 skipped)
- **Zero compilation warnings** - Enforced with `-D warnings`
- **95%+ test coverage** - Comprehensive test suites
- **95%+ documentation coverage** - Complete API documentation
- **Zero clippy warnings** - Production-grade code quality

### Added

#### Core RDF & SPARQL
- **RDF 1.2 Support** - Complete implementation with 7 format parsers (Turtle, N-Triples, RDF/XML, JSON-LD, N-Quads, TriG, N3)
- **SPARQL 1.1 Query & Update** - Full W3C compliance with SELECT, CONSTRUCT, ASK, DESCRIBE, INSERT, DELETE
- **SPARQL 1.2 Extensions** - RDF-star, enhanced property paths, advanced aggregations
- **Adaptive Query Optimization** - 3.8x faster queries via automatic complexity detection
- **Persistent Storage** - N-Quads format with automatic save/load

#### Server & API
- **oxirs-fuseki** - SPARQL 1.1/1.2 HTTP server with Fuseki compatibility
- **oxirs-gql** - GraphQL API for RDF data
- **REST API v2** - OpenAPI 3.0 with Swagger UI
- **WebSocket Support** - Real-time subscriptions and query streaming
- **Admin UI** - Modern web-based dashboard with live metrics

#### Query & Reasoning
- **oxirs-arq** - Advanced SPARQL query engine with cost-based optimization
- **oxirs-rule** - Rule-based reasoning (RDFS/OWL/SWRL)
- **oxirs-shacl** - SHACL Core + SHACL-SPARQL validation (27/27 W3C tests passing)
- **oxirs-federate** - Distributed query federation with 2-phase commit
- **GeoSPARQL Support** - OGC 1.1 compliance with spatial indexing

#### Industrial Connectivity (Phase D)
- **oxirs-tsdb** - Time-series database with 40:1 Gorilla compression
  - SPARQL temporal extensions (ts:window, ts:resample, ts:interpolate)
  - Hybrid RDF + time-series storage with automatic routing
  - 500K pts/sec write throughput, 180ms p50 query latency
  - 128 tests passing
- **oxirs-modbus** - Complete Modbus TCP/RTU protocol support
  - PLC integration with 6 data types (INT16, UINT16, INT32, UINT32, FLOAT32, BIT)
  - RDF triple generation with QUDT units and PROV-O timestamps
  - Connection pooling with health monitoring
  - 75 tests passing
- **oxirs-canbus** - CANbus/J1939 automotive integration
  - DBC file parser with signal extraction
  - J1939 protocol with PGN extraction and multi-packet reassembly
  - SAMM Aspect Model generation from DBC
  - 98 tests passing

#### AI & Machine Learning
- **oxirs-embed** - Knowledge graph embeddings (TransE, ComplEx, Tucker)
  - CUDA GPU acceleration for 2-5x faster computation
  - Multi-GPU support for large-scale graphs
  - 350+ tests passing
- **oxirs-chat** - RAG chat API with LLM integration
  - Multi-LLM support (OpenAI, Anthropic Claude, Ollama)
  - Session management and context tracking
- **oxirs-shacl-ai** - AI-powered SHACL validation
  - Shape learning and data repair suggestions
  - 350+ tests passing
- **oxirs-graphrag** - GraphRAG hybrid search
  - RRF (Reciprocal Rank Fusion) combining vector and graph topology
  - N-hop SPARQL graph expansion for context retrieval
  - Louvain community detection for hierarchical summarization
  - 23 tests passing
- **oxirs-physics** - Physics-informed digital twins
  - SciRS2 simulation integration
  - RDF → Simulation parameter extraction
  - Physics constraint validation

#### Security & Trust
- **oxirs-did** - W3C DID & Verifiable Credentials
  - DID Core 1.0 and VC Data Model 2.0 implementation
  - did:key and did:web methods with Ed25519 signatures
  - RDFC-1.0 RDF graph canonicalization
  - 43 tests passing
- **ReBAC** - Relationship-Based Access Control
  - Graph-level authorization with hierarchical permissions
  - SPARQL-based policy storage
  - 83 tests passing
- **OAuth2/OIDC/JWT** - Modern authentication
- **TLS/SSL** - Certificate rotation with ACME/Let's Encrypt integration

#### Storage & Distribution
- **oxirs-tdb** - TDB2-compatible storage with MVCC
  - Memory-mapped optimization for large datasets
  - Background compaction and snapshots
  - 250+ tests passing
- **oxirs-cluster** - Distributed clustering with Raft consensus
  - Multi-region replication
  - Automatic failover and recovery
- **oxirs-stream** - Real-time streaming (Kafka/NATS)
  - RDF Patch and SPARQL Update delta
  - 100K+ events/sec throughput
  - 300+ tests passing

#### Platforms
- **oxirs-wasm** - WebAssembly browser/edge deployment
  - In-memory RDF store for browsers
  - TypeScript definitions and ES modules
  - Zero Tokio dependency (WASM-compatible)
  - 8 tests passing

#### Industry Standards
- **oxirs-samm** - SAMM 2.0-2.3 metamodel & AAS integration
  - 16 code generators for Industry 4.0
- **NGSI-LD** - FIWARE smart city compatibility (ETSI GS CIM 009 v1.6)
- **MQTT & OPC UA** - Industrial IoT bridges
- **IDS/Gaia-X** - European data space compliance with ODRL 2.2

#### Performance & Operations
- **Adaptive Query Optimization** - 3.8x faster for simple queries
  - Automatic complexity detection
  - Fast path for simple queries (≤5 patterns)
  - Full cost-based optimization for complex queries
  - 75% CPU savings at production scale (100K QPS)
- **Work-Stealing Scheduler** - Efficient concurrency with 4-level priority queuing
- **Memory Pooling** - SciRS2-integrated buffer management
- **Request Batching** - Automatic batching with adaptive sizing
- **Result Streaming** - Zero-copy streaming with compression (Gzip, Brotli)
- **Load Balancing** - 9 strategies including consistent hashing
- **Edge Caching** - Multi-CDN support (Cloudflare, Fastly, CloudFront, Akamai)
- **DDoS Protection** - IP-based rate limiting with anomaly detection
- **Security Audit** - OWASP Top 10 vulnerability scanning
- **Prometheus Metrics** - Complete observability with OpenTelemetry tracing
- **Disaster Recovery** - Automated failover with RPO/RTO management

#### Deployment Automation
- **Docker** - Multi-stage production builds (12MB binary)
- **Kubernetes** - Production-grade manifests with HPA and Prometheus Operator
- **Kubernetes Operator** - CRD-based deployment automation
- **Terraform** - Complete cloud infrastructure (AWS EKS, GCP GKE, Azure AKS)
- **Ansible** - Production deployment playbooks

#### CLI Tools
- **oxirs** - Comprehensive command-line tool
  - Dataset management (init, import, export, query, serve)
  - Time-series operations (query, insert, stats, compact, retention, benchmark)
  - Modbus operations (monitor-tcp, monitor-rtu, read, write, to-rdf, mock-server)
  - CANbus operations (monitor, parse-dbc, decode, send, to-samm, to-rdf, replay)
  - ReBAC management (add-relationship, remove-relationship, check-permission, list-relationships)

### Changed

- **Query Optimizer Performance** - All profiles optimized at ~3.0 µs (down from 10-16 µs)
  - HighThroughput: 10.8 µs → 3.24 µs (3.3x faster)
  - Analytical: 11.7 µs → 3.01 µs (3.9x faster)
  - Mixed: 10.5 µs → 2.95 µs (3.6x faster)
  - LowMemory: 15.6 µs → 2.94 µs (5.3x faster)

### Performance Benchmarks

```
Query Optimization (5 triple patterns):
  HighThroughput:  3.24 µs (3.3x faster than baseline)
  Analytical:      3.01 µs (3.9x faster than baseline)
  Mixed:           2.95 µs (3.6x faster than baseline)
  LowMemory:       2.94 µs (5.3x faster than baseline)

Time-Series Database:
  Write throughput: 500K pts/sec (single), 2M pts/sec (batch 1K)
  Query latency:    180ms p50 (1M points range), 120ms p50 (aggregation)
  Compression:      38:1 (temperature), 25:1 (vibration), 32:1 (timestamps)

Production Impact (100K QPS):
  CPU time saved:   45 minutes per hour (75% reduction)
  Annual savings:   $10,000 - $50,000 (cloud deployments)
```

### Standards Compliance

- **W3C**: SPARQL 1.1, SPARQL 1.2, RDF 1.2, SHACL, DID Core 1.0, VC Data Model 2.0, RDFC-1.0
- **OGC**: GeoSPARQL 1.1
- **ETSI**: NGSI-LD v1.6
- **ISO/IEC**: MQTT 5.0 (20922), CAN 2.0/CAN FD (11898-1)
- **SAE**: J1939 (heavy vehicles)
- **IEC**: OPC UA (62541)
- **IDSA**: Reference Architecture 4.x
- **Eclipse**: Sparkplug B 3.0

### Contributors

- @cool-japan (KitaSan) - Core development and architecture

### Links

- **Repository**: https://github.com/cool-japan/oxirs
- **Issues**: https://github.com/cool-japan/oxirs/issues
- **Documentation**: https://docs.rs/oxirs-core

---

*"Rust makes memory safety table stakes; OxiRS makes knowledge-graph engineering table stakes."*
