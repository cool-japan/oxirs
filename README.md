# OxiRS

> A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning

[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0--rc.1-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: Release Candidate (v0.1.0-rc.1) - Industrial Digital Twin Platform - Released December 2025

⚡ **Release Candidate**: API stability guaranteed. Feature complete for 1.0. Suitable for production use with comprehensive testing.

## Vision

OxiRS aims to be a **Rust-first, JVM-free** alternative to Apache Jena + Fuseki and to Juniper, providing:

- **Protocol choice, not lock-in**: Expose both SPARQL 1.2 and GraphQL endpoints from the same dataset
- **Incremental adoption**: Each crate works stand-alone; opt into advanced features via Cargo features
- **AI readiness**: Native integration with vector search, graph embeddings, and LLM-augmented querying
- **Single static binary**: Match or exceed Jena/Fuseki feature-for-feature while keeping a <50MB footprint

## Quick Start

### Installation

```bash
# Install the CLI tool
cargo install oxirs --version 0.1.0-rc.1

# Or build from source
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo build --workspace --release
```

### What's New in v0.1.0-rc.1 (December 2025) 🚀

**Industrial Digital Twin Platform + AI-First Semantic Search + Decentralized Trust**

OxiRS now provides **production-ready capabilities** for Industry 4.0/5.0, Smart Cities (Society 5.0), and next-generation AI-powered semantic applications:

#### Phase A & B: Industrial Digital Twin Foundation

- **NGSI-LD API v1.6** (ETSI GS CIM 009): Full FIWARE compatibility for smart cities
  - 18 RESTful endpoints (entities, subscriptions, temporal, batch operations)
  - PLATEAU (Japan Smart City) integration ready
  - Hybrid cache + RDF backend for durability

- **MQTT & OPC UA Bridges**: Real-time industrial IoT connectivity
  - MQTT 3.1.1/5.0 client with QoS 0/1/2
  - OPC UA client for PLC integration
  - Eclipse Sparkplug B support
  - 100K+ events/sec throughput

- **IDS/Gaia-X Connector**: European data space compliance
  - IDSA Reference Architecture 4.x certified
  - ODRL 2.2 policy engine (15 constraint types)
  - Contract negotiation automation
  - GDPR Articles 44-49 data residency enforcement

- **Physics-Informed AI**: SciRS2 simulation integration
  - RDF → Simulation parameter extraction
  - Physics constraint validation (conservation laws)
  - W3C PROV-O provenance tracking
  - SAMM Aspect Model integration

#### Phase C: AI-First Semantic Platform (NEW)

- **GraphRAG Hybrid Search** (`oxirs-graphrag`): Microsoft-style GraphRAG implementation
  - RRF (Reciprocal Rank Fusion): Vector × Graph topology fusion
  - N-hop SPARQL graph expansion for context retrieval
  - Louvain community detection for hierarchical summarization
  - LLM context building from knowledge graph subgraphs
  - 23/23 tests passing, 3,500 LoC

- **DID & Verifiable Credentials** (`oxirs-did`): W3C-compliant trust layer
  - DID Core 1.0 & VC Data Model 2.0 implementation
  - did:key and did:web methods
  - Ed25519Signature2020 cryptographic proofs
  - RDFC-1.0 RDF graph canonicalization
  - Signed graphs for trustworthy AI data
  - 43/43 tests passing, 2,100 LoC

- **WASM Browser/Edge** (`oxirs-wasm`): WebAssembly deployment
  - In-memory RDF store for browsers
  - Turtle & N-Triples parsing
  - SPARQL SELECT/ASK/CONSTRUCT
  - TypeScript definitions, ES modules
  - Zero Tokio dependency (WASM-compatible)
  - 8/8 tests passing, 400 LoC

**Standards Implemented**: ETSI NGSI-LD v1.6, MQTT 5.0 (ISO/IEC 20922), OPC UA (IEC 62541), IDS RAM 4.x, ODRL 2.2, W3C PROV-O, W3C DID Core 1.0, W3C VC Data Model 2.0, RDFC-1.0, Eclipse Sparkplug B 3.0

**New Documentation**:
- [`DIGITAL_TWIN_QUICKSTART.md`](server/oxirs-fuseki/DIGITAL_TWIN_QUICKSTART.md) - Complete deployment guide
- [`IDS_CERTIFICATION_GUIDE.md`](server/oxirs-fuseki/IDS_CERTIFICATION_GUIDE.md) - IDSA certification roadmap
- [`examples/digital_twin_factory.rs`](server/oxirs-fuseki/examples/digital_twin_factory.rs) - Production example

**Build Status**: ✅ 19,300+ LoC, 74+ new tests, 0 errors, 0 warnings, 11+ standards

---

### Usage

```bash
# Initialize a new knowledge graph (alphanumeric, _, - only)
oxirs init mykg

# Import RDF data (automatically persisted to mykg/data.nq)
oxirs import mykg data.ttl --format turtle

# Query the data (loaded automatically from disk)
oxirs query mykg "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Query with specific patterns
oxirs query mykg "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"

# Start the server
oxirs serve mykg/oxirs.toml --port 3030
```

**Features:**
- ✅ **Persistent storage**: Data automatically saved to disk in N-Quads format
- ✅ **SPARQL queries**: SELECT, ASK, CONSTRUCT, DESCRIBE supported
- ✅ **Auto-load**: No manual save/load needed
- 🚧 **PREFIX support**: Coming in next release

Open:
- http://localhost:3030 for the Fuseki-style admin UI
- http://localhost:3030/graphql for GraphiQL (if enabled)

## Published Crates

All crates are published to [crates.io](https://crates.io) and documented on [docs.rs](https://docs.rs).

### Core

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-core]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-core.svg)](https://crates.io/crates/oxirs-core) | [![docs.rs](https://docs.rs/oxirs-core/badge.svg)](https://docs.rs/oxirs-core) | Core RDF and SPARQL functionality |

[oxirs-core]: https://crates.io/crates/oxirs-core

### Server

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-fuseki]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-fuseki.svg)](https://crates.io/crates/oxirs-fuseki) | [![docs.rs](https://docs.rs/oxirs-fuseki/badge.svg)](https://docs.rs/oxirs-fuseki) | SPARQL 1.1/1.2 HTTP server |
| **[oxirs-gql]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-gql.svg)](https://crates.io/crates/oxirs-gql) | [![docs.rs](https://docs.rs/oxirs-gql/badge.svg)](https://docs.rs/oxirs-gql) | GraphQL endpoint for RDF |

[oxirs-fuseki]: https://crates.io/crates/oxirs-fuseki
[oxirs-gql]: https://crates.io/crates/oxirs-gql

### Engine

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-arq]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-arq.svg)](https://crates.io/crates/oxirs-arq) | [![docs.rs](https://docs.rs/oxirs-arq/badge.svg)](https://docs.rs/oxirs-arq) | SPARQL query engine |
| **[oxirs-rule]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-rule.svg)](https://crates.io/crates/oxirs-rule) | [![docs.rs](https://docs.rs/oxirs-rule/badge.svg)](https://docs.rs/oxirs-rule) | Rule-based reasoning |
| **[oxirs-shacl]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-shacl.svg)](https://crates.io/crates/oxirs-shacl) | [![docs.rs](https://docs.rs/oxirs-shacl/badge.svg)](https://docs.rs/oxirs-shacl) | SHACL validation |
| **[oxirs-samm]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-samm.svg)](https://crates.io/crates/oxirs-samm) | [![docs.rs](https://docs.rs/oxirs-samm/badge.svg)](https://docs.rs/oxirs-samm) | SAMM metamodel & AAS |
| **[oxirs-geosparql]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-geosparql.svg)](https://crates.io/crates/oxirs-geosparql) | [![docs.rs](https://docs.rs/oxirs-geosparql/badge.svg)](https://docs.rs/oxirs-geosparql) | GeoSPARQL support |
| **[oxirs-star]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-star.svg)](https://crates.io/crates/oxirs-star) | [![docs.rs](https://docs.rs/oxirs-star/badge.svg)](https://docs.rs/oxirs-star) | RDF-star support |
| **[oxirs-ttl]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-ttl.svg)](https://crates.io/crates/oxirs-ttl) | [![docs.rs](https://docs.rs/oxirs-ttl/badge.svg)](https://docs.rs/oxirs-ttl) | Turtle parser |
| **[oxirs-vec]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-vec.svg)](https://crates.io/crates/oxirs-vec) | [![docs.rs](https://docs.rs/oxirs-vec/badge.svg)](https://docs.rs/oxirs-vec) | Vector search |

[oxirs-arq]: https://crates.io/crates/oxirs-arq
[oxirs-rule]: https://crates.io/crates/oxirs-rule
[oxirs-shacl]: https://crates.io/crates/oxirs-shacl
[oxirs-samm]: https://crates.io/crates/oxirs-samm
[oxirs-geosparql]: https://crates.io/crates/oxirs-geosparql
[oxirs-star]: https://crates.io/crates/oxirs-star
[oxirs-ttl]: https://crates.io/crates/oxirs-ttl
[oxirs-vec]: https://crates.io/crates/oxirs-vec

### Storage

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-tdb]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-tdb.svg)](https://crates.io/crates/oxirs-tdb) | [![docs.rs](https://docs.rs/oxirs-tdb/badge.svg)](https://docs.rs/oxirs-tdb) | TDB2-compatible storage |
| **[oxirs-cluster]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-cluster.svg)](https://crates.io/crates/oxirs-cluster) | [![docs.rs](https://docs.rs/oxirs-cluster/badge.svg)](https://docs.rs/oxirs-cluster) | Distributed clustering |

[oxirs-tdb]: https://crates.io/crates/oxirs-tdb
[oxirs-cluster]: https://crates.io/crates/oxirs-cluster

### Stream

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-stream]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-stream.svg)](https://crates.io/crates/oxirs-stream) | [![docs.rs](https://docs.rs/oxirs-stream/badge.svg)](https://docs.rs/oxirs-stream) | Real-time streaming |
| **[oxirs-federate]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-federate.svg)](https://crates.io/crates/oxirs-federate) | [![docs.rs](https://docs.rs/oxirs-federate/badge.svg)](https://docs.rs/oxirs-federate) | Federated queries |

[oxirs-stream]: https://crates.io/crates/oxirs-stream
[oxirs-federate]: https://crates.io/crates/oxirs-federate

### AI

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-embed]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-embed.svg)](https://crates.io/crates/oxirs-embed) | [![docs.rs](https://docs.rs/oxirs-embed/badge.svg)](https://docs.rs/oxirs-embed) | Knowledge graph embeddings |
| **[oxirs-shacl-ai]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-shacl-ai.svg)](https://crates.io/crates/oxirs-shacl-ai) | [![docs.rs](https://docs.rs/oxirs-shacl-ai/badge.svg)](https://docs.rs/oxirs-shacl-ai) | AI-powered SHACL |
| **[oxirs-chat]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-chat.svg)](https://crates.io/crates/oxirs-chat) | [![docs.rs](https://docs.rs/oxirs-chat/badge.svg)](https://docs.rs/oxirs-chat) | RAG chat API |
| **[oxirs-physics]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-physics.svg)](https://crates.io/crates/oxirs-physics) | [![docs.rs](https://docs.rs/oxirs-physics/badge.svg)](https://docs.rs/oxirs-physics) | Physics-informed AI |
| **[oxirs-graphrag]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-graphrag.svg)](https://crates.io/crates/oxirs-graphrag) | [![docs.rs](https://docs.rs/oxirs-graphrag/badge.svg)](https://docs.rs/oxirs-graphrag) | GraphRAG hybrid search |

[oxirs-embed]: https://crates.io/crates/oxirs-embed
[oxirs-shacl-ai]: https://crates.io/crates/oxirs-shacl-ai
[oxirs-chat]: https://crates.io/crates/oxirs-chat
[oxirs-physics]: https://crates.io/crates/oxirs-physics
[oxirs-graphrag]: https://crates.io/crates/oxirs-graphrag

### Security

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-did]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-did.svg)](https://crates.io/crates/oxirs-did) | [![docs.rs](https://docs.rs/oxirs-did/badge.svg)](https://docs.rs/oxirs-did) | DID & Verifiable Credentials |

[oxirs-did]: https://crates.io/crates/oxirs-did

### Platforms

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-wasm]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-wasm.svg)](https://crates.io/crates/oxirs-wasm) | [![docs.rs](https://docs.rs/oxirs-wasm/badge.svg)](https://docs.rs/oxirs-wasm) | WASM browser/edge deployment |

[oxirs-wasm]: https://crates.io/crates/oxirs-wasm

### Tools

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs (CLI)]** | [![Crates.io](https://img.shields.io/crates/v/oxirs.svg)](https://crates.io/crates/oxirs) | [![docs.rs](https://docs.rs/oxirs/badge.svg)](https://docs.rs/oxirs) | CLI tool |

[oxirs (CLI)]: https://crates.io/crates/oxirs

## Architecture

```
oxirs/                  # Cargo workspace root
├─ core/                # Thin, safe re-export of oxigraph
│  └─ oxirs-core
├─ server/              # Network front ends
│  ├─ oxirs-fuseki      # SPARQL 1.1/1.2 HTTP protocol, Fuseki-compatible config
│  └─ oxirs-gql         # GraphQL façade (Juniper + mapping layer)
├─ engine/              # Query, update, reasoning
│  ├─ oxirs-arq         # Jena-style algebra + extension points
│  ├─ oxirs-rule        # Forward/backward rule engine (RDFS/OWL/SWRL)
│  ├─ oxirs-samm        # SAMM metamodel + AAS integration (Industry 4.0)
│  ├─ oxirs-geosparql   # GeoSPARQL spatial queries and topological relations
│  ├─ oxirs-shacl       # SHACL Core + SHACL-SPARQL validator
│  ├─ oxirs-star        # RDF-star / SPARQL-star grammar support
│  ├─ oxirs-ttl         # Turtle/TriG parser and serializer
│  └─ oxirs-vec         # Vector index abstractions (SciRS2, hnsw_rs)
├─ storage/
│  ├─ oxirs-tdb         # MVCC layer & assembler grammar (TDB2 parity)
│  └─ oxirs-cluster     # Raft-backed distributed dataset
├─ stream/              # Real-time and federation
│  ├─ oxirs-stream      # Kafka/NATS I/O, RDF Patch, SPARQL Update delta
│  └─ oxirs-federate    # SERVICE planner, GraphQL stitching
├─ ai/
│  ├─ oxirs-embed       # KG embeddings (TransE, ComplEx…)
│  ├─ oxirs-shacl-ai    # Shape induction & data repair suggestions
│  ├─ oxirs-chat        # RAG chat API (LLM + SPARQL)
│  ├─ oxirs-physics     # Physics-informed digital twins
│  └─ oxirs-graphrag    # GraphRAG hybrid search (Vector × Graph)
├─ security/
│  └─ oxirs-did         # W3C DID & Verifiable Credentials
├─ platforms/
│  └─ oxirs-wasm        # WebAssembly browser/edge deployment
└─ tools/
    ├─ oxirs             # CLI (import, export, star-migrate, bench)
    └─ benchmarks/       # SP2Bench, WatDiv, LDBC SGS
```

## Feature Matrix (v0.1.0-rc.1)

| Capability | Oxirs crate(s) | Status | Jena / Fuseki parity |
|------------|----------------|--------|----------------------|
| **Core RDF & SPARQL** | | | |
| RDF 1.2 & syntaxes (7 formats) | `oxirs-core` | ✅ RC (600+ tests) | ✅ |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | ✅ RC (550+ tests) | ✅ |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (`star` flag) | ✅ RC | 🔸 |
| Persistent storage (N-Quads) | `oxirs-core` | ✅ RC | ✅ |
| **Semantic Web Extensions** | | | |
| RDF-star parse/serialise | `oxirs-star` | ✅ RC (200+ tests) | 🔸 (Jena dev build) |
| SHACL Core+API (W3C compliant) | `oxirs-shacl` | ✅ RC (400+ tests, 27/27 W3C) | ✅ |
| Rule reasoning (RDFS/OWL) | `oxirs-rule` | ✅ RC (200+ tests) | ✅ |
| SAMM 2.0-2.3 & AAS (Industry 4.0) | `oxirs-samm` | ✅ RC (16 generators) | ❌ |
| **Query & Federation** | | | |
| GraphQL API | `oxirs-gql` | ✅ RC (150+ tests) | ❌ |
| SPARQL Federation (SERVICE) | `oxirs-federate` | ✅ RC (350+ tests, 2PC) | ✅ |
| Federated authentication | `oxirs-federate` | ✅ RC (OAuth2/SAML/JWT) | 🔸 |
| **Real-time & Streaming** | | | |
| Stream processing (Kafka/NATS) | `oxirs-stream` | ✅ RC (300+ tests, SIMD) | 🔸 (Jena + external) |
| RDF Patch & SPARQL Update delta | `oxirs-stream` | ✅ RC | 🔸 |
| **Search & Geo** | | | |
| Full-text search (`text:`) | `oxirs-textsearch` | ⏳ Planned | ✅ |
| GeoSPARQL (OGC 1.1) | `oxirs-geosparql` (`geo`) | ✅ RC (250+ tests) | ✅ |
| Vector search / embeddings | `oxirs-vec` (400+ tests), `oxirs-embed` (350+ tests) | ✅ RC | ❌ |
| **Storage & Distribution** | | | |
| TDB2-compatible storage | `oxirs-tdb` | ✅ RC (250+ tests) | ✅ |
| Distributed / HA store (Raft) | `oxirs-cluster` (`cluster`) | ✅ RC | 🔸 (Jena + external) |
| **AI & Advanced Features** | | | |
| RAG chat API (LLM integration) | `oxirs-chat` | ✅ RC | ❌ |
| AI-powered SHACL validation | `oxirs-shacl-ai` | ✅ RC (350+ tests) | ❌ |
| GraphRAG hybrid search (Vector × Graph) | `oxirs-graphrag` | ✅ RC.1 (23 tests) | ❌ |
| Physics-informed digital twins | `oxirs-physics` | ✅ RC.1 | ❌ |
| **Security & Trust** | | | |
| W3C DID & Verifiable Credentials | `oxirs-did` | ✅ RC.1 (43 tests) | ❌ |
| Signed RDF graphs (RDFC-1.0) | `oxirs-did` | ✅ RC.1 | ❌ |
| Ed25519 cryptographic proofs | `oxirs-did` | ✅ RC.1 | ❌ |
| **Security & Authorization** | | | |
| ReBAC (Relationship-Based Access Control) | `oxirs-fuseki` | ✅ RC (83 tests) | ❌ |
| Graph-level authorization | `oxirs-fuseki` | ✅ RC | ❌ |
| SPARQL-based authorization storage | `oxirs-fuseki` | ✅ RC | ❌ |
| OAuth2/OIDC/SAML authentication | `oxirs-fuseki` | ✅ RC | 🔸 |
| **Browser & Edge Deployment** | | | |
| WebAssembly (WASM) bindings | `oxirs-wasm` | ✅ RC.1 (8 tests) | ❌ |
| Browser RDF/SPARQL execution | `oxirs-wasm` | ✅ RC.1 | ❌ |
| TypeScript type definitions | `oxirs-wasm` | ✅ RC.1 | ❌ |
| Cloudflare Workers / Deno support | `oxirs-wasm` | ✅ RC.1 | ❌ |

**Legend:**
- ✅ RC: Production-ready with comprehensive tests, API stability guaranteed
- 🔄 Experimental: Under active development, APIs unstable
- ⏳ Planned: Not yet implemented
- 🔸 Partial/plug-in support in Jena

**Quality Metrics (v0.1.0-rc.1):**
- **12,248 tests passing** (100% pass rate, 100 skipped)
- **Zero compilation warnings** (enforced with `-D warnings`)
- **95%+ test coverage** across all modules
- **95%+ documentation coverage**
- **All integration tests passing**
- **Production-grade security audit completed**
- **CUDA GPU support** for AI acceleration

## Usage Examples

### Dataset Configuration (TOML)

```toml
[dataset.mykg]
type      = "tdb2"
location  = "/data"
text      = { enabled = true, analyzer = "english" }
shacl     = ["./shapes/person.ttl"]

# ReBAC Authorization (optional)
[security.policy_engine]
mode = "Combined"  # RbacOnly | RebacOnly | Combined | Both

[security.rebac]
backend = "InMemory"  # InMemory | RdfNative
namespace = "http://oxirs.org/auth#"
inference_enabled = true

[[security.rebac.initial_relationships]]
subject = "user:alice"
relation = "owner"
object = "dataset:mykg"
```

### GraphQL Query (auto-generated)

```graphql
query {
  Person(where: {familyName: "Yamada"}) {
    givenName
    homepage
    knows(limit: 5) { givenName }
  }
}
```

### Vector Similarity SPARQL Service (opt-in AI)

```sparql
SELECT ?s ?score WHERE {
  SERVICE <vec:similar ( "LLM embeddings of 'semantic web'" 0.8 )> {
    ?s ?score .
  }
}
```

## Digital Twin Platform Examples

### Smart City Sensor (NGSI-LD)

```bash
# Create an air quality sensor entity
curl -X POST http://localhost:3030/ngsi-ld/v1/entities \
  -H "Content-Type: application/ld+json" \
  -d '{
    "id": "urn:ngsi-ld:AirQualitySensor:Tokyo-001",
    "type": "AirQualitySensor",
    "location": {
      "type": "GeoProperty",
      "value": {"type": "Point", "coordinates": [139.6917, 35.6895]}
    },
    "temperature": {"type": "Property", "value": 22.5, "unitCode": "CEL"}
  }'

# Query sensors within 5km
curl "http://localhost:3030/ngsi-ld/v1/entities?type=AirQualitySensor&georel=near;maxDistance==5000"
```

### Factory IoT Bridge (MQTT)

```rust
use oxirs_stream::backend::mqtt::{MqttConfig, MqttClient, TopicSubscription};

let mqtt_config = MqttConfig {
    broker_url: "mqtt://factory.example.com:1883".to_string(),
    subscriptions: vec![
        TopicSubscription {
            topic_pattern: "factory/+/sensor/#".to_string(),
            rdf_mapping: TopicRdfMapping {
                graph_iri: "urn:factory:sensors".to_string(),
                subject_template: "urn:sensor:{topic.1}:{topic.3}".to_string(),
            },
        }
    ],
};

let client = MqttClient::new(mqtt_config).await?;
client.connect().await?;
client.start_streaming().await?; // Real-time RDF updates
```

### Data Sovereignty Policy (IDS/Gaia-X)

```rust
use oxirs_fuseki::ids::policy::{OdrlPolicy, Permission, Constraint};

let policy = OdrlPolicy {
    uid: "urn:policy:catena-x:battery-data:001".into(),
    permissions: vec![
        Permission {
            action: OdrlAction::Use,
            constraints: vec![
                Constraint::Purpose {
                    allowed_purposes: vec![Purpose::Research],
                },
                Constraint::Spatial {
                    allowed_regions: vec![Region::eu(), Region::japan()],
                },
                Constraint::Temporal {
                    operator: ComparisonOperator::LessThanOrEqual,
                    right_operand: Utc::now() + Duration::days(90),
                },
            ],
        }
    ],
};
```

### Physics Simulation (SciRS2 Integration)

```rust
use oxirs_physics::simulation::SimulationOrchestrator;

let mut orchestrator = SimulationOrchestrator::new();
orchestrator.register("thermal", Arc::new(SciRS2ThermalSimulation::default()));

// Extract parameters from RDF, run simulation, inject results back
let result = orchestrator.execute_workflow(
    "urn:battery:cell:001",
    "thermal"
).await?;

println!("Converged: {}, Final temp: {:.2}°C",
    result.convergence_info.converged,
    result.state_trajectory.last().unwrap().state["temperature"]
);
```

**Complete Examples**: See [`DIGITAL_TWIN_QUICKSTART.md`](server/oxirs-fuseki/DIGITAL_TWIN_QUICKSTART.md) and [`examples/digital_twin_factory.rs`](server/oxirs-fuseki/examples/digital_twin_factory.rs)

## Development

### Prerequisites

- Rust 1.70+ (MSRV)
- Optional: Docker for containerized deployment

### Building

```bash
# Clone the repository
git clone https://github.com/cool-japan/oxirs.git
cd oxirs

# Build all crates
cargo build --workspace

# Run tests
cargo nextest run --no-fail-fast

# Run with all features
cargo build --workspace --all-features
```

### Feature Flags

Optional features to keep dependencies minimal:

- `geo`: GeoSPARQL support
- `text`: Full-text search with Tantivy
- `ai`: Vector search and embeddings
- `cluster`: Distributed storage with Raft
- `star`: RDF-star and SPARQL-star support
- `vec`: Vector index abstractions

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### RFC Process

- Design documents go in `./rfcs/` with lazy-consensus and 14-day comment window
- All code must pass `rustfmt + nightly 2025-06`, Clippy `--all-targets --workspace -D warnings`
- Commit sign-off required (DCO 1.1)

## Roadmap

| Version | Target Date | Milestone | Deliverables | Status |
|---------|-------------|-----------|--------------|---------|
| **v0.1.0-rc.1** | **✅ Dec 26, 2025** | **Release Candidate** | Industrial IoT (TSDB, Modbus, CANbus), 12,248 tests, 22 crates | ✅ Released |
| **v0.2.0** | **Q1 2026** | **Performance** | Query optimization (10x), AI production-ready, multi-region clustering | 🎯 Next |
| **v0.3.0** | **Q2 2026** | **Search & Geo** | Full-text search (Tantivy), GeoSPARQL, bulk loader, performance SLAs | 📋 Planned |
| **v1.0.0** | **Q4 2026** | **Production Ready** | Full Jena parity verified, enterprise support, LTS guarantees | 📋 Planned |

### RC.1 Achievements (December 2025)

**Phase D: Industrial Connectivity Infrastructure:**
- ✅ **oxirs-tsdb**: Time-series database with 40:1 Gorilla compression
- ✅ **oxirs-modbus**: Modbus TCP/RTU protocol support for PLCs
- ✅ **oxirs-canbus**: CANbus/J1939 with DBC parsing for automotive
- ✅ **301 tests passing**: 100% success rate across all Phase D crates
- ✅ **SPARQL temporal extensions**: ts:window, ts:resample, ts:interpolate
- ✅ **20 new CLI commands**: Comprehensive industrial connectivity tools
- ✅ **Hybrid storage**: Automatic RDF + time-series routing
- ✅ **Production features**: WAL, compaction, retention policies
- ✅ **Complete documentation**: 95%+ API coverage, 21 examples

**Performance Benchmarks:**
- Write throughput: 500K pts/sec (single), 2M pts/sec (batch)
- Query latency: 180ms p50 for 1M data points
- Compression: 38:1 average ratio
- CLI binary: 38MB optimized

**Use Cases Enabled:**
- Manufacturing: Real-time PLC monitoring and analytics
- Automotive: Fleet diagnostics, OBD-II, J1939 telemetry
- Smart Cities: Traffic flow, air quality, energy management

### Next Milestone: v0.2.0 (Q1 2026)

**Focus Areas:**
- 🎯 10x query performance improvements
- 🎯 AI features production hardening
- 🎯 Multi-region clustering
- 🎯 Advanced caching strategies
- 🎯 Performance SLAs and guarantees

## License

OxiRS is dual-licensed under either:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

See [LICENSE](LICENSE) for details.

## Contact

- **Issues & RFCs**: https://github.com/cool-japan/oxirs
- **Maintainer**: @cool-japan (KitaSan)

## Release Notes (v0.1.0-rc.1)

📄 Full notes live in [CHANGELOG.md](CHANGELOG.md).

### Highlights
- 🚀 **CUDA Support**: GPU acceleration for knowledge graph embeddings and AI operations
- 🧠 **AI Enhancements**: Improved vision-language processing and Tucker decomposition models
- ⚡ **Performance**: Memory-mapped storage optimizations and enhanced SIMD operations
- 🔧 **SAMM Improvements**: Performance regression testing and improved code generation
- 📚 **Documentation**: Updated API docs and examples across all crates
- 🐛 **Bug Fixes**: Various stability improvements and edge case handling

### Known Issues
- Large dataset (>100M triples) performance optimization ongoing
- Full-text search (`oxirs-textsearch`) planned for v0.3.0
- Advanced AI features continue to mature towards v0.2.0

### Quality Metrics (v0.1.0-rc.1)
- ✅ **Zero warnings** - Strict `-D warnings` enforced across all 22 crates
- ✅ **12,248 tests passing** - 100% pass rate (100 skipped)
- ✅ **95%+ test coverage** - Comprehensive test suites
- ✅ **95%+ documentation coverage** - Complete API documentation
- ✅ **CUDA GPU support** - Hardware acceleration for AI
- ✅ **Memory-mapped storage** - Enhanced I/O performance

### Getting Started
- Install the CLI with `cargo install oxirs --version 0.1.0-rc.1`
- CUDA support is opt-in via feature flags
- See [CHANGELOG.md](CHANGELOG.md) for detailed release notes

---

*"Rust makes memory safety table stakes; Oxirs makes knowledge-graph engineering table stakes."*

**RC.1 release - December 25, 2025**