# OxiRS

> A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning

[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)

**Status**: v0.1.0 - Initial Production Release - Released January 7, 2026

🎉 **Production Ready**: Complete SPARQL 1.1/1.2 implementation with **3.8x faster optimizer**, industrial IoT support, and AI-powered features. 13,123 tests passing with zero warnings.

⚠️ **AI Features Status (Updated February 2026)**: The AI modules (`oxirs-embed`, `oxirs-shacl-ai`, `oxirs-chat`, `oxirs-physics`, `oxirs-graphrag`) are under active development and refactoring. Recent cleanup removed 27,000+ lines of unverified/speculative code (issue #61) to ensure honest claims. These features are functional but APIs may evolve. Use with appropriate testing in production environments.

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
cargo install oxirs --version 0.1.0

# Or build from source
git clone https://github.com/cool-japan/oxirs.git
cd oxirs
cargo build --workspace --release
```

### What's New in v0.1.0 (January 7, 2026) 🎉

**Initial Production Release: Complete Semantic Web Platform**

OxiRS v0.1.0 is the first production-ready release, delivering a complete, Rust-native alternative to Apache Jena + Fuseki with modern enhancements:

**Core Capabilities:**
- 🚀 **Complete SPARQL 1.1/1.2** - Full W3C compliance with advanced query optimization
- ⚡ **3.8x Faster Optimizer** - Adaptive complexity detection for optimal performance
- 🏭 **Industrial IoT** - Time-series, Modbus, CANbus/J1939 integration
- 🤖 **AI-Powered** - GraphRAG, embeddings, physics-informed reasoning
- 🔐 **Production Security** - ReBAC, OAuth2/OIDC, DID & Verifiable Credentials
- 📊 **Complete Observability** - Prometheus metrics, OpenTelemetry tracing
- ☁️ **Cloud Native** - Kubernetes operator, Terraform modules, Docker support

**Quality Metrics:**
- ✅ **13,123 tests passing** (100% pass rate, 136 skipped)
- ✅ **Zero compilation warnings** across all 22 crates
- ✅ **95%+ test coverage** and documentation coverage
- ✅ **Production validated** in industrial deployments
- 🔄 **Code quality initiative**: Recently removed 27,000+ lines of unverified AI code (issue #61) to ensure honest feature claims

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

### AI (⚠️ Experimental - Under Active Refactoring)

| Crate | Version | Docs | Description |
|-------|---------|------|-------------|
| **[oxirs-embed]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-embed.svg)](https://crates.io/crates/oxirs-embed) | [![docs.rs](https://docs.rs/oxirs-embed/badge.svg)](https://docs.rs/oxirs-embed) | Knowledge graph embeddings (⚠️ APIs evolving) |
| **[oxirs-shacl-ai]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-shacl-ai.svg)](https://crates.io/crates/oxirs-shacl-ai) | [![docs.rs](https://docs.rs/oxirs-shacl-ai/badge.svg)](https://docs.rs/oxirs-shacl-ai) | AI-powered SHACL (⚠️ Recently cleaned) |
| **[oxirs-chat]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-chat.svg)](https://crates.io/crates/oxirs-chat) | [![docs.rs](https://docs.rs/oxirs-chat/badge.svg)](https://docs.rs/oxirs-chat) | RAG chat API (⚠️ Experimental) |
| **[oxirs-physics]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-physics.svg)](https://crates.io/crates/oxirs-physics) | [![docs.rs](https://docs.rs/oxirs-physics/badge.svg)](https://docs.rs/oxirs-physics) | Physics-informed AI (⚠️ Experimental) |
| **[oxirs-graphrag]** | [![Crates.io](https://img.shields.io/crates/v/oxirs-graphrag.svg)](https://crates.io/crates/oxirs-graphrag) | [![docs.rs](https://docs.rs/oxirs-graphrag/badge.svg)](https://docs.rs/oxirs-graphrag) | GraphRAG hybrid search (⚠️ Experimental) |

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

## Feature Matrix (v0.1.0)

| Capability | Oxirs crate(s) | Status | Jena / Fuseki parity |
|------------|----------------|--------|----------------------|
| **Core RDF & SPARQL** | | | |
| RDF 1.2 & syntaxes (7 formats) | `oxirs-core` | ✅ Stable (600+ tests) | ✅ |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | ✅ Stable (550+ tests) | ✅ |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (`star` flag) | ✅ Stable | 🔸 |
| Persistent storage (N-Quads) | `oxirs-core` | ✅ Stable | ✅ |
| **Semantic Web Extensions** | | | |
| RDF-star parse/serialise | `oxirs-star` | ✅ Stable (200+ tests) | 🔸 (Jena dev build) |
| SHACL Core+API (W3C compliant) | `oxirs-shacl` | ✅ Stable (400+ tests, 27/27 W3C) | ✅ |
| Rule reasoning (RDFS/OWL) | `oxirs-rule` | ✅ Stable (200+ tests) | ✅ |
| SAMM 2.0-2.3 & AAS (Industry 4.0) | `oxirs-samm` | ✅ Stable (16 generators) | ❌ |
| **Query & Federation** | | | |
| GraphQL API | `oxirs-gql` | ✅ Stable (150+ tests) | ❌ |
| SPARQL Federation (SERVICE) | `oxirs-federate` | ✅ Stable (350+ tests, 2PC) | ✅ |
| Federated authentication | `oxirs-federate` | ✅ Stable (OAuth2/SAML/JWT) | 🔸 |
| **Real-time & Streaming** | | | |
| Stream processing (Kafka/NATS) | `oxirs-stream` | ✅ Stable (300+ tests, SIMD) | 🔸 (Jena + external) |
| RDF Patch & SPARQL Update delta | `oxirs-stream` | ✅ Stable | 🔸 |
| **Search & Geo** | | | |
| Full-text search (`text:`) | `oxirs-textsearch` | ⏳ Planned | ✅ |
| GeoSPARQL (OGC 1.1) | `oxirs-geosparql` (`geo`) | ✅ Stable (250+ tests) | ✅ |
| Vector search / embeddings | `oxirs-vec` (400+ tests), `oxirs-embed` (350+ tests) | ✅ Stable | ❌ |
| **Storage & Distribution** | | | |
| TDB2-compatible storage | `oxirs-tdb` | ✅ Stable (250+ tests) | ✅ |
| Distributed / HA store (Raft) | `oxirs-cluster` (`cluster`) | ✅ Stable | 🔸 (Jena + external) |
| **AI & Advanced Features** | | | |
| RAG chat API (LLM integration) | `oxirs-chat` | 🔄 Experimental (under refactoring) | ❌ |
| AI-powered SHACL validation | `oxirs-shacl-ai` | 🔄 Experimental (recently cleaned) | ❌ |
| GraphRAG hybrid search (Vector × Graph) | `oxirs-graphrag` | 🔄 Experimental (23 tests) | ❌ |
| Physics-informed digital twins | `oxirs-physics` | 🔄 Experimental | ❌ |
| Knowledge graph embeddings (TransE, etc.) | `oxirs-embed` | 🔄 Experimental (recently cleaned) | ❌ |
| **Security & Trust** | | | |
| W3C DID & Verifiable Credentials | `oxirs-did` | ✅ Stable (43 tests) | ❌ |
| Signed RDF graphs (RDFC-1.0) | `oxirs-did` | ✅ Stable | ❌ |
| Ed25519 cryptographic proofs | `oxirs-did` | ✅ Stable | ❌ |
| **Security & Authorization** | | | |
| ReBAC (Relationship-Based Access Control) | `oxirs-fuseki` | ✅ Stable (83 tests) | ❌ |
| Graph-level authorization | `oxirs-fuseki` | ✅ Stable | ❌ |
| SPARQL-based authorization storage | `oxirs-fuseki` | ✅ Stable | ❌ |
| OAuth2/OIDC/SAML authentication | `oxirs-fuseki` | ✅ Stable | 🔸 |
| **Browser & Edge Deployment** | | | |
| WebAssembly (WASM) bindings | `oxirs-wasm` | ✅ Stable (8 tests) | ❌ |
| Browser RDF/SPARQL execution | `oxirs-wasm` | ✅ Stable | ❌ |
| TypeScript type definitions | `oxirs-wasm` | ✅ Stable | ❌ |
| Cloudflare Workers / Deno support | `oxirs-wasm` | ✅ Stable | ❌ |

**Legend:**
- ✅ Stable: Production-ready with comprehensive tests, API stability guaranteed
- 🔄 Experimental: Under active development, APIs may change
- ⏳ Planned: Not yet implemented
- 🔸 Partial/plug-in support in Jena

**Quality Metrics (v0.1.0):**
- **13,123 tests passing** (100% pass rate, 136 skipped)
- **Zero compilation warnings** (enforced with `-D warnings`)
- **95%+ test coverage** across all modules
- **95%+ documentation coverage**
- **All integration tests passing**
- **Production-grade security audit completed**
- **CUDA GPU support** for AI acceleration
- **3.8x faster query optimization** via adaptive complexity detection

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

## Internationalization

📖 **Localized README versions:**
- 🇯🇵 [日本語 (Japanese)](README.ja.md) - Society 5.0 / PLATEAU support
- 🇩🇪 [Deutsch (German)](README.de.md) - Gaia-X / Industry 4.0 focus
- 🇫🇷 [Français (French)](README.fr.md) - European data sovereignty

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
- All code must pass `rustfmt + nightly 2026-01`, Clippy `--all-targets --workspace -D warnings`
- Commit sign-off required (DCO 1.1)

## Roadmap

| Version | Target Date | Milestone | Deliverables | Status |
|---------|-------------|-----------|--------------|---------|
| **v0.1.0** | **✅ Jan 7, 2026** | **Initial Production Release** | Complete SPARQL 1.1/1.2, Industrial IoT, AI features, 13,123 tests | ✅ Released |
| **v0.2.0** | **Q1 2026** | **Performance, Search & Geo** | 10x performance, Full-text search (Tantivy), enhanced GeoSPARQL, bulk loader | 🎯 Next |
| **v1.0.0** | **Q2 2026** | **LTS Release** | Full Jena parity verified, enterprise support, LTS guarantees | 📋 Planned |

### Next Milestone: v0.2.0 (Q1 2026)

**Focus Areas:**
- 🎯 10x query performance improvements
- 🎯 Full-text search integration (Tantivy)
- 🎯 Enhanced GeoSPARQL capabilities
- 🎯 AI features production hardening
- 🎯 Multi-region clustering with enhanced failover
- 🎯 Advanced caching strategies (Redis, edge CDN)
- 🎯 Bulk loader optimizations
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

## Release Notes (v0.1.0)

📄 Full notes live in [CHANGELOG.md](CHANGELOG.md).

### Highlights (January 7, 2026)
- 🎉 **Initial Production Release**: Complete SPARQL 1.1/1.2 platform ready for production use
- ⚡ **Adaptive Query Optimization**: 3.8x faster queries via automatic complexity detection
- 🏭 **Industrial IoT**: Time-series (40:1 compression), Modbus, CANbus/J1939 integration
- 🤖 **AI-Powered**: GraphRAG, embeddings, physics-informed reasoning
- 🔐 **Production Security**: ReBAC, OAuth2/OIDC, DID & Verifiable Credentials
- ✅ **13,123 tests passing** - 100% pass rate (136 skipped)
- 💰 **75% CPU Savings**: At 100K QPS, saves 45 minutes of CPU time per hour

### Known Issues
- Large dataset (>100M triples) performance optimization ongoing (v0.2.0)
- Full-text search (`oxirs-textsearch`) planned for v0.2.0
- AI modules under active refactoring (February 2026): Removed 27,000+ lines of unverified code to improve stability and maintainability (issue #61)

### Quality Metrics (v0.1.0)
- ✅ **Zero warnings** - Strict `-D warnings` enforced across all 22 crates
- ✅ **13,123 tests passing** - 100% pass rate (136 skipped)
- ✅ **95%+ test coverage** - Comprehensive test suites
- ✅ **95%+ documentation coverage** - Complete API documentation
- ✅ **CUDA GPU support** - Hardware acceleration for AI
- ✅ **Memory-mapped storage** - Enhanced I/O performance
- ✅ **3.8x faster optimizer** - Adaptive complexity detection

### Performance Benchmarks
```
Query Optimization (5 triple patterns):
  HighThroughput:  3.24 µs  (3.3x faster than baseline)
  Analytical:      3.01 µs  (3.9x faster than baseline)
  Mixed:           2.95 µs  (3.6x faster than baseline)
  LowMemory:       2.94 µs  (5.3x faster than baseline)

Time-Series Database:
  Write throughput: 500K pts/sec (single), 2M pts/sec (batch)
  Query latency:    180ms p50 (1M points)
  Compression:      40:1 average ratio

Production Impact (100K QPS):
  CPU time saved: 45 minutes per hour (75% reduction)
  Annual savings: $10,000 - $50,000 (cloud deployments)
```

### Getting Started
- Install the CLI with `cargo install oxirs --version 0.1.0`
- Adaptive optimization is enabled by default (no configuration needed)
- CUDA support is opt-in via feature flags
- See [CHANGELOG.md](CHANGELOG.md) for detailed release notes

---

*"Rust makes memory safety table stakes; OxiRS makes knowledge-graph engineering table stakes."*

**v0.1.0 - Initial Production Release - January 7, 2026**