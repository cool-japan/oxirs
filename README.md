# OxiRS

> A Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning

[![Rust](https://github.com/cool-japan/oxirs/workflows/Rust/badge.svg)](https://github.com/cool-japan/oxirs/actions)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Vision

OxiRS aims to be a **Rust-first, JVM-free** alternative to Apache Jena + Fuseki, providing:

- **Protocol choice, not lock-in**: Expose both SPARQL 1.2 and GraphQL endpoints from the same dataset
- **Incremental adoption**: Each crate works stand-alone; opt into advanced features via Cargo features
- **AI readiness**: Native integration with vector search, graph embeddings, and LLM-augmented querying
- **Single static binary**: Match or exceed Jena/Fuseki feature-for-feature while keeping a <50MB footprint

## Quick Start

```bash
# Install the CLI tool
cargo install --git https://github.com/cool-japan/oxirs oxide

# Initialize a new knowledge graph
oxide init mykg

# Start the server
oxide serve ./mykg.toml --port 3030
```

Open:
- http://localhost:3030 for the Fuseki-style admin UI
- http://localhost:3030/graphql for GraphiQL

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
│  ├─ oxirs-shacl       # SHACL Core + SHACL-SPARQL validator
│  ├─ oxirs-star        # RDF-star / SPARQL-star grammar support
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
│  └─ oxirs-chat        # RAG chat API (LLM + SPARQL)
└─ tools/
    ├─ oxide             # CLI (import, export, star-migrate, bench)
    └─ benchmarks/       # SP2Bench, WatDiv, LDBC SGS
```

## Feature Matrix

| Capability | Oxirs crate(s) | Status | Jena / Fuseki parity |
|------------|----------------|--------|----------------------|
| RDF 1.2 & syntaxes | `oxirs-core` | ✅ | ✅ |
| RDF-star parse/serialise | `oxirs-star` | ✅ *(draft)* | 🔸 (Jena dev build) |
| SPARQL 1.1 Query & Update | `oxirs-fuseki` + `oxirs-arq` | ✅ | ✅ |
| SPARQL 1.2 / SPARQL-star | `oxirs-arq` (`star` flag) | 🔄 | 🔸 |
| Rule reasoning (RDFS/OWL) | `oxirs-rule` | 🔄 | ✅ |
| SHACL Core+API | `oxirs-shacl` | 🔄 | ✅ |
| Full-text search (`text:`) | `oxirs-textsearch` | 🔄 | ✅ |
| GeoSPARQL | `oxirs-geosparql` (`geo`) | ⏳ | ✅ |
| GraphQL API | `oxirs-gql` | 🔄 | ❌ |
| Vector search / embeddings | `oxirs-vec`, `oxirs-embed` (`ai`) | ⏳ | ❌ |
| Distributed / HA store | `oxirs-cluster` (`cluster`) | ⏳ | 🔸 (Jena + external) |

Legend: ✅ done, 🔄 in progress, ⏳ planned, 🔸 partial/plug-in

## Usage Examples

### Dataset Configuration (TOML)

```toml
[dataset.mykg]
type      = "tdb2"
location  = "/data"
text      = { enabled = true, analyzer = "english" }
shacl     = ["./shapes/person.ttl"]
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

| Quarter | Milestone | Deliverables |
|---------|-----------|--------------|
| **2025 Q3** | **Phase 0** – Boot & Serve | `oxirs-core`, `oxirs-fuseki` with basic Fuseki config, Docker & Helm chart |
| **2025 Q4** | **Phase 1** – GraphQL & RDF-star | `oxirs-gql` (GraphQL-LD mapping), `oxirs-star`, SPARQL 1.2 grammar |
| **2026 Q1** | **Phase 2** – Reasoning & Validation | `oxirs-rule` (forward chain), `oxirs-shacl` with test suite parity |
| **2026 Q2** | **Phase 3** – Text & Geo Extensions | `oxirs-textsearch` (tantivy), `oxirs-geosparql`, bulk-loader CLI |
| **2026 Q3** | **Phase 4** – AI Augmentation | `oxirs-vec`, `oxirs-embed`, `oxirs-chat`; SciRS2 integration |
| **2026 Q4** | **Phase 5** – Streaming & Cluster | `oxirs-stream`, `oxirs-cluster`, Raft demo across 3 nodes |

## License

Dual licensed under MIT OR Apache-2.0 licenses.

## Contact

- **Issues & RFCs**: https://github.com/cool-japan/oxirs
- **Maintainer**: @cool-japan (KitaSan)

---

*"Rust makes memory safety table stakes; Oxirs makes knowledge-graph engineering table stakes."*