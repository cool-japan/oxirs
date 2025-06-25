# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OxiRS is a Rust-native, modular platform for Semantic Web, SPARQL 1.2, GraphQL, and AI-augmented reasoning. It aims to be a JVM-free alternative to Apache Jena + Fuseki with enhanced capabilities.

## Key Commands

### Build & Test
```bash
# Full build and test cycle (preferred)
./scripts/build.sh

# Build with all features
./scripts/build.sh --all-features

# Run tests continuously until no warnings (following no warnings policy)
cargo nextest run --no-fail-fast

# Linting
cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check
```

### Development
```bash
# Setup development environment (first time)
./scripts/setup-dev.sh

# Run specific module tests
cargo nextest run -p oxirs-core --no-fail-fast

# Run the Fuseki server
cargo run -p oxirs-fuseki -- --config oxirs.toml

# CLI tool
cargo run -p oxide -- <command>
```

## Architecture

### Module Organization
The project uses a Cargo workspace with modules organized by concern:

- **core/**: Foundation modules (oxirs-core provides RDF/SPARQL basics)
- **server/**: HTTP servers (oxirs-fuseki for SPARQL, oxirs-gql for GraphQL)
- **engine/**: Query processors (oxirs-arq, oxirs-rule, oxirs-shacl)
- **storage/**: Persistence layers (oxirs-tdb for disk storage, oxirs-cluster for distributed)
- **stream/**: Real-time processing (oxirs-stream for Kafka/NATS, oxirs-federate for federation)
- **ai/**: ML integration (oxirs-embed for embeddings, oxirs-chat for RAG)
- **tools/**: CLI utilities (oxide command-line tool)

### Key Design Principles
1. **Incremental Adoption**: Each crate works standalone
2. **Protocol Choice**: SPARQL and GraphQL from same dataset
3. **OxiGraph Independence**: Currently extracting and adapting OxiGraph code to eliminate external dependencies
4. **Single Binary Target**: Aiming for <50MB footprint with full Jena/Fuseki feature parity

### Configuration
Uses TOML configuration (`oxirs.toml`) with sections for:
- Server settings (host, port, admin UI)
- Dataset configurations
- Security/authentication
- Feature flags (text search, vector search, RDF-star, clustering, streaming)

## Current Development Focus

Based on TODO.md, the project is actively:
1. Extracting OxiGraph code into oxirs-core to remove external dependencies
2. Implementing missing Jena/Fuseki features for compatibility
3. Building distributed storage with Raft consensus
4. Adding AI capabilities (embeddings, chat, shape learning)

## Testing Strategy

- Use `cargo nextest` exclusively (not `cargo test`)
- Run with `--no-fail-fast` to see all failures
- Each module has its own test suite
- Integration tests in `tests/` directories
- Example datasets in `data/` for testing

## Important Notes

- **File Size Limit**: Refactor files exceeding 2000 lines
- **Crate Versions**: Always use latest versions from crates.io
- **No Warnings**: Code must compile without any warnings
- **Module Independence**: Each crate should be usable standalone
- **Jena Compatibility**: When implementing features, check Apache Jena codebase at ~/work/jena/ for reference
- **GraphQL Implementation**: Use Juniper patterns from ~/work/juniper/ as reference
- **Oxigraph Reference**: Check ~/work/oxigraph/ for RDF/SPARQL implementation patterns