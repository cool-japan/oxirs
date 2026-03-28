# OxiRS SHACL 🔍

[![Version](https://img.shields.io/badge/version-0.2.3-blue)](https://github.com/cool-japan/oxirs/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Build Status](https://github.com/cool-japan/oxirs/workflows/CI/badge.svg)](https://github.com/cool-japan/oxirs/actions)

**Status**: Production Release (v0.2.3) - Released March 16, 2026

✨ **Production Release**: Production-ready with API stability guarantees. Semantic versioning enforced.

A high-performance SHACL (Shapes Constraint Language) validator for RDF data, implemented in Rust as part of the OxiRS ecosystem. Currently in active development with core constraint support implemented.

## 🎯 Features

### ✅ Implemented
- **Core SHACL Constraints** - All basic constraint types (class, datatype, cardinality, etc.)
- **Property Path Support** - Sequence, alternative, inverse, and Kleene paths
- **Logical Constraints** - Full support for `sh:and`, `sh:or`, `sh:not`, `sh:xone`
- **Shape-based Constraints** - Nested shape validation and qualified cardinality
- **Target Selection** - Class, node, and property-based targeting
- **Basic Validation Engine** - Core validation logic and reporting

### 🚧 In Development
- **W3C Test Suite Compliance** - Working toward full specification compliance
- **Performance Optimization** - Constraint evaluation caching and parallelization
- **SHACL-SPARQL Extensions** - Advanced SPARQL-based constraints
- **Validation Reports** - Multiple output formats and detailed violation information
- **API Stabilization** - Builder patterns and comprehensive error handling

### 🔮 Planned
- **Streaming Validation** - Real-time validation for RDF streams
- **Enterprise Features** - Analytics, security hardening, and federation support
- **Integration Tools** - CLI utilities and ecosystem integration

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-shacl = "0.2.3"
oxirs-core = "0.2.3"
```

### Basic Usage

> ⚠️ **Note**: API is still under development and subject to change.

```rust
use oxirs_shacl::{ValidationEngine, ValidationConfig, Shape};
use oxirs_core::store::Store;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // This example shows the intended API - implementation in progress
    
    // Create shapes and validation configuration
    let shapes = vec![/* shapes loaded from RDF */];
    let config = ValidationConfig::default();
    
    // Create validation engine
    let engine = ValidationEngine::new(&shapes, config);
    
    // Create and populate RDF store
    let store = Store::new()?;
    // store.load_from_reader(...)?;
    
    // Validate data (API under development)
    // let report = engine.validate_store(&store)?;
    
    Ok(())
}
```

For current implementation details, see the [source code](src/) and [tests](tests/).

## 🏗️ Current Development Status

This crate is actively under development. The core constraint types and validation engine architecture are implemented, but the public API is still being finalized.

### Implementation Progress
- ✅ Core constraint types (class, datatype, cardinality, range, string, etc.)
- ✅ Property path evaluation engine  
- ✅ Logical constraints (and, or, not, xone)
- ✅ Basic validation engine architecture
- 🚧 W3C SHACL test suite compliance
- 🚧 Public API stabilization
- 🚧 Performance optimization
- 🔮 SHACL-SPARQL extensions
- 🔮 Comprehensive validation reports

## 🧪 Development & Testing

### Running Tests

```bash
# Run basic tests
cargo nextest run --no-fail-fast

# Run with all features
cargo nextest run --all-features --no-fail-fast

# Run specific modules
cargo test constraints
cargo test validation
```

### Code Quality

```bash
# Lint code (following "no warnings policy")
cargo clippy --workspace --all-targets -- -D warnings

# Format code
cargo fmt --all -- --check

# Generate documentation
cargo doc --no-deps --open
```

### Building from Source

```bash
git clone https://github.com/cool-japan/oxirs.git
cd oxirs/engine/oxirs-shacl
cargo build --release
```

### Running Benchmarks

```bash
cargo bench
```

## 🤝 Contributing

This crate is under active development. Contributions are welcome! Current priorities:

1. **W3C SHACL test suite compliance** - Help fix failing tests
2. **Performance optimization** - Constraint evaluation improvements  
3. **API stabilization** - Builder patterns and error handling
4. **Documentation** - Examples and API documentation

### Development Setup

1. Install Rust (latest stable)
2. Clone the repository
3. Run `cargo nextest run --no-fail-fast` to verify setup
4. Make your changes following the "no warnings policy"
5. Run `cargo clippy --workspace --all-targets -- -D warnings`
6. Run `cargo fmt --all`
7. Submit a pull request

## 📜 License

Licensed under the Apache License, Version 2.0.

## 📚 Related Projects

- **[oxirs-core](../../core/oxirs-core)** - Core RDF data structures and operations
- **[oxirs-arq](../oxirs-arq)** - SPARQL query engine  
- **[oxirs-fuseki](../../server/oxirs-fuseki)** - SPARQL server

---

*Part of the [OxiRS](https://github.com/cool-japan/oxirs) ecosystem - High-performance RDF tools for Rust*