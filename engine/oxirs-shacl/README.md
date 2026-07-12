# OxiRS SHACL 🔍

[![Version](https://img.shields.io/badge/version-0.3.2-blue)](https://github.com/cool-japan/oxirs/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Build Status](https://github.com/cool-japan/oxirs/workflows/CI/badge.svg)](https://github.com/cool-japan/oxirs/actions)

**Status**: v0.3.2 - Released 2026-07-12

✨ **Production Release**: Production-ready with API stability guarantees. Semantic versioning enforced.

A high-performance SHACL (Shapes Constraint Language) validator for RDF data, implemented in Rust as part of the OxiRS ecosystem, with full W3C SHACL Core compliance (27/27 constraint types).

## 🎯 Features

### ✅ Implemented
- **SHACL Core** - Complete constraint validation: 27/27 W3C constraint types (class, datatype, cardinality, range, string, logical, shape-based, and more)
- **Property Path Support** - Sequence, alternative, inverse, and Kleene (zero-or-more, one-or-more, zero-or-one) paths
- **Logical Constraints** - Full support for `sh:and`, `sh:or`, `sh:not`, `sh:xone`
- **Shape-based Constraints** - Nested shape validation and qualified cardinality
- **Target Selection** - Class, node, and property-based targeting, plus SPARQL-based (`sh:target`) and single-hop property-path targets that execute against the store
- **Subclass-Aware Class Targeting** - Reflexive+transitive `rdfs:subClassOf` closure (`advanced_features::subclass_closure`, Floyd–Warshall over a boolean adjacency matrix), so `sh:class` and implicit-class targets honor subclassing instead of exact-type matching only
- **SHACL-SPARQL / SHACL-AF** - SPARQL-based constraints, SPARQL targets, and ASK validators
- **W3C Test Suite Compliance** - 27/27 W3C SHACL Core constraint types; 47/47 real conformance tests passing
- **Performance Optimization** - Constraint evaluation caching, parallelization, and incremental validation
- **Validation Reports** - W3C-compliant violation reports; JUnit, TAP, SARIF, and JSON output for CI/CD integration
- **Distributed Validation** - Coordinator-worker architecture
- **Cross-Module Integration** - GraphQL, Fuseki, Stream, and AI modules
- **Constraint Component Library** - 30+ pre-built validators
- **ShEx Migration** - ShEx to SHACL migration tool
- **LSP Integration** - Language Server Protocol support for IDE tooling
- **Interactive Designer** - Step-by-step shape creation wizard
- **API Stability** - Public APIs are stable; semantic versioning enforced

### 🔮 Planned
- **Streaming Validation** - Real-time validation for RDF streams

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-shacl = "0.3.2"
oxirs-core = "0.3.2"
```

### Basic Usage

```rust
use indexmap::IndexMap;
use oxirs_core::{NamedNode, RdfStore};
use oxirs_shacl::{
    constraints::{cardinality_constraints::MinCountConstraint, Constraint},
    ConstraintComponentId, Shape, ShapeId, Target, ValidationConfig, ValidationEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a shape: every ex:Person must have at least one value for the target property
    let mut person_shape =
        Shape::node_shape(ShapeId("http://example.org/PersonShape".to_string()));
    person_shape.add_target(Target::class(NamedNode::new_unchecked(
        "http://example.org/Person",
    )));
    person_shape.add_constraint(
        ConstraintComponentId::new("sh:minCount"),
        Constraint::MinCount(MinCountConstraint { min_count: 1 }),
    );

    let mut shapes = IndexMap::new();
    shapes.insert(person_shape.id.clone(), person_shape);

    // Validate an RDF store against the shapes
    let config = ValidationConfig::default();
    let mut engine = ValidationEngine::new(&shapes, config);
    let store = RdfStore::new()?;
    let report = engine.validate_store(&store)?;
    println!("Conforms: {}", report.conforms());

    Ok(())
}
```

For more complete, runnable examples, see [`examples/basic_validation.rs`](examples/basic_validation.rs)
and [`examples/parallel_validation.rs`](examples/parallel_validation.rs), plus the
[source code](src/) and [tests](tests/).

## 🏗️ Current Development Status

This crate is production-ready. Core constraint types, the validation engine, and the public API are stable and covered by 2,140 passing tests (`--all-features`).

### Implementation Progress
- ✅ Core constraint types (class, datatype, cardinality, range, string, etc.)
- ✅ Property path evaluation engine
- ✅ Logical constraints (and, or, not, xone)
- ✅ Validation engine architecture, with SPARQL-based and property-path targets executing against the store
- ✅ W3C SHACL test suite compliance (27/27 constraint types, 47/47 conformance tests)
- ✅ Public API stabilization
- ✅ Performance optimization (caching, parallelization, incremental validation)
- ✅ SHACL-SPARQL extensions
- ✅ Comprehensive validation reports (JUnit, TAP, SARIF, JSON)

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

This crate is production-ready; contributions are still welcome! Current priorities:

1. **Streaming validation** - Real-time validation for RDF streams
2. **Further W3C SHACL-AF conformance breadth** - continued coverage beyond SHACL Core
3. **Performance optimization** - Constraint evaluation improvements
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