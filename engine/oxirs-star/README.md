# OxiRS-Star

RDF-star and SPARQL-star implementation providing comprehensive support for quoted triples in the OxiRS ecosystem.

## Overview

OxiRS-Star extends the standard RDF model with RDF-star capabilities, enabling triples to be used as subjects or objects in other triples (quoted triples). This allows for sophisticated metadata and provenance tracking in RDF datasets.

## Features

- ✅ **Complete RDF-star data model** with proper type safety
- ✅ **Multi-format parsing** for Turtle-star, N-Triples-star, TriG-star, N-Quads-star
- ✅ **SPARQL-star query execution** with quoted triple patterns
- ✅ **Serialization support** for all major RDF-star formats
- ✅ **Storage backend integration** with oxirs-core
- ✅ **Performance optimization** for nested quoted triples
- ✅ **Reification strategies** for legacy RDF compatibility

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies] 
oxirs-star = { path = "path/to/oxirs/engine/oxirs-star" }
```

### Basic Usage

```rust
use oxirs_star::{StarStore, StarTriple, StarTerm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut store = StarStore::new();

    // Create a quoted triple
    let quoted = StarTriple::new(
        StarTerm::iri("http://example.org/person1")?,
        StarTerm::iri("http://example.org/age")?,
        StarTerm::literal("25")?,
    );

    // Use the quoted triple as a subject for metadata
    let meta_triple = StarTriple::new(
        StarTerm::quoted_triple(quoted),
        StarTerm::iri("http://example.org/certainty")?,
        StarTerm::literal("0.9")?,
    );

    store.insert(&meta_triple)?;
    Ok(())
}
```

### Parsing RDF-star Formats

```rust
use oxirs_star::parser::StarParser;

let turtle_star = r#"
    <<:alice :age 25>> :certainty 0.9 .
    <<:bob :knows :alice>> :source :survey2023 .
"#;

let parser = StarParser::new();
let triples = parser.parse_turtle_star(turtle_star)?;
```

### SPARQL-star Queries

```rust
use oxirs_star::query::StarQueryEngine;

let query = r#"
    SELECT ?triple ?certainty WHERE {
        ?triple :certainty ?certainty .
        ?triple { ?s :age ?age }
        FILTER(?age > 20)
    }
"#;

let engine = StarQueryEngine::new(&store);
let results = engine.execute(query)?;
```

## Formats Supported

| Format | Parser | Serializer | Status |
|--------|--------|------------|--------|
| Turtle-star | ✅ | ✅ | Complete |
| N-Triples-star | ✅ | ✅ | Complete |
| TriG-star | 🔄 | 🔄 | In Progress |
| N-Quads-star | ❌ | ❌ | Planned |
| JSON-LD-star | ❌ | ❌ | Future |

## Architecture

```
┌─────────────────┐
│   SPARQL-star   │  Query execution with quoted triple patterns
│     Engine      │
└─────────────────┘
         │
┌─────────────────┐
│   RDF-star      │  Core data model: StarTriple, StarTerm, etc.
│     Model       │
└─────────────────┘
         │
┌─────────────────┐
│   Parsers &     │  Multi-format I/O for RDF-star serializations
│  Serializers    │
└─────────────────┘
         │
┌─────────────────┐
│   StarStore     │  Optimized storage with quoted triple indexing
│                 │
└─────────────────┘
         │
┌─────────────────┐
│   oxirs-core    │  Integration with core RDF infrastructure
│                 │
└─────────────────┘
```

## Configuration

```rust
use oxirs_star::StarConfig;

let config = StarConfig {
    max_nesting_depth: 10,
    enable_reification_fallback: true,
    strict_mode: false,
    enable_sparql_star: true,
    buffer_size: 8192,
};

oxirs_star::init_star_system(config)?;
```

## Performance

OxiRS-Star is designed for high-performance RDF-star processing:

- **Optimized indexing** for quoted triple patterns
- **Memory-efficient** nested triple representation  
- **Streaming support** for large datasets
- **Concurrent access** with lock-free data structures
- **Bulk operations** for high-throughput scenarios

## Testing

```bash
# Run all tests
cargo nextest run --no-fail-fast

# Run specific test suites
cargo nextest run -p oxirs-star --no-fail-fast

# Run benchmarks
cargo bench

# Test with specific features
cargo nextest run --features "reification,sparql-star" --no-fail-fast
```

## Current Limitations

- N-Quads-star parser not yet implemented (see TODO.md)
- TriG-star serializer incomplete
- Some unsafe code in storage layer needs refactoring
- Limited SPARQL-star built-in function support

## Contributing

See [TODO.md](TODO.md) for current development priorities. Key areas needing work:

1. **Parser completion** - N-Quads-star implementation
2. **Serializer completion** - TriG-star and N-Quads-star  
3. **Storage optimization** - Remove unsafe code, improve indexing
4. **Test coverage** - Comprehensive test suite for all formats
5. **Performance** - Benchmarking and optimization

## Dependencies

- `oxirs-core` - Core RDF functionality
- `serde` - Serialization support
- `anyhow` / `thiserror` - Error handling
- `tracing` - Logging and instrumentation

## License

Same as OxiRS project license.

## See Also

- [RDF-star Working Group](https://www.w3.org/2021/12/rdf-star.html) - W3C standardization
- [SPARQL-star Specification](https://w3c.github.io/rdf-star/cg-spec/editors_draft.html) - Query language spec
- [oxirs-core](../../core/oxirs-core/) - Core RDF functionality
- [TODO.md](TODO.md) - Detailed development roadmap