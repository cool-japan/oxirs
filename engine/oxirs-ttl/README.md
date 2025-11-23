# OxiRS TTL - RDF Turtle Family Parser & Serializer

[![Version](https://img.shields.io/badge/version-0.1.0--beta.2-blue)](https://github.com/cool-japan/oxirs/releases)
[![Tests](https://img.shields.io/badge/tests-461%20passing-green)](https://github.com/cool-japan/oxirs)
[![Compliance](https://img.shields.io/badge/W3C-97%25%20compliant-brightgreen)](https://www.w3.org/TR/turtle/)

**Status**: Beta Release (v0.1.0-beta.2) - November 2025

✨ **Production-Ready**: API-stable with comprehensive testing (461 tests), W3C compliance, and performance optimizations.

High-performance parsers and serializers for RDF formats in the Turtle family including Turtle, N-Triples, TriG, N-Quads, and N3. Ported from Oxigraph's oxttl crate with extensive enhancements for OxiRS.

## Features

### Supported Formats

| Format | Extension | Parsing | Serialization | Named Graphs | Description |
|--------|-----------|---------|---------------|--------------|-------------|
| **Turtle** | `.ttl` | ✅ | ✅ | ❌ | Full Turtle 1.1 with prefixes, collections, and abbreviated syntax |
| **N-Triples** | `.nt` | ✅ | ✅ | ❌ | Simple line-based triple format |
| **TriG** | `.trig` | ✅ | ✅ | ✅ | Turtle extension with named graphs |
| **N-Quads** | `.nq` | ✅ | ✅ | ✅ | N-Triples extension with named graphs |
| **N3** | `.n3` | 🚧 | ❌ | ✅ | Experimental: Variables and formulas |

### RDF 1.2 Support 🆕

**Quoted Triples** (RDF-star):
```turtle
@prefix ex: <http://example.org/> .

# Quoted triple as subject
<< ex:alice ex:knows ex:bob >> ex:certainty 0.9 .

# Nested quoted triples
<< << ex:alice ex:knows ex:bob >> ex:source ex:socialNetwork >> ex:timestamp "2025-11-23" .
```

**Directional Language Tags**:
```turtle
ex:greeting "Hello"@en--ltr .  # Left-to-right
ex:greeting "مرحبا"@ar--rtl .  # Right-to-left
```

Enable with feature flag:
```toml
[dependencies]
oxirs-ttl = { version = "0.1.0-beta.2", features = ["rdf-12"] }
```

### Advanced Features

- ✅ **Streaming Support** - Memory-efficient parsing of multi-GB files
- ✅ **Async I/O** - Tokio-based async parsing (feature: `async-tokio`)
- ✅ **Parallel Processing** - Multi-threaded parsing with rayon (feature: `parallel`)
- ✅ **Error Recovery** - Lenient mode continues parsing after errors
- ✅ **Incremental Parsing** - Parse as bytes arrive with checkpointing
- ✅ **Format Auto-Detection** - Automatic format detection from extension/MIME/content
- ✅ **W3C Compliance** - 97% pass rate on official W3C Turtle test suite
- ✅ **Performance Optimizations** - SIMD lexing, zero-copy parsing, lazy IRI resolution
- ✅ **Serialization Optimizations** - Predicate grouping, object lists, blank node inlining, collection syntax

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-ttl = "0.1.0-beta.2"

# With all features
oxirs-ttl = { version = "0.1.0-beta.2", features = ["async-tokio", "parallel", "rdf-12"] }
```

## Quick Start

### Basic Turtle Parsing

```rust
use oxirs_ttl::turtle::TurtleParser;
use std::io::Cursor;

let turtle_data = r#"
@prefix ex: <http://example.org/> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

ex:alice foaf:name "Alice" ;
         foaf:knows ex:bob, ex:charlie .
"#;

let parser = TurtleParser::new();
for result in parser.for_reader(Cursor::new(turtle_data)) {
    let triple = result?;
    println!("{}", triple);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Automatic Format Detection

```rust
use oxirs_ttl::toolkit::format_detector::FormatDetector;
use std::fs::File;

// Detect format from file extension
let format = FormatDetector::detect_from_path("data.ttl")?;
println!("Detected format: {:?}", format);

// Auto-detect from content
let content = std::fs::read_to_string("data.ttl")?;
let detection = FormatDetector::detect_from_content(&content);
println!("Format: {:?}, Confidence: {}", detection.format, detection.confidence);
```

### N-Triples Parsing

```rust
use oxirs_ttl::ntriples::NTriplesParser;
use std::io::Cursor;

let ntriples_data = r#"
<http://example.org/alice> <http://xmlns.com/foaf/0.1/name> "Alice" .
<http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> .
"#;

let parser = NTriplesParser::new();
for result in parser.for_reader(Cursor::new(ntriples_data)) {
    let triple = result?;
    println!("{}", triple);
}
```

### TriG Parsing (Named Graphs)

```rust
use oxirs_ttl::trig::TriGParser;
use std::io::Cursor;

let trig_data = r#"
@prefix ex: <http://example.org/> .

ex:graph1 {
    ex:alice ex:knows ex:bob .
}

ex:graph2 {
    ex:charlie ex:knows ex:dave .
}
"#;

let parser = TriGParser::new();
for result in parser.for_reader(Cursor::new(trig_data)) {
    let quad = result?;
    println!("Graph: {:?}, Triple: {}", quad.graph_name, quad);
}
```

## Advanced Usage

### Streaming for Large Files

Memory-efficient parsing of multi-gigabyte RDF files:

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};
use std::fs::File;

// Configure streaming with 10K triple batches
let config = StreamingConfig::default()
    .with_batch_size(10_000)
    .with_progress_callback(Box::new(|stats| {
        println!("Processed {} triples in {:.2}s",
            stats.triples_parsed, stats.elapsed_seconds);
    }));

let file = File::open("large_dataset.ttl")?;
let parser = StreamingParser::with_config(file, config);

let mut total = 0;
for batch in parser.batches() {
    let triples = batch?;
    total += triples.len();
    // Process batch (e.g., insert into database)
}
println!("Total: {} triples", total);
```

### Async I/O with Tokio

Non-blocking async parsing (requires `async-tokio` feature):

```rust
use oxirs_ttl::async_parser::AsyncTurtleParser;
use tokio::fs::File;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("data.ttl").await?;
    let parser = AsyncTurtleParser::new();

    let triples = parser.parse_async(file).await?;
    println!("Parsed {} triples", triples.len());
    Ok(())
}
```

### Parallel Processing

Multi-threaded parsing for large files (requires `parallel` feature):

```rust
use oxirs_ttl::parallel::ParallelStreamingParser;
use std::fs::File;

let file = File::open("large_dataset.ttl")?;
let parser = ParallelStreamingParser::new(file, 4)?; // 4 threads

let triples = parser.collect_all()?;
println!("Parsed {} triples in parallel", triples.len());
```

### Error Recovery (Lenient Mode)

Continue parsing despite syntax errors:

```rust
use oxirs_ttl::turtle::TurtleParser;

let turtle_with_errors = r#"
@prefix ex: <http://example.org/> .
ex:good ex:pred "value" .
ex:bad ex:pred "unclosed string
ex:also_good ex:pred "value2" .
"#;

// Lenient mode collects errors but continues parsing
let parser = TurtleParser::new_lenient();
match parser.parse_document(turtle_with_errors) {
    Ok(triples) => println!("Parsed {} valid triples", triples.len()),
    Err(e) => println!("Errors encountered: {}", e),
}
```

### Incremental Parsing

Parse as bytes arrive (useful for network streams):

```rust
use oxirs_ttl::{IncrementalParser, ParseState};

let mut parser = IncrementalParser::new();

// Feed data as it arrives
parser.push_data(b"@prefix ex: <http://example.org/> .\n")?;
parser.push_data(b"ex:subject ex:predicate \"object\" .\n")?;
parser.push_eof();

// Parse complete statements
let triples = parser.parse_available()?;
println!("Parsed {} triples", triples.len());

assert_eq!(parser.state(), ParseState::Complete);
```

### Serialization with Pretty Printing

```rust
use oxirs_ttl::turtle::TurtleSerializer;
use oxirs_ttl::toolkit::{Serializer, SerializationConfig};
use oxirs_core::model::{NamedNode, Triple, Literal};

let triples = vec![
    Triple::new(
        NamedNode::new("http://example.org/alice")?,
        NamedNode::new("http://xmlns.com/foaf/0.1/name")?,
        Literal::new_simple_literal("Alice")
    ),
    Triple::new(
        NamedNode::new("http://example.org/alice")?,
        NamedNode::new("http://xmlns.com/foaf/0.1/age")?,
        Literal::new_typed_literal("30", NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?)
    ),
];

// Configure pretty printing
let config = SerializationConfig::default()
    .with_pretty(true)
    .with_use_prefixes(true)
    .with_indent("  ");

let serializer = TurtleSerializer::with_config(config);
let mut output = Vec::new();
serializer.serialize(&triples, &mut output)?;

println!("{}", String::from_utf8(output)?);
```

### Optimized Serialization

Compact output with predicate grouping, object lists, and collection syntax:

```rust
use oxirs_ttl::turtle::TurtleSerializer;

// Predicate grouping: ex:alice ex:name "Alice" ; ex:age 30 .
// Object lists: ex:alice ex:knows ex:bob, ex:charlie .
// Blank nodes: [ ex:prop "value" ; ex:other "data" ]
// Collections: ex:list (ex:item1 ex:item2 ex:item3) .

let serializer = TurtleSerializer::new();
let turtle = serializer.serialize_optimized(&triples)?;

// ~76% more compact than verbose representation
println!("{}", turtle);
```

### Performance Profiling

Track parsing performance with built-in profiler:

```rust
use oxirs_ttl::profiling::TtlProfiler;
use std::fs::File;

let mut profiler = TtlProfiler::new();
let file = File::open("data.ttl")?;

profiler.start_parse();
let triples = parse_with_profiler(&file, &mut profiler)?;
profiler.end_parse();

// Get detailed statistics
let stats = profiler.get_stats();
println!("Parsed {} triples in {:.2}s ({:.0} triples/sec)",
    stats.triples_parsed,
    stats.elapsed_seconds,
    stats.throughput);
```

## Serialization Optimizations

OxiRS-TTL provides highly optimized Turtle serialization:

### Predicate Grouping

Same subject with multiple predicates uses semicolon syntax:

```turtle
# Before (verbose)
ex:alice ex:name "Alice" .
ex:alice ex:age 30 .
ex:alice ex:city "Wonderland" .

# After (optimized)
ex:alice ex:name "Alice" ;
         ex:age 30 ;
         ex:city "Wonderland" .
```

### Object Lists

Same subject and predicate with multiple objects uses comma syntax:

```turtle
# Before (verbose)
ex:alice ex:knows ex:bob .
ex:alice ex:knows ex:charlie .
ex:alice ex:knows ex:dave .

# After (optimized)
ex:alice ex:knows ex:bob, ex:charlie, ex:dave .
```

### Blank Node Inlining

Anonymous blank nodes use compact property list syntax:

```turtle
# Before (verbose)
_:b1 ex:city "Wonderland" .
_:b1 ex:country "Fantasy" .
ex:alice ex:location _:b1 .

# After (optimized)
ex:alice ex:location [ ex:city "Wonderland" ; ex:country "Fantasy" ] .
```

### Collection Syntax

RDF collections use compact parenthesis syntax:

```turtle
# Before (verbose - 7 triples)
_:b1 rdf:first ex:item1 .
_:b1 rdf:rest _:b2 .
_:b2 rdf:first ex:item2 .
_:b2 rdf:rest _:b3 .
_:b3 rdf:first ex:item3 .
_:b3 rdf:rest rdf:nil .
ex:list ex:items _:b1 .

# After (optimized - 1 triple)
ex:list ex:items (ex:item1 ex:item2 ex:item3) .
```

## Performance

### Benchmarks

Measured on Apple M1 with typical RDF datasets:

| Format | Parse Speed | Serialize Speed | Features |
|--------|-------------|-----------------|----------|
| **Turtle** | 250-300K triples/s | 180-200K triples/s | SIMD lexing, zero-copy |
| **N-Triples** | 400-500K triples/s | 350-400K triples/s | Line-based, optimized |
| **TriG** | 200-250K triples/s | 160-180K triples/s | Multi-graph support |
| **N-Quads** | 350-450K triples/s | 300-350K triples/s | Quad format |

### Performance Features

- **SIMD Lexing**: Uses `memchr` for 2-4x faster byte scanning
- **Zero-Copy Parsing**: Minimizes string allocations with `Cow<str>`
- **String Interning**: Deduplicates common IRIs (RDF namespaces)
- **Lazy IRI Resolution**: Defers IRI normalization until needed
- **Buffer Pooling**: Reuses parsing buffers in streaming mode
- **Parallel Processing**: Multi-threaded parsing for large files

## Testing & Compliance

### Test Coverage

- **461 tests passing** (437 integration + 24 doc tests)
- **Property-based testing** with proptest
- **Memory leak tests** for production safety
- **Performance regression tests** for baseline tracking
- **Fuzzing infrastructure** for parser robustness

### W3C Compliance

| Test Suite | Pass Rate | Status |
|------------|-----------|--------|
| W3C Turtle Test Suite | 97% (33/34) | ✅ Excellent |
| W3C TriG Test Suite | 94% (33/35) | ✅ Excellent |
| RDF 1.2 Features | 100% (19/19) | ✅ Complete |

## Integration with OxiRS

### With oxirs-core

```rust
use oxirs_core::model::Dataset;
use oxirs_ttl::turtle::TurtleParser;
use std::io::Cursor;

let turtle_data = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate "object" .
"#;

let parser = TurtleParser::new();
let triples = parser.parse_document(turtle_data)?;

let mut dataset = Dataset::new();
for triple in triples {
    dataset.insert(triple.into())?;
}
```

## Error Handling

```rust
use oxirs_ttl::{TurtleParser, TurtleParseError};

match parser.parse_document(data) {
    Ok(triples) => println!("Parsed {} triples", triples.len()),
    Err(TurtleParseError::Syntax(e)) => {
        eprintln!("Syntax error at {}:{}: {}",
            e.position.line, e.position.column, e.message);
    }
    Err(TurtleParseError::Io(e)) => {
        eprintln!("I/O error: {}", e);
    }
    Err(e) => eprintln!("Parse error: {}", e),
}
```

## Configuration

### Streaming Configuration

```rust
use oxirs_ttl::StreamingConfig;

let config = StreamingConfig::default()
    .with_batch_size(10_000)           // Triples per batch
    .with_buffer_size(64 * 1024)       // 64KB read buffer
    .with_progress_reporting(true)     // Enable progress callbacks
    .with_error_recovery(true);        // Lenient mode
```

### Serialization Configuration

```rust
use oxirs_ttl::toolkit::SerializationConfig;

let config = SerializationConfig::default()
    .with_pretty(true)                 // Pretty printing
    .with_indent("  ")                 // 2-space indentation
    .with_use_prefixes(true)           // Abbreviate with prefixes
    .with_max_line_length(80)          // Wrap at 80 chars
    .with_base_iri("http://example.org/"); // Base IRI for relative IRIs
```

## Status

### Beta.2 Release (November 2025) ✅

**Core Features** (100% Complete):
- ✅ Turtle, TriG, N-Triples, N-Quads parsing & serialization
- ✅ RDF 1.2: Quoted triples + directional language tags
- ✅ Streaming, async I/O, parallel processing
- ✅ Error recovery with lenient mode
- ✅ Incremental parsing with checkpointing
- ✅ Format auto-detection
- ✅ W3C compliance testing (97% pass rate)
- ✅ Performance optimizations (SIMD, zero-copy, lazy IRI)
- ✅ Serialization optimizations (predicate grouping, collections)
- ✅ Fuzzing infrastructure
- ✅ Memory leak testing
- ✅ Comprehensive documentation (24 doc tests)

**Advanced Features** (Beta.2):
- ✅ RFC 3987 IRI validation
- ✅ RFC 3986 IRI resolution
- ✅ N3 types & built-in registry (40+ predicates)
- ✅ Profiling & performance metrics
- ✅ Sample data infrastructure

**Future Work**:
- 🚧 Full N3 formula parsing
- 🚧 N3 reasoning primitives
- 🚧 RDF 1.2 official compliance tests

## Contributing

This is a foundational module for OxiRS. Contributions welcome!

### Development

```bash
# Run tests
cargo test --all-features

# Run benchmarks
cargo bench

# Check for warnings (no warnings policy)
cargo clippy --all-features --all-targets -- -D warnings

# Format code
cargo fmt --all

# Run fuzzing (requires cargo-fuzz)
cd fuzz && ./run_all_fuzzers.sh
```

## Documentation

### Guides and Tutorials

- **[Documentation Hub](docs/README.md)** - Complete documentation index
- **[Streaming Tutorial](docs/STREAMING_TUTORIAL.md)** - Memory-efficient large file processing
- **[Async Usage Guide](docs/ASYNC_GUIDE.md)** - Non-blocking I/O with Tokio
- **[Performance Tuning Guide](docs/PERFORMANCE_GUIDE.md)** - Optimization techniques

### API Reference

- [API Documentation](https://docs.rs/oxirs-ttl) - Full API reference
- [Module docs: streaming.rs](src/streaming.rs) - Streaming API examples
- [Module docs: profiling.rs](src/profiling.rs) - Performance tracking

### Specifications

- [W3C Turtle Spec](https://www.w3.org/TR/turtle/) - Turtle 1.1 specification
- [W3C TriG Spec](https://www.w3.org/TR/trig/) - TriG specification
- [RDF 1.2 Spec](https://www.w3.org/TR/rdf12-concepts/) - RDF 1.2 concepts

## License

MIT OR Apache-2.0

## See Also

- [oxirs-core](../../core/oxirs-core/) - RDF data model and core types
- [oxirs-star](../oxirs-star/) - RDF-star extended format support
- [oxirs-arq](../oxirs-arq/) - SPARQL query engine
- [Oxigraph](https://github.com/oxigraph/oxigraph) - Original inspiration
