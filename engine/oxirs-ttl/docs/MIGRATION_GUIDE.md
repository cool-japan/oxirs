# Migration Guide: From Oxigraph/Rio to OxiRS-TTL

This guide helps you migrate from the Oxigraph `rio` parser to `oxirs-ttl`.

## Table of Contents

1. [Why Migrate?](#why-migrate)
2. [Key Differences](#key-differences)
3. [Basic Usage Migration](#basic-usage-migration)
4. [Advanced Features](#advanced-features)
5. [Performance Optimization](#performance-optimization)
6. [Breaking Changes](#breaking-changes)
7. [Migration Checklist](#migration-checklist)

## Why Migrate?

OxiRS-TTL offers several advantages over Oxigraph's rio:

| Feature | Rio (Oxigraph) | OxiRS-TTL |
|---------|---------------|-----------|
| **N3 Support** | ❌ No | ✅ Full (formulas, variables, implications) |
| **Incremental Parsing** | ❌ No | ✅ Yes (parse as bytes arrive) |
| **Error Recovery** | Limited | ✅ Lenient mode with error collection |
| **Streaming** | Basic | ✅ Advanced (configurable batch sizes) |
| **Async I/O** | ❌ No | ✅ Tokio integration |
| **Parallel Processing** | ❌ No | ✅ Rayon-based parallel parsing |
| **RDF 1.2** | Partial | ✅ Full (quoted triples, directional tags) |
| **Serialization** | Basic | ✅ Optimized (predicate grouping, collections) |
| **Performance** | Good | ✅ Excellent (SIMD, zero-copy) |
| **Test Coverage** | ~200 tests | ✅ 469 tests (97% W3C compliance) |

## Key Differences

### 1. Package Name

```toml
# Old (Oxigraph/Rio)
[dependencies]
rio_turtle = "0.8"
rio_api = "0.8"

# New (OxiRS-TTL)
[dependencies]
oxirs-ttl = "0.1.0-rc.2"
oxirs-core = "0.1.0-rc.2"  # For RDF data model
```

### 2. Module Structure

```rust
// Old (Rio)
use rio_turtle::{TurtleParser, TurtleError};
use rio_api::parser::TriplesParser;

// New (OxiRS-TTL)
use oxirs_ttl::turtle::TurtleParser;
use oxirs_ttl::{Parser, TurtleResult};
use oxirs_core::model::Triple;
```

### 3. API Style

Rio uses a callback-based API, while OxiRS-TTL uses an iterator-based API:

```rust
// Old (Rio) - Callback-based
let mut parser = TurtleParser::new(reader, None);
parser.parse_all(&mut |triple| {
    println!("{:?}", triple);
    Ok(())
})?;

// New (OxiRS-TTL) - Iterator-based
let parser = TurtleParser::new();
for result in parser.for_reader(reader) {
    let triple = result?;
    println!("{:?}", triple);
}
```

## Basic Usage Migration

### Parsing Turtle

#### Before (Rio)

```rust
use rio_turtle::TurtleParser;
use rio_api::parser::TriplesParser;
use std::io::Cursor;

let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate "object" .
"#;

let mut count = 0;
let mut parser = TurtleParser::new(Cursor::new(turtle), None);
parser.parse_all(&mut |_triple| {
    count += 1;
    Ok(())
})?;

println!("Parsed {} triples", count);
```

#### After (OxiRS-TTL)

```rust
use oxirs_ttl::turtle::TurtleParser;
use oxirs_ttl::Parser;
use std::io::Cursor;

let turtle = r#"
@prefix ex: <http://example.org/> .
ex:subject ex:predicate "object" .
"#;

let parser = TurtleParser::new();
let triples: Vec<_> = parser.for_reader(Cursor::new(turtle))
    .collect::<Result<Vec<_>, _>>()?;

println!("Parsed {} triples", triples.len());
```

### Parsing N-Triples

#### Before (Rio)

```rust
use rio_turtle::NTriplesParser;
use rio_api::parser::TriplesParser;

let mut parser = NTriplesParser::new(reader);
parser.parse_all(&mut |triple| {
    // Process triple
    Ok(())
})?;
```

#### After (OxiRS-TTL)

```rust
use oxirs_ttl::ntriples::NTriplesParser;
use oxirs_ttl::Parser;

let parser = NTriplesParser::new();
for result in parser.for_reader(reader) {
    let triple = result?;
    // Process triple
}
```

### Parsing TriG

#### Before (Rio)

```rust
use rio_turtle::TriGParser;
use rio_api::parser::QuadsParser;

let mut parser = TriGParser::new(reader, None);
parser.parse_all(&mut |quad| {
    println!("{:?}", quad);
    Ok(())
})?;
```

#### After (OxiRS-TTL)

```rust
use oxirs_ttl::trig::TriGParser;
use oxirs_ttl::Parser;

let parser = TriGParser::new();
for result in parser.for_quads_reader(reader) {
    let quad = result?;
    println!("{:?}", quad);
}
```

## Advanced Features

### 1. Error Recovery (New in OxiRS-TTL)

```rust
use oxirs_ttl::turtle::TurtleParser;

// Rio: No built-in error recovery
// OxiRS-TTL: Lenient mode continues parsing after errors

let parser = TurtleParser::new_lenient();
let result = parser.parse_document(turtle_with_errors)?;
// Returns valid triples, skips invalid ones
```

### 2. Streaming Large Files

#### Before (Rio)

```rust
// Rio: Parse entire document at once
let mut parser = TurtleParser::new(file, None);
parser.parse_all(&mut |triple| {
    // Process triple
    Ok(())
})?;
```

#### After (OxiRS-TTL)

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};

// OxiRS-TTL: Process in configurable batches
let config = StreamingConfig::default()
    .with_batch_size(10_000);  // 10K triples per batch

let parser = StreamingParser::with_config(file, config);
for batch in parser.batches() {
    let triples = batch?;
    // Process batch of 10K triples
}
```

### 3. Async I/O (New in OxiRS-TTL)

```rust
use oxirs_ttl::async_parser::AsyncTurtleParser;
use tokio::fs::File;

// Rio: No async support
// OxiRS-TTL: Full Tokio integration

let file = File::open("large.ttl").await?;
let parser = AsyncTurtleParser::new();

let mut stream = parser.parse_tokio(file).await?;
while let Some(triple) = stream.next().await {
    let triple = triple?;
    // Async processing
}
```

### 4. Parallel Processing (New in OxiRS-TTL)

```rust
use oxirs_ttl::parallel::ParallelParser;

// Rio: No parallel support
// OxiRS-TTL: Rayon-based parallel parsing

let parser = ParallelParser::new();
let triples = parser.parse_file_parallel("huge.ttl")?;
// Automatically uses all CPU cores
```

### 5. Incremental Parsing (New in OxiRS-TTL)

```rust
use oxirs_ttl::incremental::IncrementalParser;

// Rio: Must have complete document
// OxiRS-TTL: Parse as bytes arrive

let mut parser = IncrementalParser::new();

// Feed data as it arrives
parser.push_data(b"@prefix ex: <http://example.org/> .\n")?;
parser.push_data(b"ex:s ex:p \"object\" .\n")?;
parser.push_eof();

// Parse available complete statements
let triples = parser.parse_available()?;
```

### 6. N3 Support (New in OxiRS-TTL)

```rust
use oxirs_ttl::formats::n3_parser::AdvancedN3Parser;

// Rio: No N3 support
// OxiRS-TTL: Full N3 with formulas, variables, implications

let input = r#"
    @prefix ex: <http://example.org/> .
    @forAll ?x, ?y .

    { ?x ex:knows ?y } => { ?y ex:knows ?x } .
"#;

let mut parser = AdvancedN3Parser::new(input)?;
let doc = parser.parse_document()?;

println!("Statements: {}", doc.statements.len());
println!("Implications: {}", doc.implications.len());
```

## Performance Optimization

### 1. Zero-Copy Parsing

```rust
use oxirs_ttl::toolkit::zero_copy::ZeroCopyIriParser;

// OxiRS-TTL uses zero-copy parsing where possible
let parser = ZeroCopyIriParser::new();
let iri = parser.parse("<http://example.org/resource>")?;
// Returns Cow<str> - no allocation if no escapes
```

### 2. String Interning

```rust
use oxirs_ttl::toolkit::string_interner::StringInterner;

// OxiRS-TTL deduplicates common IRIs
let mut interner = StringInterner::new();
let iri1 = interner.intern("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
let iri2 = interner.intern("http://www.w3.org/1999/02/22-rdf-syntax-ns#type");
assert!(std::ptr::eq(iri1, iri2));  // Same pointer!
```

### 3. SIMD Acceleration

```rust
use oxirs_ttl::toolkit::simd_lexer::SimdLexer;

// OxiRS-TTL uses SIMD (memchr) for fast tokenization
let lexer = SimdLexer::new(content);
// 2-4x faster than byte-by-byte scanning
```

## Breaking Changes

### 1. Error Types

```rust
// Rio
use rio_turtle::TurtleError;

// OxiRS-TTL
use oxirs_ttl::error::TurtleParseError;
```

### 2. Triple/Quad Representation

```rust
// Rio uses its own types
use rio_api::model::{Triple, NamedNode};

// OxiRS-TTL uses oxirs-core types
use oxirs_core::model::{Triple, NamedNode};
```

### 3. Parser Construction

```rust
// Rio: Parser takes reader in constructor
let parser = TurtleParser::new(reader, base_iri);

// OxiRS-TTL: Parser is reusable, reader passed to parse method
let parser = TurtleParser::new();
let triples = parser.for_reader(reader);
```

## Migration Checklist

### Phase 1: Update Dependencies

- [ ] Update `Cargo.toml` to use `oxirs-ttl` and `oxirs-core`
- [ ] Remove `rio_turtle` and `rio_api` dependencies
- [ ] Update import statements

### Phase 2: Update Parsing Code

- [ ] Convert callback-based parsing to iterator-based
- [ ] Update error types (`TurtleError` → `TurtleParseError`)
- [ ] Update data model types (Rio types → `oxirs-core` types)
- [ ] Adjust parser construction (reader in method, not constructor)

### Phase 3: Add New Features (Optional)

- [ ] Add error recovery with lenient mode
- [ ] Implement streaming for large files
- [ ] Add async I/O for network sources
- [ ] Enable parallel processing for huge datasets
- [ ] Use incremental parsing for real-time data

### Phase 4: Optimize Performance (Optional)

- [ ] Enable zero-copy parsing where possible
- [ ] Use string interning for common namespaces
- [ ] Leverage SIMD acceleration
- [ ] Profile and optimize hot paths

### Phase 5: Test & Validate

- [ ] Run existing test suite
- [ ] Add tests for new features
- [ ] Benchmark performance improvements
- [ ] Verify W3C compliance

## Example: Complete Migration

### Before (Rio)

```rust
use rio_turtle::TurtleParser;
use rio_api::parser::TriplesParser;
use std::fs::File;
use std::io::BufReader;

fn parse_turtle_file(path: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut count = 0;
    let mut parser = TurtleParser::new(reader, None);

    parser.parse_all(&mut |triple| {
        count += 1;
        // Process triple
        Ok(())
    })?;

    Ok(count)
}
```

### After (OxiRS-TTL)

```rust
use oxirs_ttl::turtle::TurtleParser;
use oxirs_ttl::Parser;
use std::fs::File;
use std::io::BufReader;

fn parse_turtle_file(path: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let parser = TurtleParser::new();
    let triples: Vec<_> = parser.for_reader(reader)
        .collect::<Result<Vec<_>, _>>()?;

    // Or for large files, use streaming:
    // let config = StreamingConfig::default().with_batch_size(10_000);
    // let streaming = StreamingParser::with_config(file, config);
    // for batch in streaming.batches() {
    //     let triples = batch?;
    //     // Process batch
    // }

    Ok(triples.len())
}
```

## Getting Help

- **Documentation**: https://docs.rs/oxirs-ttl
- **Examples**: See `examples/` directory
- **Issues**: https://github.com/cool-japan/oxirs/issues
- **Discussions**: https://github.com/cool-japan/oxirs/discussions

## See Also

- [README.md](../README.md) - Main documentation
- [STREAMING_TUTORIAL.md](STREAMING_TUTORIAL.md) - Streaming guide
- [ASYNC_GUIDE.md](ASYNC_GUIDE.md) - Async I/O guide
- [PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) - Performance optimization

---

**Note**: This migration guide is actively maintained. If you encounter any issues or have suggestions, please open an issue on GitHub.
