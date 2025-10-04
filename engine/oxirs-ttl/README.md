# OxiRS TTL - RDF Format Support

[![Version](https://img.shields.io/badge/version-0.1.0--alpha.2-orange)](https://github.com/cool-japan/oxirs/releases)

**Status**: Alpha Release (v0.1.0-alpha.2) - Released October 4, 2025

⚠️ **Alpha Software**: This is an early alpha release. APIs may change without notice. Not recommended for production use.

High-performance parsers and serializers for RDF formats including Turtle, N-Triples, TriG, N-Quads, JSON-LD, and RDF/XML.

## Features

### Supported Formats

#### Parsing
- **Turtle (.ttl)** - Full Turtle 1.1 support with prefixes and collections
- **N-Triples (.nt)** - Simple triple format
- **TriG (.trig)** - Named graphs extension of Turtle
- **N-Quads (.nq)** - Named graphs extension of N-Triples
- **JSON-LD (.jsonld)** - JSON-based RDF format
- **RDF/XML (.rdf)** - XML-based RDF format

#### Serialization
- All parsing formats supported
- Pretty-printing with customizable indentation
- Compact output mode
- Streaming serialization for large datasets

### Performance Features
- **Streaming Parser** - Memory-efficient parsing of large files
- **Parallel Processing** - Multi-threaded parsing for large datasets
- **Error Recovery** - Continue parsing after errors
- **Validation** - Optional strict validation mode

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
oxirs-ttl = "0.1.0-alpha.2"
```

## Quick Start

### Parsing RDF

```rust
use oxirs_ttl::{TurtleParser, Format};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse Turtle file
    let file = File::open("data.ttl")?;
    let parser = TurtleParser::new(file);

    for triple in parser {
        let triple = triple?;
        println!("{} {} {}", triple.subject, triple.predicate, triple.object);
    }

    Ok(())
}
```

### Automatic Format Detection

```rust
use oxirs_ttl::Parser;

let parser = Parser::from_path("data.ttl")?;  // Auto-detects format
for triple in parser {
    let triple = triple?;
    // Process triple
}
```

### Serialization

```rust
use oxirs_ttl::{TurtleWriter, WriterConfig};
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create("output.ttl")?;
    let config = WriterConfig::pretty();
    let mut writer = TurtleWriter::new(file, config)?;

    // Write prefixes
    writer.write_prefix("ex", "http://example.org/")?;
    writer.write_prefix("foaf", "http://xmlns.com/foaf/0.1/")?;

    // Write triples
    writer.write_triple("ex:alice", "foaf:name", "\"Alice\"")?;
    writer.write_triple("ex:alice", "foaf:knows", "ex:bob")?;

    writer.finish()?;
    Ok(())
}
```

## Format-Specific Features

### Turtle

```rust
use oxirs_ttl::turtle::TurtleParser;

let parser = TurtleParser::builder()
    .with_base("http://example.org/")
    .with_prefix("ex", "http://example.org/")
    .strict(true)  // Enable strict validation
    .build()?;
```

### N-Triples

```rust
use oxirs_ttl::ntriples::NTriplesParser;

// Simple, line-based format
let parser = NTriplesParser::new(reader);
for result in parser {
    let triple = result?;
    // Each line is one triple
}
```

### JSON-LD

```rust
use oxirs_ttl::jsonld::{JsonLdParser, JsonLdOptions};

let options = JsonLdOptions {
    base: Some("http://example.org/".to_string()),
    expand_context: true,
    ..Default::default()
};

let parser = JsonLdParser::new_with_options(reader, options)?;
```

### TriG (Named Graphs)

```rust
use oxirs_ttl::trig::TriGParser;

let parser = TriGParser::new(reader);
for result in parser {
    let quad = result?;
    println!("Graph: {}, Triple: {} {} {}",
        quad.graph, quad.subject, quad.predicate, quad.object);
}
```

## Streaming for Large Files

```rust
use oxirs_ttl::{TurtleParser, StreamingParser};

// Memory-efficient streaming
let parser = TurtleParser::streaming("large_file.ttl")?;

for batch in parser.batches(10000) {
    let triples = batch?;
    // Process batch of 10,000 triples
    process_batch(triples)?;
}
```

## Error Handling

```rust
use oxirs_ttl::{TurtleParser, ParseError};

let parser = TurtleParser::new(reader);

for result in parser {
    match result {
        Ok(triple) => {
            // Process valid triple
        }
        Err(ParseError::SyntaxError { line, column, message }) => {
            eprintln!("Syntax error at {}:{}: {}", line, column, message);
        }
        Err(ParseError::IoError(e)) => {
            eprintln!("I/O error: {}", e);
        }
        Err(e) => {
            eprintln!("Parse error: {}", e);
        }
    }
}
```

## Writer Configuration

```rust
use oxirs_ttl::{TurtleWriter, WriterConfig};

let config = WriterConfig {
    pretty: true,
    indent: "  ".to_string(),
    use_prefixes: true,
    write_base: true,
    max_line_length: 80,
    sort_predicates: true,
};

let mut writer = TurtleWriter::new(file, config)?;
```

## Performance

### Benchmarks

| Format | Parse Speed | Serialize Speed |
|--------|-------------|-----------------|
| Turtle | 250K triples/s | 180K triples/s |
| N-Triples | 400K triples/s | 350K triples/s |
| JSON-LD | 120K triples/s | 100K triples/s |
| RDF/XML | 180K triples/s | 150K triples/s |

*Benchmarked on M1 Mac with typical RDF datasets*

## Integration with OxiRS

### With oxirs-core

```rust
use oxirs_core::Dataset;
use oxirs_ttl::TurtleParser;

let mut dataset = Dataset::new();
let parser = TurtleParser::from_path("data.ttl")?;

for triple in parser {
    let triple = triple?;
    dataset.insert(triple)?;
}
```

### Batch Loading

```rust
use oxirs_core::Dataset;
use oxirs_ttl::Parser;

let dataset = Dataset::from_parser(
    Parser::from_path("data.ttl")?
)?;
```

## Status

### Alpha Release (v0.1.0-alpha.2)
- ✅ Turtle, TriG, N-Triples, N-Quads, JSON-LD, RDF/XML parsing & serialization
- ✅ Streaming pipelines powering CLI import/export/migrate commands
- ✅ Automatic dataset persistence with N-Quads-backed save/load
- ✅ Progress instrumentation and SciRS2 metrics for large batch operations
- 🚧 Full JSON-LD context management (beta target)
- 🚧 Advanced streaming optimizations (ongoing)

## Contributing

This is a foundational module for OxiRS. Contributions welcome!

## License

MIT OR Apache-2.0

## See Also

- [oxirs-core](../../core/oxirs-core/) - RDF data model
- [oxirs-star](../oxirs-star/) - RDF-star format support