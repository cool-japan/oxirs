# Test Data Files

This directory contains sample RDF files in various formats for testing and development.

## Files

### `sample.ttl` - Turtle Format

Complete Turtle example demonstrating:
- Prefix declarations
- Basic triples with rdf:type abbreviation (`a`)
- Predicate-object lists (`;`)
- Object lists (`,`)
- Blank nodes (anonymous `[]` and labeled `_:`)
- Collections/lists `()`
- Multi-line string literals (`"""`)
- Language-tagged strings (`@en`, `@ja`)
- Typed literals (`^^xsd:integer`)
- Numeric shortcuts (integers, decimals, scientific notation)
- Boolean literals (`true`, `false`)
- Unicode characters

**Triple Count**: ~25 triples
**Use Case**: General Turtle parser testing, serialization examples

### `sample.nt` - N-Triples Format

N-Triples serialization of a subset of the Turtle data:
- Full IRI format (no prefixes)
- Simple subject-predicate-object statements
- Unicode escape sequences (`\uXXXX`)
- Language-tagged strings
- Typed literals

**Triple Count**: ~13 triples
**Use Case**: Line-based parser testing, streaming benchmarks

### `sample.trig` - TriG Format

TriG example with multiple named graphs:
- Default graph with dataset metadata
- Named graph: People (`ex:graph:people`)
- Named graph: Organizations (`ex:graph:orgs`)
- Named graph: Metadata (`ex:graph:metadata`)
- All Turtle syntax features within named graphs

**Triple Count**: ~20 triples across 4 graphs
**Use Case**: Named graph testing, quad storage validation

### `sample.nq` - N-Quads Format

N-Quads serialization with multiple graphs:
- Default graph statements (no graph component)
- Quads with named graph IRIs
- People graph quads
- Organizations graph quads

**Quad Count**: ~21 quads
**Use Case**: Quad-based parser testing, streaming benchmarks

## Usage Examples

### Parsing Turtle

```rust
use oxirs_ttl::turtle::TurtleParser;
use oxirs_ttl::Parser;
use std::fs::File;
use std::io::BufReader;

let file = File::open("data/sample.ttl")?;
let parser = TurtleParser::new();

for result in parser.for_reader(BufReader::new(file)) {
    let triple = result?;
    println!("{}", triple);
}
```

### Streaming Large Files

```rust
use oxirs_ttl::{StreamingParser, StreamingConfig};
use std::fs::File;

let config = StreamingConfig::default().with_batch_size(1000);
let file = File::open("data/sample.nt")?;
let parser = StreamingParser::with_config(file, config);

for batch in parser.batches() {
    let triples = batch?;
    println!("Batch: {} triples", triples.len());
}
```

### Parsing TriG

```rust
use oxirs_ttl::trig::TriGParser;
use std::fs::read_to_string;

let content = read_to_string("data/sample.trig")?;
let parser = TriGParser::new();
let quads = parser.parse_str(&content)?;

println!("Parsed {} quads", quads.len());
```

## Testing

These files are automatically used in integration tests:

```bash
# Run tests that use these files
cargo test --test integration_tests

# Run with specific file
cargo test -- sample_ttl
```

## Adding New Test Files

When adding new test files:

1. Use meaningful filenames (e.g., `unicode.ttl`, `large_graph.trig`)
2. Add comments explaining the test purpose
3. Keep files small (<100KB) for unit tests
4. For large file tests, use generated data instead
5. Update this README with file description

## Validation

All sample files can be validated using:

```bash
# Using oxirs CLI (when available)
oxirs validate data/sample.ttl

# Using rapper (if installed)
rapper -i turtle data/sample.ttl
rapper -i ntriples data/sample.nt
rapper -i trig data/sample.trig
rapper -i nquads data/sample.nq
```

## File Statistics

| File | Format | Size | Triples/Quads | Graphs |
|------|--------|------|---------------|--------|
| sample.ttl | Turtle | ~1.5KB | ~25 | 1 (default) |
| sample.nt | N-Triples | ~1.2KB | ~13 | 1 (default) |
| sample.trig | TriG | ~1.8KB | ~20 | 4 named |
| sample.nq | N-Quads | ~2.0KB | ~21 | 3 (1 default + 2 named) |

## License

These test files are part of the OxiRS project and use the same license.
