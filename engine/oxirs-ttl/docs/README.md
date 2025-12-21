# OxiRS TTL Documentation

Comprehensive documentation for the oxirs-ttl RDF parser and serializer.

## Quick Links

- [README](../README.md) - Project overview and quick start
- [API Documentation](https://docs.rs/oxirs-ttl) - Full API reference
- [GitHub Repository](https://github.com/cool-japan/oxirs/tree/main/engine/oxirs-ttl)

## Tutorials

### For Beginners

- **[README.md](../README.md#quick-start)** - Start here for basic usage examples
  - Basic Turtle parsing
  - Format auto-detection
  - N-Triples, TriG, N-Quads parsing
  - Simple serialization

### For Advanced Users

- **[Streaming Tutorial](STREAMING_TUTORIAL.md)** - Memory-efficient large file processing
  - Why streaming?
  - Basic and advanced streaming
  - Progress tracking
  - Error handling in streams
  - Performance optimization
  - Real-world examples (database import, format conversion, statistical analysis)

- **[Async Usage Guide](ASYNC_GUIDE.md)** - Non-blocking I/O with Tokio
  - Why async?
  - Basic async parsing
  - Async streaming
  - Concurrent parsing
  - Network integration (HTTP, WebSocket, S3, Kafka)
  - Error handling and timeouts
  - Performance optimization

- **[Performance Tuning Guide](PERFORMANCE_GUIDE.md)** - Optimization techniques
  - Performance overview
  - Profiling and measurement
  - Parse performance (zero-copy, SIMD, lazy resolution)
  - Serialization performance
  - Memory optimization
  - I/O optimization
  - Parallel processing
  - Platform-specific tuning
  - Benchmarking

## Feature Guides

### Core Features

| Feature | Documentation | Description |
|---------|---------------|-------------|
| **Turtle Parsing** | [README](../README.md#basic-turtle-parsing) | Full Turtle 1.1 support |
| **N-Triples** | [README](../README.md#n-triples-parsing) | Line-based simple format |
| **TriG** | [README](../README.md#trig-parsing-named-graphs) | Named graphs extension |
| **N-Quads** | [README](../README.md#n-quads-parsing) | Quad-based format |
| **RDF 1.2** | [README](../README.md#rdf-12-support-) | Quoted triples & directional tags |

### Advanced Features

| Feature | Documentation | Description |
|---------|---------------|-------------|
| **Streaming** | [Streaming Tutorial](STREAMING_TUTORIAL.md) | Memory-efficient batch processing |
| **Async I/O** | [Async Guide](ASYNC_GUIDE.md) | Non-blocking Tokio integration |
| **Parallel Processing** | [Performance Guide](PERFORMANCE_GUIDE.md#parallel-processing) | Multi-threaded parsing |
| **Error Recovery** | [README](../README.md#error-recovery-lenient-mode) | Lenient mode for production |
| **Incremental Parsing** | [README](../README.md#incremental-parsing) | Parse as bytes arrive |
| **Format Detection** | [README](../README.md#automatic-format-detection) | Auto-detect RDF formats |
| **Serialization Optimizations** | [README](../README.md#serialization-optimizations) | Compact output (predicate grouping, collections) |

## By Use Case

### I want to...

#### Parse RDF Files

- **Small files (<10MB)**: [Basic Turtle Parsing](../README.md#basic-turtle-parsing)
- **Large files (>10MB)**: [Streaming Tutorial](STREAMING_TUTORIAL.md#basic-streaming)
- **Very large files (>1GB)**: [Streaming Tutorial](STREAMING_TUTORIAL.md#example-4-memory-constrained-environment)

#### Serialize RDF Data

- **Basic serialization**: [README - Serialization](../README.md#serialization-with-pretty-printing)
- **Compact output**: [README - Optimized Serialization](../README.md#optimized-serialization)
- **Large datasets**: [Performance Guide - Streaming Serialization](PERFORMANCE_GUIDE.md#streaming-serialization)

#### Build Web Services

- **HTTP endpoints**: [Async Guide - Web Service](ASYNC_GUIDE.md#example-1-async-web-service)
- **WebSocket streaming**: [Async Guide - WebSocket](ASYNC_GUIDE.md#websocket-rdf-streaming)
- **REST API**: [Async Guide - Network Integration](ASYNC_GUIDE.md#network-integration)

#### Process Multiple Files

- **Sequential**: [Streaming Tutorial - Basic Streaming](STREAMING_TUTORIAL.md#streaming-turtle-files)
- **Concurrent**: [Async Guide - Concurrent Parsing](ASYNC_GUIDE.md#concurrent-parsing)
- **Parallel**: [Performance Guide - Parallel Processing](PERFORMANCE_GUIDE.md#parallel-processing)

#### Import to Database

- **PostgreSQL**: [Streaming Tutorial - Database Import](STREAMING_TUTORIAL.md#example-1-database-import)
- **Batch inserts**: [Streaming Tutorial - Advanced Configuration](STREAMING_TUTORIAL.md#advanced-configuration)
- **Real-time updates**: [Async Guide - Kafka](ASYNC_GUIDE.md#example-3-real-time-rdf-updates)

#### Convert Formats

- **Turtle to N-Quads**: [Streaming Tutorial - Format Conversion](STREAMING_TUTORIAL.md#example-2-rdf-format-conversion)
- **Batch conversion**: [Performance Guide - Parallel Conversion](PERFORMANCE_GUIDE.md#parallel-format-conversion)

#### Optimize Performance

- **Profiling**: [Performance Guide - Profiling](PERFORMANCE_GUIDE.md#profiling-and-measurement)
- **Memory optimization**: [Performance Guide - Memory Optimization](PERFORMANCE_GUIDE.md#memory-optimization)
- **I/O optimization**: [Performance Guide - I/O Optimization](PERFORMANCE_GUIDE.md#io-optimization)
- **Benchmarking**: [Performance Guide - Benchmarking](PERFORMANCE_GUIDE.md#benchmarking)

## Code Examples

### Quick Reference

```rust
// Basic parsing
use oxirs_ttl::turtle::TurtleParser;
let parser = TurtleParser::new();
let triples = parser.parse_file("data.ttl")?;

// Streaming
use oxirs_ttl::StreamingParser;
let parser = StreamingParser::new(file);
for batch in parser.batches() {
    let triples = batch?;
    // Process batch
}

// Async
use oxirs_ttl::async_parser::AsyncTurtleParser;
let file = File::open("data.ttl").await?;
let parser = AsyncTurtleParser::new();
let triples = parser.parse_async(file).await?;

// Serialization
use oxirs_ttl::turtle::TurtleSerializer;
let serializer = TurtleSerializer::new();
let turtle = serializer.serialize_optimized(&triples)?;
```

See individual tutorials for comprehensive examples.

## Performance Reference

### Throughput

| Format | Parse Speed | Serialize Speed |
|--------|-------------|-----------------|
| Turtle | 250-300K triples/s | 180-200K triples/s |
| N-Triples | 400-500K triples/s | 350-400K triples/s |
| TriG | 200-250K triples/s | 160-180K triples/s |
| N-Quads | 350-450K triples/s | 300-350K triples/s |

*Measured on Apple M1 with typical RDF datasets*

### Memory Usage

| Dataset Size | Streaming | In-Memory |
|--------------|-----------|-----------|
| 100MB | ~50MB | ~300MB |
| 1GB | ~50MB | ~3GB |
| 10GB | ~50MB | OOM |

*With 10K batch size for streaming*

## W3C Compliance

| Test Suite | Pass Rate | Status |
|------------|-----------|--------|
| W3C Turtle | 97% (33/34) | ✅ Excellent |
| W3C TriG | 94% (33/35) | ✅ Excellent |
| RDF 1.2 | 100% (19/19) | ✅ Complete |

## Testing

- **461 tests passing** (437 integration + 24 doc tests)
- **Property-based testing** with proptest
- **Memory leak tests** for production safety
- **Performance regression tests** for baseline tracking
- **Fuzzing infrastructure** for robustness

## Contributing

See main [README.md](../README.md#contributing) for contribution guidelines.

## Support

- **Issues**: [GitHub Issues](https://github.com/cool-japan/oxirs/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cool-japan/oxirs/discussions)
- **Documentation**: [API Docs](https://docs.rs/oxirs-ttl)

## License

MIT OR Apache-2.0

## See Also

- [oxirs-core](../../../core/oxirs-core/) - RDF data model
- [oxirs-star](../../oxirs-star/) - RDF-star support
- [oxirs-arq](../../oxirs-arq/) - SPARQL query engine
- [W3C Turtle Spec](https://www.w3.org/TR/turtle/)
- [W3C TriG Spec](https://www.w3.org/TR/trig/)
- [RDF 1.2 Spec](https://www.w3.org/TR/rdf12-concepts/)
