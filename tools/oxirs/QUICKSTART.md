# OxiRS Quick Start Guide

## Beta.1 Release - Production-Ready with API Stability

### Installation

```bash
# Install from crates.io
cargo install oxirs --version 0.1.0-beta.1

# Or build from source
git clone https://github.com/cool-japan/oxirs
cd oxirs
cargo build --release --bin oxirs
```

### Basic Commands

#### 1. Query - SPARQL Query Execution

Execute SPARQL queries with real query engine integration:

```bash
# Query from file
oxirs query --dataset ./data/mydata --query query.sparql

# Inline query
oxirs query --dataset ./data/mydata --query "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Different output formats
oxirs query --dataset ./data/mydata --query query.sparql --format json
oxirs query --dataset ./data/mydata --query query.sparql --format csv
oxirs query --dataset ./data/mydata --query query.sparql --format xml
```

#### 2. Import - Load RDF Data

Import RDF data in any of 7 supported formats:

```bash
# Import Turtle file
oxirs import --dataset ./data/mydata --file data.ttl --format turtle

# Import N-Triples with specific graph
oxirs import --dataset ./data/mydata --file data.nt --format ntriples --graph http://example.org/graph1

# Auto-detect format from extension
oxirs import --dataset ./data/mydata --file data.ttl
```

Supported formats:
- `turtle` (.ttl)
- `ntriples` (.nt)
- `nquads` (.nq)
- `trig` (.trig)
- `rdfxml` (.rdf, .xml)
- `jsonld` (.jsonld)
- `n3` (.n3)

#### 3. Export - Serialize RDF Data

Export data in any RDF format:

```bash
# Export to Turtle
oxirs export --dataset ./data/mydata --file output.ttl --format turtle

# Export specific graph
oxirs export --dataset ./data/mydata --file graph1.nt --format ntriples --graph http://example.org/graph1

# Export to N-Quads (includes all graphs)
oxirs export --dataset ./data/mydata --file all-data.nq --format nquads
```

#### 4. Migrate - Convert Between Formats

High-performance streaming format conversion:

```bash
# Convert Turtle to N-Triples
oxirs migrate --source data.ttl --target data.nt --from turtle --to ntriples

# Convert N-Quads to TriG
oxirs migrate --source data.nq --target data.trig --from nquads --to trig

# All 49 format combinations supported (7x7)
```

Features:
- **Streaming architecture** - no intermediate storage
- **Memory efficient** - processes large files
- **Progress tracking** - real-time feedback
- **Error statistics** - comprehensive reporting

#### 5. Batch - Parallel Bulk Import

Import multiple files with parallel processing:

```bash
# Import multiple files (auto-detect formats)
oxirs batch import --dataset ./data/mydata --files file1.ttl file2.nt file3.nq

# Specify format for all files
oxirs batch import --dataset ./data/mydata --files *.ttl --format turtle

# Control parallelism (default: 4 workers)
oxirs batch import --dataset ./data/mydata --files *.nt --parallel 8

# Import to specific graph
oxirs batch import --dataset ./data/mydata --files *.ttl --graph http://example.org/batch
```

Features:
- **Parallel processing** - configurable worker threads
- **Auto-format detection** - from file extensions
- **Thread-safe** - concurrent store access
- **Per-file errors** - isolated error handling
- **Global statistics** - aggregated metrics

#### 6. Update - SPARQL Update Operations

Execute SPARQL UPDATE operations:

```bash
# Update from file
oxirs update --dataset ./data/mydata --update update.sparql

# Inline update
oxirs update --dataset ./data/mydata --update "INSERT DATA { <s> <p> <o> }"
```

#### 7. Serve - HTTP SPARQL Server

Start production-ready SPARQL server:

```bash
# Start with configuration file
oxirs serve --config oxirs.toml

# Custom host and port
oxirs serve --config oxirs.toml --host 0.0.0.0 --port 8080

# Enable GraphQL endpoint
oxirs serve --config oxirs.toml --graphql
```

Server endpoints:
- SPARQL Query: `http://localhost:3030/sparql`
- SPARQL Update: `http://localhost:3030/update`
- GraphQL: `http://localhost:3030/graphql` (if enabled)
- Health: `http://localhost:3030/health/live`
- Metrics: `http://localhost:3030/metrics`

### Configuration

Copy the example configuration:

```bash
cp oxirs.toml.example oxirs.toml
```

Edit `oxirs.toml` to configure:
- Server settings (host, port, CORS)
- Dataset locations and types
- Authentication (JWT, OAuth2, Basic)
- Tool-specific settings

### Performance

#### Benchmark Results (Beta.1)

| Operation | Dataset Size | Throughput | Improvement |
|-----------|--------------|------------|-------------|
| N-Quads Serialization | 10,000 quads | 12.8 Melem/s | +22% |
| N-Triples Serialization | 10,000 quads | 3.4 Melem/s | +21% |
| Turtle Serialization | 10,000 quads | 3.7 Melem/s | +19% |
| N-Triples Parsing | 10,000 quads | 1.1 Melem/s | +30% |
| Format Conversion | 10,000 quads | 920 Kelem/s | +29% |

Performance improvements in Beta.1:
- Optimized memory allocation for parsing and serialization
- SIMD-accelerated operations for string processing
- Parallel batch operations with work-stealing
- Efficient caching for repeated operations

### Examples

#### Complete Workflow

```bash
# 1. Create dataset directory
mkdir -p ./data/myproject

# 2. Import initial data
oxirs import --dataset ./data/myproject --file schema.ttl --format turtle

# 3. Batch import multiple files
oxirs batch import --dataset ./data/myproject --files data/*.nt --parallel 8

# 4. Query the data
oxirs query --dataset ./data/myproject --query "
  PREFIX ex: <http://example.org/>
  SELECT ?name WHERE {
    ?person ex:name ?name .
  }
" --format json

# 5. Export for backup
oxirs export --dataset ./data/myproject --file backup.nq --format nquads

# 6. Convert format
oxirs migrate --source backup.nq --target backup.ttl --from nquads --to turtle

# 7. Start server
oxirs serve --config oxirs.toml
```

#### Server Usage

```bash
# Query via HTTP
curl -X POST http://localhost:3030/sparql \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Update via HTTP
curl -X POST http://localhost:3030/update \
  -H "Content-Type: application/sparql-update" \
  -d "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"

# Health check
curl http://localhost:3030/health/live

# Metrics (Prometheus format)
curl http://localhost:3030/metrics
```

### Testing

Run integration tests:

```bash
cargo test --test integration_rdf_pipeline
```

Run performance benchmarks:

```bash
cargo bench --bench cli_performance
```

### Troubleshooting

#### Common Issues

**Dataset not found:**
```bash
# Ensure dataset directory exists
mkdir -p ./data/mydata
```

**Format not recognized:**
```bash
# Specify format explicitly
oxirs import --dataset ./data/mydata --file data.txt --format turtle
```

**Permission denied:**
```bash
# Check directory permissions
chmod 755 ./data
```

**Server port in use:**
```bash
# Use different port
oxirs serve --config oxirs.toml --port 3031
```

### Features

#### Beta.1 Highlights

✅ **Production-Ready**
- API stability with semantic versioning guarantees
- Complete SPARQL 1.1/1.2 query and update support
- 7 RDF format support (Turtle, N-Triples, N-Quads, TriG, RDF/XML, JSON-LD, N3)
- High-performance parsing and serialization (20-30% faster than alpha.2)

✅ **Enhanced User Experience**
- Comprehensive help text for all commands
- Interactive prompts with validation
- Color-coded output and progress indicators
- Shell completion for Bash, Zsh, Fish, PowerShell

✅ **Security & Reliability**
- Input validation and sanitization
- Secure credential handling
- Comprehensive error messages with suggestions
- Audit logging for all operations

✅ **Performance**
- SIMD-accelerated string operations
- Parallel batch operations with work-stealing
- Memory-efficient streaming architecture
- 20-30% performance improvement over alpha.2

✅ **Quality**
- 95%+ test coverage
- 8,690+ tests passing
- Zero compilation warnings
- Comprehensive integration tests

✅ **Observability**
- Structured logging with tracing support
- Performance metrics and profiling
- Health endpoints for monitoring
- Prometheus-compatible metrics

### What's New in Beta.1

**API Stability**:
- All commands and flags are now stable
- Semantic versioning guarantees
- Deprecation warnings for future changes

**Enhanced Commands**:
- `oxirs explain` - Query execution plan analysis
- `oxirs template` - Reusable SPARQL query templates
- `oxirs history` - Persistent query history management
- `oxirs validate` - Enhanced RDF validation with SHACL/ShEx

**Better Error Messages**:
- Clear, actionable error messages
- Suggestions for fixing common issues
- Context-aware help text

**Performance Improvements**:
- 20-30% faster parsing and serialization
- Optimized memory usage
- Parallel processing for batch operations

### Next Steps

- Explore [Full Documentation](README.md)
- Check [Examples](examples/)
- Review [Configuration Guide](oxirs.toml.example)
- Join [Community](https://github.com/cool-japan/oxirs)

See [CHANGELOG.md](../../CHANGELOG.md) for detailed release notes.

---

**OxiRS Beta.1** - Production-ready SPARQL 1.2 server with AI augmentation and stable APIs