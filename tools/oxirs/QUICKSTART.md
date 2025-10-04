# OxiRS Quick Start Guide

## Alpha.2 Release - Production-Ready Features

### Installation

```bash
cargo install oxirs
# or build from source
cargo build --release
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

#### Benchmark Results (Alpha.2)

| Operation | Dataset Size | Throughput |
|-----------|--------------|------------|
| N-Quads Serialization | 10,000 quads | 10.5 Melem/s |
| N-Triples Serialization | 10,000 quads | 2.8 Melem/s |
| Turtle Serialization | 10,000 quads | 3.1 Melem/s |
| N-Triples Parsing | 10,000 quads | 843 Kelem/s |
| Format Conversion | 10,000 quads | 716 Kelem/s |

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

#### Alpha.2 Highlights

✅ **Production-Ready**
- Real SPARQL query execution
- Complete RDF import/export pipeline
- 7 RDF format support (Turtle, N-Triples, N-Quads, TriG, RDF/XML, JSON-LD, N3)
- High-performance N-Triples/N-Quads parser

✅ **Performance**
- Parallel batch operations
- Streaming architecture
- Memory-efficient processing
- Benchmark suite with real metrics

✅ **Quality**
- 7/7 integration tests passing
- 3,750+ unit tests
- Zero compilation warnings
- Comprehensive error handling

✅ **Observability**
- Progress tracking
- Performance metrics
- Health endpoints
- Prometheus integration

### Next Steps

- Explore [Full Documentation](README.md)
- Check [Examples](examples/)
- Review [Configuration Guide](oxirs.toml.example)
- Join [Community](https://github.com/cool-japan/oxirs)

---

**OxiRS Alpha.2** - Production-ready SPARQL 1.2 server with AI augmentation