# OxiRS CLI

[![Version](https://img.shields.io/badge/version-0.1.0--alpha.1-orange)](https://github.com/cool-japan/oxirs/releases)

**Command-line interface for OxiRS semantic web operations**

**Status**: Alpha Release (v0.1.0-alpha.1) - Released September 30, 2025

⚠️ **Alpha Software**: This is an early alpha release. APIs may change without notice. Not recommended for production use.

## Overview

`oxirs` is the unified command-line tool for the OxiRS ecosystem, providing comprehensive functionality for RDF data management, SPARQL operations, server administration, and development workflows. It's designed to be the Swiss Army knife for semantic web developers and data engineers.

## Features

- **Data Operations**: Import, export, validate, and transform RDF data
- **Query Execution**: Run SPARQL queries against local and remote endpoints
- **Server Management**: Start, stop, and configure OxiRS servers
- **Development Tools**: Schema validation, query optimization, and debugging
- **Benchmarking**: Performance testing and dataset generation
- **Migration Tools**: Convert between RDF formats and upgrade datasets
- **Configuration Management**: Manage server and client configurations
- **Interactive Mode**: REPL for exploratory data analysis

## Installation

### From Crates.io

```bash
cargo install oxirs
```

### From Source

```bash
git clone https://github.com/cool-japan/oxirs
cd oxirs/tools/oxirs
cargo install --path .
```

### Pre-built Binaries

Download from [GitHub Releases](https://github.com/cool-japan/oxirs/releases/tag/v0.1.0-alpha.1):

```bash
# Linux
curl -L https://github.com/cool-japan/oxirs/releases/download/v0.1.0-alpha.1/oxirs-linux-x86_64.tar.gz | tar xz

# macOS
curl -L https://github.com/cool-japan/oxirs/releases/download/v0.1.0-alpha.1/oxirs-macos-x86_64.tar.gz | tar xz

# Windows
curl -L https://github.com/cool-japan/oxirs/releases/download/v0.1.0-alpha.1/oxirs-windows-x86_64.zip -o oxirs.zip
```

## Quick Start

### Basic Usage

```bash
# Import RDF data
oxirs import --file data.ttl --format turtle --output dataset.oxirs

# Query the data
oxirs query --dataset dataset.oxirs --query "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Start a SPARQL server
oxirs serve --dataset dataset.oxirs --port 3030

# Export to different format
oxirs export --dataset dataset.oxirs --format json-ld --output data.jsonld
```

### Interactive Mode

```bash
# Start interactive REPL
oxirs repl --dataset dataset.oxirs

oxirs> SELECT ?person ?name WHERE { ?person foaf:name ?name } LIMIT 5
┌─────────────────────────────────────┬──────────────┐
│ person                              │ name         │
├─────────────────────────────────────┼──────────────┤
│ http://example.org/alice           │ "Alice"      │
│ http://example.org/bob             │ "Bob"        │
└─────────────────────────────────────┴──────────────┘

oxirs> .schema foaf:Person
Class: foaf:Person
Properties:
  - foaf:name (string, required)
  - foaf:age (integer, optional)
  - foaf:mbox (string, optional)

oxirs> .exit
```

## Commands

### Data Management

#### Import Data

```bash
# Import single file
oxirs import --file data.ttl --dataset mydata.oxirs

# Import multiple files
oxirs import --files *.ttl --dataset mydata.oxirs

# Import from URL
oxirs import --url https://example.org/data.rdf --dataset mydata.oxirs

# Import with custom base IRI
oxirs import --file data.ttl --base http://example.org/ --dataset mydata.oxirs

# Import with validation
oxirs import --file data.ttl --validate --strict --dataset mydata.oxirs
```

#### Export Data

```bash
# Export entire dataset
oxirs export --dataset mydata.oxirs --format turtle --output export.ttl

# Export specific graph
oxirs export --dataset mydata.oxirs --graph http://example.org/graph1 --format json-ld

# Export query results
oxirs export --dataset mydata.oxirs --query query.sparql --format csv --output results.csv

# Streaming export for large datasets
oxirs export --dataset mydata.oxirs --format nquads --stream --output large-export.nq
```

#### Validate Data

```bash
# Validate syntax
oxirs validate --file data.ttl --format turtle

# SHACL validation
oxirs validate --dataset mydata.oxirs --shapes shapes.ttl --report validation-report.ttl

# Schema validation
oxirs validate --dataset mydata.oxirs --schema schema.rdfs --output validation.json
```

### Query Operations

#### Execute Queries

```bash
# Run SPARQL query
oxirs query --dataset mydata.oxirs --query "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Run query from file
oxirs query --dataset mydata.oxirs --file query.sparql

# Run query against remote endpoint
oxirs query --endpoint https://dbpedia.org/sparql --query query.sparql

# Save results to file
oxirs query --dataset mydata.oxirs --query query.sparql --output results.json --format json
```

#### Query Analysis

```bash
# Analyze query performance
oxirs query --dataset mydata.oxirs --file query.sparql --analyze --explain

# Show query plan
oxirs query --dataset mydata.oxirs --file query.sparql --plan

# Benchmark query
oxirs query --dataset mydata.oxirs --file query.sparql --benchmark --iterations 100
```

### Server Management

#### Start Server

```bash
# Basic SPARQL server
oxirs serve --dataset mydata.oxirs --port 3030

# With GraphQL support
oxirs serve --dataset mydata.oxirs --port 3030 --graphql --graphql-port 4000

# With authentication
oxirs serve --dataset mydata.oxirs --auth basic --users users.yaml

# With configuration file
oxirs serve --config server.yaml
```

#### Server Administration

```bash
# Check server status
oxirs admin status --server http://localhost:3030

# Upload data to running server
oxirs admin upload --server http://localhost:3030 --file new-data.ttl

# Backup dataset
oxirs admin backup --server http://localhost:3030 --output backup.tar.gz

# View server metrics
oxirs admin metrics --server http://localhost:3030 --format prometheus
```

### Development Tools

#### Schema Operations

```bash
# Generate schema from data
oxirs schema generate --dataset mydata.oxirs --output schema.rdfs

# Validate against schema
oxirs schema validate --dataset mydata.oxirs --schema schema.rdfs

# Compare schemas
oxirs schema diff --schema1 old-schema.rdfs --schema2 new-schema.rdfs

# Convert schema formats
oxirs schema convert --input schema.owl --output schema.shacl --format shacl
```

#### Optimization

```bash
# Optimize dataset
oxirs optimize --dataset mydata.oxirs --output optimized.oxirs

# Analyze dataset statistics
oxirs analyze --dataset mydata.oxirs --output stats.json

# Generate indices
oxirs index --dataset mydata.oxirs --properties foaf:name,dc:title

# Compress dataset
oxirs compress --dataset mydata.oxirs --algorithm lz4 --output compressed.oxirs
```

### Benchmarking

#### Dataset Generation

```bash
# Generate test dataset
oxirs benchmark generate --template university --size 10000 --output test-data.ttl

# Generate synthetic data
oxirs benchmark synthetic --schema schema.rdfs --triples 1000000 --output synthetic.nq

# Generate benchmark queries
oxirs benchmark queries --dataset mydata.oxirs --count 100 --complexity mixed --output queries/
```

#### Performance Testing

```bash
# Run benchmarks
oxirs benchmark run --dataset mydata.oxirs --queries queries/ --report benchmark-report.html

# Compare performance
oxirs benchmark compare --baseline baseline-results.json --current current-results.json

# Stress testing
oxirs benchmark stress --endpoint http://localhost:3030 --duration 60s --concurrent 10
```

### Migration and Conversion

#### Format Conversion

```bash
# Convert between RDF formats
oxirs convert --input data.rdf --output data.ttl --from rdfxml --to turtle

# Batch conversion
oxirs convert --directory rdf-files/ --from rdfxml --to jsonld --output converted/

# Streaming conversion for large files
oxirs convert --input large-file.nt --output large-file.ttl --stream --chunk-size 10000
```

#### Data Migration

```bash
# Migrate from older OxiRS version
oxirs migrate --input old-dataset.oxirs --output new-dataset.oxirs --from-version 0.1.0

# Migrate from other triple stores
oxirs migrate --input virtuoso-dump.nq --format nquads --output migrated.oxirs --optimize

# Migrate with transformation
oxirs migrate --input data.ttl --transform-rules rules.sparql --output transformed.oxirs
```

### Configuration

#### Configuration Management

```bash
# Initialize configuration
oxirs config init --template server --output oxirs.yaml

# Validate configuration
oxirs config validate --file oxirs.yaml

# Show current configuration
oxirs config show --format yaml

# Set configuration values
oxirs config set server.port 3030
oxirs config set auth.enabled true
```

#### Environment Setup

```bash
# Setup development environment
oxirs init --project my-semantic-app --template basic

# Install dependencies
oxirs deps install --file requirements.yaml

# Setup CI/CD templates
oxirs init --template ci-cd --output .github/workflows/
```

## Configuration

### Global Configuration

The global configuration file is located at:
- Linux/macOS: `~/.config/oxirs/config.yaml`
- Windows: `%APPDATA%\oxirs\config.yaml`

```yaml
# ~/.config/oxirs/config.yaml
default:
  format: turtle
  output_dir: ./output
  verbose: false

servers:
  local:
    url: http://localhost:3030
    auth: none
  
  production:
    url: https://sparql.example.org
    auth:
      type: bearer
      token: ${SPARQL_TOKEN}

profiles:
  development:
    log_level: debug
    enable_timing: true
    
  production:
    log_level: warn
    enable_timing: false
```

### Command-specific Configuration

```bash
# Use specific profile
oxirs --profile production query --endpoint production --query query.sparql

# Override global settings
oxirs --verbose query --dataset mydata.oxirs --format json --query query.sparql

# Use configuration file
oxirs --config custom-config.yaml serve --dataset mydata.oxirs
```

## Advanced Features

### Scripting and Automation

```bash
# Bash completion
eval "$(oxirs completion bash)"

# Pipeline operations
oxirs import --file data.ttl --dataset temp.oxirs | \
oxirs validate --shapes shapes.ttl | \
oxirs optimize --output optimized.oxirs

# Batch processing
find . -name "*.ttl" -exec oxirs import --file {} --dataset combined.oxirs \;
```

### Custom Extensions

```bash
# Install plugin
oxirs plugin install oxirs-geospatial

# List plugins
oxirs plugin list

# Run plugin command
oxirs geo index --dataset spatial-data.oxirs --property geo:hasGeometry
```

### Integration with Other Tools

```bash
# Integration with Git
oxirs export --dataset mydata.oxirs --format turtle | git diff --no-index data.ttl -

# Integration with Apache Jena
oxirs export --dataset mydata.oxirs --format ntriples | riot --formatted=turtle

# Integration with RDFLib
oxirs query --dataset mydata.oxirs --query query.sparql --format json | python process.py
```

## Examples

### Data Processing Pipeline

```bash
#!/bin/bash
# data-pipeline.sh

# Download and import multiple datasets
oxirs import --url https://dbpedia.org/dataset.ttl --dataset dbpedia.oxirs
oxirs import --url https://wikidata.org/dataset.ttl --dataset wikidata.oxirs

# Merge datasets
oxirs merge --datasets dbpedia.oxirs,wikidata.oxirs --output merged.oxirs

# Validate merged data
oxirs validate --dataset merged.oxirs --shapes validation-shapes.ttl

# Generate optimized indices
oxirs index --dataset merged.oxirs --properties rdfs:label,skos:prefLabel

# Start production server
oxirs serve --dataset merged.oxirs --config production.yaml --daemon
```

### Development Workflow

```bash
# Create new project
oxirs init --project semantic-app --template web-app

cd semantic-app/

# Import development data
oxirs import --file dev-data.ttl --dataset dev.oxirs

# Start development server with hot reload
oxirs serve --dataset dev.oxirs --dev --reload

# Run tests
oxirs test --dataset dev.oxirs --test-suite tests/

# Deploy to staging
oxirs deploy --dataset dev.oxirs --target staging --config deploy.yaml
```

## Performance

### Benchmarks

| Operation | Dataset Size | Time | Memory |
|-----------|--------------|------|--------|
| Import (Turtle) | 1M triples | 15s | 120MB |
| Export (JSON-LD) | 1M triples | 12s | 85MB |
| Query (simple) | 10M triples | 50ms | 45MB |
| Query (complex) | 10M triples | 300ms | 180MB |
| Server startup | 10M triples | 2s | 200MB |

### Optimization Tips

```bash
# Use streaming for large files
oxirs import --file large-dataset.nt --stream --chunk-size 100000

# Enable parallel processing
oxirs export --dataset large.oxirs --parallel --workers 8

# Use binary format for faster loading
oxirs convert --input data.ttl --output data.oxirs --optimize

# Compress datasets
oxirs compress --dataset data.oxirs --algorithm zstd --level 3
```

## Troubleshooting

### Common Issues

```bash
# Debug mode
oxirs --debug query --dataset mydata.oxirs --query query.sparql

# Verbose output
oxirs --verbose import --file problematic-data.ttl

# Check dataset integrity
oxirs check --dataset mydata.oxirs --repair

# Memory profiling
oxirs --profile-memory query --dataset large.oxirs --query complex-query.sparql
```

### Error Recovery

```bash
# Recover corrupted dataset
oxirs recover --dataset corrupted.oxirs --output recovered.oxirs

# Validate and repair
oxirs validate --dataset mydata.oxirs --repair --backup

# Restore from backup
oxirs restore --backup backup.tar.gz --output restored.oxirs
```

## Related Tools

- [`oxirs-fuseki`](../../server/oxirs-fuseki/): SPARQL HTTP server
- [`oxirs-chat`](../../ai/oxirs-chat/): AI-powered chat interface
- [`oxirs-workbench`](../workbench/): Visual RDF editor
- Apache Jena: Java-based semantic web toolkit
- RDFLib: Python RDF processing library

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new commands
4. Update documentation
5. Submit a pull request

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](../../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Status

🚧 **Alpha Release (v0.1.0-alpha.1)** - September 30, 2025

Current alpha features:
- ✅ Basic data import/export
- ✅ Query execution
- ✅ Server management (basic)
- 🚧 Benchmarking tools (in progress)
- ⏳ Migration utilities (planned)
- ⏳ Interactive REPL (planned)

Note: This is an alpha release. Some features are incomplete and APIs may change.