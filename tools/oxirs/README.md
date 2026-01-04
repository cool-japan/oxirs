# OxiRS CLI

[![Version](https://img.shields.io/badge/version-0.1.0--rc.2-blue)](https://github.com/cool-japan/oxirs/releases)

**Command-line interface for OxiRS semantic web operations**

**Status**: Release Candidate (v0.1.0-rc.2) - Released December 2025

⚡ **Production-Ready**: APIs are stable and tested. Ready for production use with comprehensive documentation.

## Overview

`oxirs` is the unified command-line tool for the OxiRS ecosystem, providing comprehensive functionality for RDF data management, SPARQL operations, server administration, and development workflows. It's designed to be the Swiss Army knife for semantic web developers and data engineers working with knowledge graphs and semantic data.

## What's New in v0.1.0-rc.2 (December 2025)

- **API Stability**: All CLI commands and flags are now stable with semantic versioning guarantees
- **Enhanced Documentation**: Comprehensive help text, examples, and error messages for all commands
- **Production Hardening**: Improved error handling, logging, and resource management
- **Performance Improvements**: Faster query execution, import/export operations, and batch processing
- **Better User Experience**: Enhanced progress indicators, colored output, and interactive prompts
- **Security Enhancements**: Input validation, secure credential handling, and audit logging

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
# Install the latest release candidate
cargo install oxirs --version 0.1.0-rc.2

# Or install with all optional features
cargo install oxirs --version 0.1.0-rc.2 --features all-features
```

### From Source

```bash
git clone https://github.com/cool-japan/oxirs
cd oxirs/tools/oxirs
cargo install --path .

# Or with all features
cargo install --path . --features all-features
```

### Shell Completion

Generate shell completion scripts for your shell:

```bash
# Bash
oxirs --completion bash > ~/.local/share/bash-completion/completions/oxirs

# Zsh
oxirs --completion zsh > ~/.zfunc/_oxirs

# Fish
oxirs --completion fish > ~/.config/fish/completions/oxirs.fish

# PowerShell
oxirs --completion powershell > oxirs.ps1
```

## Quick Start

### Dataset Name Rules

Dataset names must follow these rules:
- **Only alphanumeric characters, underscores (_), and hyphens (-)** are allowed
- **No dots (.), slashes (/), or file extensions** (e.g., `.oxirs`)
- Maximum length: 255 characters
- Cannot be empty

✅ Valid: `mydata`, `my_dataset`, `test-data-2024`
❌ Invalid: `dataset.oxirs`, `my/data`, `data.ttl`

### Basic Usage

```bash
# Initialize a new dataset
oxirs init mydata

# Import RDF data into the dataset
oxirs import mydata data.ttl --format turtle

# Query the data
oxirs query mydata "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Start a SPARQL server
oxirs serve mydata --port 3030

# Export to different format
oxirs export mydata output.jsonld --format json-ld
```

### Interactive Mode

```bash
# Start interactive REPL
oxirs interactive --dataset mydata

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
# Initialize dataset first
oxirs init mydata

# Import single file (dataset name must be alphanumeric, _, - only)
oxirs import mydata data.ttl --format turtle

# Import with named graph
oxirs import mydata data.ttl --format turtle --graph http://example.org/graph1

# Import N-Triples
oxirs import mydata data.nt --format ntriples

# Import RDF/XML
oxirs import mydata data.rdf --format rdfxml

# Import JSON-LD
oxirs import mydata data.jsonld --format jsonld
```

#### Export Data

```bash
# Export entire dataset
oxirs export mydata export.ttl --format turtle

# Export specific graph
oxirs export mydata output.jsonld --format json-ld --graph http://example.org/graph1

# Export to N-Triples
oxirs export mydata output.nt --format ntriples

# Export to RDF/XML
oxirs export mydata output.rdf --format rdfxml
```

#### Validate Data

```bash
# Validate RDF syntax
oxirs rdfparse data.ttl --format turtle

# SHACL validation
oxirs shacl --dataset mydata --shapes shapes.ttl --format text

# ShEx validation
oxirs shex --dataset mydata --schema schema.shex --format text
```

### Query Operations

#### Execute Queries

```bash
# Run SPARQL query
oxirs query mydata "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Run query from file
oxirs query mydata query.sparql --file

# Output formats: table, json, csv, tsv
oxirs query mydata query.sparql --file --output json

# Advanced query with arq tool
oxirs arq --dataset mydata --query "SELECT * WHERE { ?s ?p ?o }" --results table
```

#### Query Analysis

```bash
# Parse and validate SPARQL query
oxirs qparse query.sparql --print-ast

# Show query algebra
oxirs qparse query.sparql --print-algebra

# Parse SPARQL update
oxirs uparse update.sparql --print-ast

# Query optimization analysis (PostgreSQL EXPLAIN-style)
oxirs explain mydata "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
oxirs explain mydata query.sparql --file --mode analyze
oxirs explain mydata query.sparql --file --mode full

# Shows: query structure, complexity score, optimization hints
```

#### Query Templates

```bash
# List all available query templates
oxirs template list

# Filter by category
oxirs template list --category basic
oxirs template list --category aggregation

# Show template details
oxirs template show select-by-type

# Render query from template with parameters
oxirs template render select-by-type \
  --param type_iri=http://xmlns.com/foaf/0.1/Person \
  --param limit=50

# Available templates:
# Basic: select-all, select-by-type, select-with-filter, ask-exists
# Advanced: construct-graph
# Aggregation: count-instances, group-by-count
# PropertyPaths: transitive-closure
# Federation: federated-query
# Analytics: statistics-summary
```

#### Query History

```bash
# List recent queries (automatically tracked)
oxirs history list
oxirs history list --limit 20 --dataset mydata

# Show full query details
oxirs history show 1

# Re-execute a previous query
oxirs history replay 5
oxirs history replay 5 --output json

# Search query history
oxirs history search "SELECT"
oxirs history search "foaf:Person"

# View history statistics
oxirs history stats

# Clear history
oxirs history clear

# History tracks: dataset, query text, execution time,
# result count, success/failure, timestamps
# Stored in: ~/.local/share/oxirs/query_history.json
```

### Server Management

#### Start Server

```bash
# Start SPARQL server with configuration file
oxirs serve mydata/oxirs.toml --port 3030

# With GraphQL support enabled
oxirs serve mydata/oxirs.toml --port 3030 --graphql

# Specify host and port
oxirs serve mydata/oxirs.toml --host 0.0.0.0 --port 8080
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

### Dataset Configuration

When you run `oxirs init mydata`, it creates a configuration file at `mydata/oxirs.toml`:

```toml
# OxiRS Configuration
# Generated by oxirs init

[general]
default_format = "turtle"

[server]
port = 3030
host = "localhost"
enable_cors = true
enable_graphql = false

[datasets.mydata]
name = "mydata"
location = "."
dataset_type = "tdb2"
read_only = false
enable_reasoning = false
enable_validation = false
enable_text_search = false
enable_vector_search = false
```

### Configuration Fields

- `general.default_format`: Default RDF serialization format
- `server.port`: HTTP server port
- `server.host`: Server bind address
- `server.enable_graphql`: Enable GraphQL endpoint
- `datasets.{name}.location`: Storage path (`.` means dataset directory itself)
- `datasets.{name}.dataset_type`: Storage backend (`tdb2` or `memory`)
- `datasets.{name}.read_only`: Prevent modifications
- Feature flags: `enable_reasoning`, `enable_validation`, `enable_text_search`, `enable_vector_search`

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

---

## Best Practices

### Command Cheat Sheet

```bash
# Quick reference for common tasks

# Data Operations
oxirs init mydata                              # Create new dataset
oxirs import mydata file.ttl -f turtle         # Import data
oxirs export mydata output.nq -f nquads        # Export data
oxirs query mydata "SELECT * WHERE {?s ?p ?o}" # Query data

# Server Operations
oxirs serve mydata --port 3030                 # Start server
oxirs serve mydata --graphql                   # With GraphQL

# Format Conversion
oxirs migrate --source data.ttl --target data.nt --from turtle --to ntriples

# Validation
oxirs rdfparse file.ttl -f turtle              # Validate syntax

# Analysis
oxirs explain mydata query.sparql --file       # Query analysis
oxirs tdbstats mydata --detailed               # Dataset statistics
```

### Performance Tips

**For Large Datasets (>1M triples)**:
- Use batch import with parallel processing: `oxirs batch import --dataset mydata --files *.nt --parallel 8`
- Use TDB loader for bulk loading: `oxirs tdbloader mydata *.nt --progress --stats`
- Stream large exports: `oxirs export mydata output.nq --format nquads | gzip > output.nq.gz`

**For Query Performance**:
- Analyze queries before execution: `oxirs explain mydata query.sparql --file --mode full`
- Use appropriate output format (JSON for programmatic use, table for humans)
- Enable query caching for repeated queries

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Dataset not found" | Run `oxirs init <name>` first to create the dataset |
| "Format not recognized" | Specify format explicitly with `--format` flag |
| "Permission denied" | Check directory permissions with `chmod 755 <dir>` |
| "Port already in use" | Use different port with `--port <num>` |
| "Out of memory" | Use streaming operations or increase batch size |
| "Invalid SPARQL syntax" | Use `oxirs qparse` to validate query syntax |

**Debug Mode**:
```bash
# Enable verbose logging
oxirs --verbose query mydata "SELECT * WHERE {?s ?p ?o}"

# Debug specific modules
RUST_LOG=oxirs_core=debug,oxirs_arq=trace oxirs query mydata query.sparql
```

---

**OxiRS CLI v0.1.0-rc.2** - Production-ready command-line interface for semantic web operations
