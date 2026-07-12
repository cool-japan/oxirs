# OxiRS CLI

[![Version](https://img.shields.io/badge/version-0.3.2-blue)](https://github.com/cool-japan/oxirs/releases)

**Command-line interface for OxiRS semantic web operations**

**Status**: v0.3.2 — in development on branch `0.3.2`, last verified 2026-07-12

**Tests**: 1799 passed, 0 failed (`cargo nextest run -p oxirs --all-features`)

⚡ **Production-Ready**: APIs are stable and tested. Ready for production use with comprehensive documentation.

## Overview

`oxirs` is the unified command-line tool for the OxiRS ecosystem, providing comprehensive functionality for RDF data management, SPARQL operations, server administration, and development workflows. It's designed to be the Swiss Army knife for semantic web developers and data engineers working with knowledge graphs and semantic data.

This crate is an end-user CLI application, not a published library: `Cargo.toml` sets `publish = false` so it can depend on `publish = false` quarantine adapter crates (for optional GPU/CUDA, GEOS, DuckDB, and Kafka/Pulsar integrations) without putting C FFI on the published Pure-Rust dependency surface. It ships as a release binary built from source rather than via `cargo install oxirs` from crates.io — see [Installation](#installation).

## What's New in v0.3.2 (2026-07-11)

- **Pure-Rust Policy v2**: six previously in-tree C-FFI integrations (NVIDIA GPU monitoring, GEOS, DuckDB, Kafka, Pulsar) were extracted into separate `publish = false` adapter crates. This CLI's optional `tsdb-duckdb` feature now depends on the `oxirs-tsdb-adapter-duckdb` bridge crate instead of an in-tree DuckDB integration
- **SHACL improvements reach `oxirs shacl`**: `sh:class` and implicit-class targets now honor `rdfs:subClassOf` closure instead of exact-type matching only
- **Dependency refresh**: SciRS2 0.6.0, oxiarc-* 0.3.5, oxicrypto/oxitls 0.2.0

See the [workspace CHANGELOG](../../CHANGELOG.md) for the complete list of engine-level changes shipping in this release.

## Features

- **Data Operations**: Import, export, and validate RDF data (7 formats: Turtle, N-Triples, N-Quads, RDF/XML, JSON-LD, TriG, N3)
- **Query Execution**: Run SPARQL queries and updates against local datasets or remote endpoints
- **Server Management**: Start and configure the embedded OxiRS SPARQL/GraphQL server
- **Development Tools**: Query explain/optimize analysis, dataset indexing, schema generation, graph analytics
- **Benchmarking**: Performance testing and synthetic dataset generation
- **Migration Tools**: Convert between RDF formats, migrate from Jena TDB1/TDB2, Virtuoso, RDF4J, Blazegraph, and GraphDB
- **Industrial Connectivity**: Time-series (TSDB), Modbus, and CANbus/J1939 tooling with SAMM/AAS model generation
- **Configuration Management**: Generate, validate, and inspect server configuration files
- **Interactive Mode**: REPL for exploratory SPARQL data analysis

## Installation

### From Source

`oxirs` is not published to crates.io (`publish = false`, see [Overview](#overview)). Build and install it from a checkout:

```bash
git clone https://github.com/cool-japan/oxirs
cd oxirs/tools/oxirs
cargo install --path .

# Or with all optional features (full-text search, geo, PDF export)
cargo install --path . --features all-features
```

To just produce a binary without installing it onto your `PATH`:

```bash
git clone https://github.com/cool-japan/oxirs
cd oxirs
cargo build --release --bin oxirs
# binary at target/release/oxirs
```

### Shell Completion

`oxirs` exposes a `--completion` flag that prints a completion script for your shell:

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

# Export to a different format
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

# Dot-commands: .help .stats .history .save .load .export .import
#               .batch .search .format .template .session .clear .exit
oxirs> .stats
oxirs> .exit
```

## Commands

`oxirs` has ~60 top-level commands. `oxirs --help` lists all of them; `oxirs <command> --help` (or `oxirs <command> <subcommand> --help`) shows exact flags; `oxirs docs --format markdown` generates a complete CLI reference. The tour below covers the major groups with verified, copy-pasteable examples.

### Data Management

```bash
# Initialize dataset first
oxirs init mydata

# Import single file (dataset name must be alphanumeric, _, - only)
oxirs import mydata data.ttl --format turtle

# Import with named graph
oxirs import mydata data.ttl --format turtle --graph http://example.org/graph1

# Import N-Triples / RDF/XML / JSON-LD
oxirs import mydata data.nt --format ntriples
oxirs import mydata data.rdf --format rdfxml
oxirs import mydata data.jsonld --format jsonld

# Export entire dataset / a specific named graph
oxirs export mydata export.ttl --format turtle
oxirs export mydata output.jsonld --format json-ld --graph http://example.org/graph1
```

### Validation & Reasoning

```bash
# Validate RDF syntax (Jena rdfparse equivalent)
oxirs rdf-parse data.ttl --format turtle

# SHACL validation
oxirs shacl --dataset mydata --shapes shapes.ttl --format text

# ShEx validation
oxirs shex --dataset mydata --schema schema.shex --format text

# RDFS/OWL-RL inference
oxirs infer data.ttl --profile rdfs --output inferred.ttl

# Generate a SHACL/ShEx/OWL schema from existing data
oxirs schema-gen data.ttl --schema-type shacl --output schema.ttl --stats
```

### Query Operations

```bash
# Run a SPARQL query
oxirs query mydata "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Run a query from a file, with a specific output format
# (table, json, csv, tsv, xml, html, markdown)
oxirs query mydata query.sparql --file --output json

# SPARQL update
oxirs update mydata "INSERT DATA { <http://ex.org/s> <http://ex.org/p> \"o\" }"

# Advanced local query processor (Jena arq equivalent)
oxirs arq --dataset mydata --query "SELECT * WHERE { ?s ?p ?o }" --results table --explain

# Remote SPARQL query/update against an HTTP endpoint
oxirs r-sparql --service http://localhost:3030/sparql --query "SELECT * WHERE { ?s ?p ?o }"
oxirs r-update --service http://localhost:3030/update --update "CLEAR GRAPH <http://ex.org/g>"

# Parse and validate a query/update without executing it
oxirs q-parse query.sparql --file --print-ast --print-algebra
oxirs u-parse update.sparql --file --print-ast

# Query EXPLAIN/ANALYZE (PostgreSQL-style query plan analysis)
oxirs explain mydata "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
oxirs explain mydata query.sparql --file --mode analyze
oxirs explain mydata query.sparql --file --mode full --graphviz plan.dot

# Standalone query optimization analyzer (prints suggestions, no dataset needed)
oxirs optimize query.sparql --file
```

### Query Templates

```bash
# List all available query templates
oxirs template list

# Filter by category (basic, advanced, analytics, graph, federation, paths, aggregation)
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
# Paths: transitive-closure
# Federation: federated-query
# Analytics: statistics-summary
```

### Query History

```bash
# List recent queries (automatically tracked)
oxirs history list
oxirs history list --limit 20 --dataset mydata

# Show full query details / re-execute a previous query
oxirs history show 1
oxirs history replay 5 --output json

# Search history, view statistics/analytics, clear history
oxirs history search "SELECT"
oxirs history stats
oxirs history analytics --dataset mydata
oxirs history clear
```

### Server & Configuration

```bash
# Start SPARQL server (config file or dataset directory), optionally with GraphQL
oxirs serve mydata/oxirs.toml --port 3030
oxirs serve mydata --host 0.0.0.0 --port 8080 --graphql

# Generate / validate / inspect a configuration file
oxirs config init --output oxirs.toml
oxirs config validate oxirs.toml
oxirs config show oxirs.toml
```

### RDF Processing Tools (Jena-equivalent)

```bash
# Parse and serialize RDF, with validation/counting (riot equivalent)
oxirs riot data.ttl --output ntriples --out data.nt
oxirs riot data.ttl --validate --count

# Concatenate multiple files into one output
oxirs rdf-cat a.ttl b.ttl --format turtle --output combined.ttl

# Copy a dataset/file with format conversion
oxirs rdf-copy data.rdf data.ttl --source-format rdfxml --target-format turtle

# Compare two RDF datasets/files
oxirs rdf-diff old.ttl new.ttl --format text
```

### Storage (TDB) Tools

```bash
# Bulk load files directly into a TDB location
oxirs tdb-loader mydata data1.nt data2.nt --progress --stats

# Dump / query / update a TDB location directly (bypassing the dataset registry)
oxirs tdb-dump mydata --output dump.nq --format nquads
oxirs tdb-query mydata "SELECT * WHERE { ?s ?p ?o }" --results text
oxirs tdb-update mydata "INSERT DATA { <urn:s> <urn:p> <urn:o> }"

# Statistics, backup (optionally encrypted), and compaction
oxirs tdb-stats mydata --detailed
oxirs tdb-backup mydata backup/ --compress --encrypt   # prompts for a password
oxirs tdb-compact mydata --delete-old

# Point-in-time recovery: checkpoints and transaction-log based restore
oxirs pitr init mydata --auto-archive
oxirs pitr checkpoint mydata before-migration
oxirs pitr list mydata
```

### Benchmarking & Dataset Generation

```bash
# Generate a synthetic dataset for testing (top-level `generate`)
oxirs generate test-data.ttl --size small --type rdf --format turtle --seed 42

# Run a benchmark suite against a dataset (sp2bench, watdiv, ldbc, bsbm, custom)
oxirs benchmark run mydata --suite sp2bench --iterations 10 --detailed

# Generate a synthetic benchmark dataset
oxirs benchmark generate bench-data.ttl --size medium --triples 1000000

# Analyze a query log / compare two benchmark result files for regressions
oxirs benchmark analyze query.log --suggestions --patterns
oxirs benchmark compare baseline-results.json current-results.json --threshold 10.0
```

### Migration

```bash
# Convert RDF data between formats
oxirs migrate format data.rdf data.ttl --from rdfxml --to turtle

# Migrate from Apache Jena TDB1/TDB2, Virtuoso, RDF4J, Blazegraph, or GraphDB
oxirs migrate from-tdb2 /path/to/jena-tdb2-dir mydataset
oxirs migrate from-virtuoso "connection-string" mydataset --graphs all
oxirs migrate from-blazegraph http://localhost:9999/blazegraph/sparql mydataset --namespace kb
```

### Index Management

```bash
oxirs index list mydata
oxirs index rebuild mydata --index spo
oxirs index stats mydata --format json
oxirs index optimize mydata
```

### SAMM / AAS / Package Tools (Java ESMF SDK compatible)

```bash
# Validate / pretty-print a SAMM Aspect model, or generate artifacts from it
oxirs aspect validate model.ttl --detailed
oxirs aspect prettyprint model.ttl --output formatted.ttl
oxirs aspect to model.ttl rust --output generated/

# Asset Administration Shell (AAS) submodel <-> Aspect Model conversion
oxirs aas list aas-file.xml
oxirs aas to-aspect aas-file.xml --output-directory models/

# Namespace package import/export
oxirs package export urn:samm:org.example:1.0.0 --output package.zip
oxirs package import package.zip --models-root models/
```

### Industrial Connectivity

```bash
# Time-series database: query/insert/export with SPARQL temporal extensions
oxirs tsdb query mydata --series 1 --start 2026-01-01T00:00:00Z --end 2026-01-31T23:59:59Z --aggregate avg
oxirs tsdb insert mydata --series 1 --value 22.5
oxirs tsdb stats mydata --detailed
oxirs tsdb export mydata --series 1 --output data.csv --format csv

# Modbus TCP/RTU monitoring and RDF mapping
oxirs modbus monitor-tcp --address 192.168.1.100:502 --start 40001 --count 10 --interval 1000
oxirs modbus read --device 192.168.1.100:502 --address 40001 --count 5 --datatype float32
oxirs modbus to-rdf --device 192.168.1.100:502 --config modbus_map.toml --output data.ttl
oxirs modbus mock-server --port 5020

# CANbus/J1939 monitoring, DBC parsing, and SAMM generation
oxirs canbus monitor --interface can0 --dbc vehicle.dbc --j1939
oxirs canbus parse-dbc --file vehicle.dbc --detailed
oxirs canbus decode --id 0x0CF00400 --data DEADBEEF --dbc vehicle.dbc
oxirs canbus to-samm --dbc vehicle.dbc --output ./models/
```

### Utilities

```bash
oxirs iri "http://example.org/path" --validate --normalize
oxirs lang-tag "en-US" --validate --normalize
oxirs j-uuid --count 5 --format uuid
oxirs utf8 "some text" --validate
oxirs www-enc "hello world" --encoding url
oxirs www-dec "hello%20world" --decoding url
oxirs r-set results.csv --input-format csv --output-format table
```

### Productivity

```bash
# Command aliases
oxirs alias add qa "query mydata"
oxirs alias list

# Query result cache and LRU result-cache management
oxirs cache stats
oxirs result-cache stats

# Generate CLI reference documentation (markdown, html, man, text)
oxirs docs --format markdown --output CLI_REFERENCE.md

# Interactive tutorial mode
oxirs tutorial
```

### Performance, Analytics & Access Control

```bash
# Performance monitoring/profiling subsystem (monitor, profile, compare, health,
# report, optimizer, advisor, predictor)
oxirs performance monitor
oxirs performance advisor --help

# SPARQL query profiler with latency statistics
oxirs profile run --dataset mydata --query query.sparql --file --iterations 10

# RDF graph analytics (pagerank, community, betweenness, closeness, degree, paths, stats)
oxirs graph-analytics mydata --operation pagerank --top 20

# Export RDF graph visualization (dot/graphviz, mermaid, cytoscape/json)
oxirs visualize mydata graph.dot --format dot

# Streaming SPARQL results (NDJSON/CSV/TSV) for large result sets
oxirs stream query --dataset mydata --query query.sparql --file --format json --chunk-size 1000

# ReBAC relationship management (export, import, migrate, verify, stats)
oxirs rebac stats

# CI/CD integration file generation (report, docker, github, gitlab)
oxirs cicd github --output .github/workflows/ci.yml
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
- `server.port` / `server.host`: HTTP server bind address
- `server.enable_cors` / `server.enable_graphql`: CORS and GraphQL endpoint toggles
- `datasets.{name}.location`: Storage path (`.` means dataset directory itself)
- `datasets.{name}.dataset_type`: Storage backend (`tdb2` or `memory`)
- `datasets.{name}.read_only`: Prevent modifications
- Feature flags: `enable_reasoning`, `enable_validation`, `enable_text_search`, `enable_vector_search`

### Global Flags

```bash
# Use a specific configuration profile
oxirs --profile production serve mydata

# Verbose logging / quiet mode / disable color
oxirs --verbose query mydata "SELECT * WHERE { ?s ?p ?o }"
oxirs --quiet import mydata data.ttl --format turtle
oxirs --no-color query mydata "SELECT * WHERE { ?s ?p ?o }"

# Use a specific configuration file
oxirs --config custom-config.toml serve mydata
```

## Advanced Features

### Scripting and Automation

```bash
# Bash completion
eval "$(oxirs --completion bash)"

# Batch processing over many files
for f in rdf-files/*.ttl; do
  oxirs import combined "$f" --format turtle
done
```

### Integration with Other Tools

```bash
# Pipe SPARQL results (JSON output) into another tool
oxirs query mydata query.sparql --file --output json | python process.py

# Export then hand off to Jena's riot for reformatting
oxirs export mydata dump.nt --format ntriples
riot --formatted=turtle dump.nt
```

## Examples

### Data Processing Pipeline

```bash
#!/bin/bash
# data-pipeline.sh

oxirs init combined

# Import multiple local datasets into one
oxirs import combined dbpedia-sample.ttl --format turtle
oxirs import combined wikidata-sample.ttl --format turtle

# Validate against SHACL shapes
oxirs shacl --dataset combined --shapes validation-shapes.ttl

# Inspect index health and rebuild if needed
oxirs index stats combined
oxirs index rebuild combined

# Start the server
oxirs serve combined --port 3030
```

### Development Workflow

```bash
# Create and populate a dataset
oxirs init dev-kg
oxirs import dev-kg dev-data.ttl --format turtle

# Explore interactively
oxirs interactive --dataset dev-kg

# Start a development server
oxirs serve dev-kg --port 3030 --graphql

# Benchmark before shipping
oxirs benchmark run dev-kg --suite sp2bench --detailed
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
# Bulk-load large files directly into TDB storage
oxirs tdb-loader mydata large-dataset.nt --progress --stats

# Analyze a query before running it at scale
oxirs explain mydata query.sparql --file --mode full

# Stream large result sets instead of buffering them
oxirs stream query --dataset mydata --query query.sparql --file --chunk-size 50000

# Compact TDB storage after heavy write workloads
oxirs tdb-compact mydata
```

## Troubleshooting

### Common Issues

```bash
# Verbose output for diagnosing failures
oxirs --verbose import mydata problematic-data.ttl --format turtle

# Validate a query's syntax before debugging further
oxirs q-parse query.sparql --file --print-ast

# Check dataset statistics/health
oxirs tdb-stats mydata --detailed
```

### Recovery

```bash
# Point-in-time recovery from transaction logs (requires `oxirs pitr init` beforehand)
oxirs pitr list mydata
oxirs pitr recover-timestamp mydata 2026-07-01T00:00:00Z restored/

# Restore from a tdb-backup archive by pointing tdb-loader at the extracted backup
oxirs tdb-loader restored-mydata /path/to/extracted-backup/*.nq
```

## Related Tools

- [`oxirs-fuseki`](../../server/oxirs-fuseki/): SPARQL HTTP server
- [`oxirs-chat`](../../ai/oxirs-chat/): AI-powered chat interface
- [`oxirs-tauri`](../../desktop/oxirs-tauri/): Desktop GUI (chat, visual SPARQL builder, CAN bus monitor)
- Apache Jena: Java-based semantic web toolkit
- RDFLib: Python RDF processing library

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new commands
4. Update documentation
5. Submit a pull request

## License

Licensed under:

- Apache License, Version 2.0 ([LICENSE](../../LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)

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
oxirs migrate format data.ttl data.nt --from turtle --to ntriples

# Validation
oxirs rdf-parse file.ttl -f turtle             # Validate syntax
oxirs shacl --dataset mydata --shapes shapes.ttl --format text

# Analysis
oxirs explain mydata query.sparql --file       # Query analysis
oxirs tdb-stats mydata --detailed              # Dataset statistics
```

### Performance Tips

**For Large Datasets (>1M triples)**:
- Use the TDB bulk loader: `oxirs tdb-loader mydata *.nt --progress --stats`
- Compact storage after large write batches: `oxirs tdb-compact mydata`
- Stream large query results instead of buffering: `oxirs stream query --dataset mydata --query q.sparql --file`

**For Query Performance**:
- Analyze queries before execution: `oxirs explain mydata query.sparql --file --mode full`
- Use appropriate output format (JSON for programmatic use, table for humans)
- Check `oxirs performance advisor` and `oxirs profile run` for optimization suggestions

### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Dataset not found" | Run `oxirs init <name>` first to create the dataset |
| "Format not recognized" | Specify format explicitly with `--format` flag |
| "Permission denied" | Check directory permissions with `chmod 755 <dir>` |
| "Port already in use" | Use a different port with `--port <num>` |
| "Dataset directory already exists" | `oxirs init` refuses to overwrite; choose a new name or remove the old directory |
| "Invalid SPARQL syntax" | Use `oxirs q-parse` to validate query syntax before running it |

**Debug Mode**:
```bash
# Enable verbose logging
oxirs --verbose query mydata "SELECT * WHERE {?s ?p ?o}"

# Debug specific modules via tracing-subscriber's env filter
RUST_LOG=oxirs_core=debug,oxirs_arq=trace oxirs query mydata query.sparql
```

---

**OxiRS CLI v0.3.2** - Production-ready command-line interface for semantic web operations
