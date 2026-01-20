# OxiRS CLI Command Reference Manual

**Version**: 0.1.0
**Last Updated**: January 7, 2026
**Status**: Production-Ready

## Table of Contents

- [Core Commands](#core-commands)
- [Data Processing](#data-processing)
- [Query Tools](#query-tools)
- [Storage Management](#storage-management)
- [Validation & Reasoning](#validation--reasoning)
- [SAMM & Industry 4.0](#samm--industry-40)
- [Utility Commands](#utility-commands)
- [Advanced Features](#advanced-features)
- [Developer Tools](#developer-tools)

---

## Core Commands

### `oxirs init`

Initialize a new RDF dataset.

**Usage:**
```bash
oxirs init <NAME> [OPTIONS]
```

**Arguments:**
- `<NAME>` - Dataset name (alphanumeric, underscore, hyphen only; no dots)

**Options:**
- `--format <FORMAT>` - Dataset format: `tdb2` (default), `memory`
- `--location <PATH>` - Custom dataset location (default: `./data/<NAME>`)

**Examples:**
```bash
# Create a persistent TDB2 dataset
oxirs init mydata

# Create an in-memory dataset
oxirs init tempdata --format memory

# Create dataset at custom location
oxirs init mydata --location /var/lib/oxirs/data
```

**Notes:**
- Dataset names must be alphanumeric with underscores/hyphens only
- TDB2 datasets are persisted to disk automatically
- Memory datasets are lost when the process ends

---

### `oxirs query`

Execute SPARQL SELECT/ASK/CONSTRUCT/DESCRIBE queries.

**Usage:**
```bash
oxirs query <DATASET> <QUERY> [OPTIONS]
```

**Arguments:**
- `<DATASET>` - Dataset name or path
- `<QUERY>` - SPARQL query string or file path

**Options:**
- `-f, --file` - Treat query as file path
- `-o, --output <FORMAT>` - Output format: `json`, `csv`, `tsv`, `table` (default), `xml`, `html`, `markdown`, `md`

**Examples:**
```bash
# Inline query with table output
oxirs query mydata "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# Query from file with JSON output
oxirs query mydata query.sparql --file --output json

# Complex query with CSV export
oxirs query mydata "
  PREFIX foaf: <http://xmlns.com/foaf/0.1/>
  SELECT ?name ?email WHERE {
    ?person foaf:name ?name .
    ?person foaf:mbox ?email .
  }
" --output csv > contacts.csv

# HTML report generation
oxirs query mydata report.sparql --file --output html > report.html
```

**Output Formats:**
- `table` - ASCII table (human-readable, default)
- `json` - JSON with bindings
- `csv` - Comma-separated values
- `tsv` - Tab-separated values
- `xml` - SPARQL XML Results Format
- `html` - HTML table
- `markdown`/`md` - Markdown table

**Notes:**
- Supports SPARQL 1.1 and SPARQL 1.2 features
- Query complexity is automatically estimated
- Large result sets are streamed efficiently

---

### `oxirs import`

Import RDF data into a dataset.

**Usage:**
```bash
oxirs import <DATASET> <FILE> [OPTIONS]
```

**Arguments:**
- `<DATASET>` - Target dataset name
- `<FILE>` - Input RDF file path

**Options:**
- `-f, --format <FORMAT>` - RDF format: `turtle`, `ntriples`, `nquads`, `trig`, `rdfxml`, `jsonld`, `n3`
- `-g, --graph <URI>` - Named graph URI (default: default graph)

**Examples:**
```bash
# Import Turtle file (format auto-detected)
oxirs import mydata schema.ttl

# Import with explicit format
oxirs import mydata data.nt --format ntriples

# Import to named graph
oxirs import mydata data.ttl --graph http://example.org/graph1

# Import N-Quads (includes graph information)
oxirs import mydata dataset.nq --format nquads
```

**Supported Formats:**
- `turtle` (.ttl) - Terse RDF Triple Language
- `ntriples` (.nt) - N-Triples
- `nquads` (.nq) - N-Quads (with named graphs)
- `trig` (.trig) - TriG (Turtle with named graphs)
- `rdfxml` (.rdf, .xml) - RDF/XML
- `jsonld` (.jsonld) - JSON-LD
- `n3` (.n3) - Notation3

**Notes:**
- Format auto-detection based on file extension
- Progress indicators for large files
- Validation before import
- Atomic import (all-or-nothing)

---

### `oxirs export`

Export RDF data from a dataset.

**Usage:**
```bash
oxirs export <DATASET> <FILE> [OPTIONS]
```

**Arguments:**
- `<DATASET>` - Source dataset name
- `<FILE>` - Output file path

**Options:**
- `-f, --format <FORMAT>` - Output format (same as import)
- `-g, --graph <URI>` - Export specific named graph only

**Examples:**
```bash
# Export entire dataset as N-Quads
oxirs export mydata backup.nq --format nquads

# Export to Turtle
oxirs export mydata data.ttl --format turtle

# Export specific graph
oxirs export mydata graph1.nt --graph http://example.org/graph1 --format ntriples

# Compress output
oxirs export mydata data.nq.gz --format nquads
gzip data.nq.gz
```

**Notes:**
- Streaming export for large datasets
- Named graphs preserved in N-Quads/TriG formats
- Progress tracking for large exports

---

### `oxirs update`

Execute SPARQL UPDATE operations.

**Usage:**
```bash
oxirs update <DATASET> <UPDATE> [OPTIONS]
```

**Arguments:**
- `<DATASET>` - Target dataset name
- `<UPDATE>` - SPARQL UPDATE string or file path

**Options:**
- `-f, --file` - Treat update as file path

**Examples:**
```bash
# Insert data
oxirs update mydata "
  PREFIX ex: <http://example.org/>
  INSERT DATA {
    ex:alice ex:knows ex:bob .
    ex:bob ex:knows ex:charlie .
  }
"

# Delete and insert
oxirs update mydata "
  PREFIX ex: <http://example.org/>
  DELETE { ?s ex:status ?old }
  INSERT { ?s ex:status 'active' }
  WHERE { ?s ex:status ?old }
" --file

# Bulk update from file
oxirs update mydata updates.sparql --file
```

**Supported Operations:**
- `INSERT DATA` - Add triples
- `DELETE DATA` - Remove triples
- `DELETE/INSERT WHERE` - Conditional updates
- `LOAD` - Load from URL
- `CLEAR` - Clear graph
- `CREATE` - Create graph
- `DROP` - Drop graph

---

### `oxirs serve`

Start HTTP SPARQL server.

**Usage:**
```bash
oxirs serve --config <CONFIG> [OPTIONS]
```

**Options:**
- `--config <FILE>` - Configuration file path (required)
- `--port <PORT>` - Override port (default: 3030)
- `--host <HOST>` - Override host (default: localhost)
- `--graphql` - Enable GraphQL endpoint

**Examples:**
```bash
# Start server with config
oxirs serve --config oxirs.toml

# Custom port
oxirs serve --config oxirs.toml --port 8080

# Enable GraphQL
oxirs serve --config oxirs.toml --graphql

# Bind to all interfaces
oxirs serve --config oxirs.toml --host 0.0.0.0
```

**Endpoints:**
- `GET/POST /sparql` - SPARQL Query endpoint
- `POST /update` - SPARQL Update endpoint
- `GET/POST /graphql` - GraphQL endpoint (if enabled)
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe
- `GET /metrics` - Prometheus metrics

**Notes:**
- Supports CORS configuration
- Authentication via JWT/OAuth2/Basic
- Rate limiting and DDoS protection
- TLS/SSL support

---

## Data Processing

### `oxirs riot`

RDF parsing and serialization (Jena riot equivalent).

**Usage:**
```bash
oxirs riot <INPUT...> [OPTIONS]
```

**Arguments:**
- `<INPUT>` - Input file(s)

**Options:**
- `--output <FORMAT>` - Output format: `turtle`, `ntriples`, `rdfxml`, `jsonld`, `trig`, `nquads` (default: turtle)
- `--out <FILE>` - Output file (default: stdout)
- `--syntax <FORMAT>` - Input format (auto-detect if not specified)
- `--base <URI>` - Base URI for relative URIs
- `--validate` - Validate syntax only (no output)
- `--count` - Count triples/quads only

**Examples:**
```bash
# Convert Turtle to N-Triples
oxirs riot data.ttl --output ntriples

# Validate RDF/XML
oxirs riot schema.rdf --validate

# Count triples
oxirs riot data.ttl --count

# Merge multiple files
oxirs riot file1.ttl file2.ttl --output nquads --out merged.nq

# Convert with base URI
oxirs riot data.ttl --base http://example.org/ --output turtle
```

---

### `oxirs batch`

Parallel bulk import.

**Usage:**
```bash
oxirs batch import --dataset <DATASET> --files <FILE...> [OPTIONS]
```

**Options:**
- `--dataset <NAME>` - Target dataset
- `--files <FILE...>` - Input files (glob patterns supported)
- `--format <FORMAT>` - Format (auto-detect if not specified)
- `--graph <URI>` - Target named graph
- `--parallel <N>` - Number of parallel workers (default: 4)

**Examples:**
```bash
# Import all Turtle files
oxirs batch import --dataset mydata --files *.ttl

# Parallel import with 8 workers
oxirs batch import --dataset mydata --files data/*.nt --parallel 8

# Import to specific graph
oxirs batch import --dataset mydata --files *.ttl --graph http://example.org/bulk
```

**Notes:**
- Work-stealing scheduler for optimal performance
- Per-file error handling
- Global statistics aggregation
- Thread-safe concurrent store access

---

### `oxirs migrate`

Convert between RDF formats.

**Usage:**
```bash
oxirs migrate --source <FILE> --target <FILE> --from <FORMAT> --to <FORMAT>
```

**Options:**
- `--source <FILE>` - Source file
- `--target <FILE>` - Target file
- `--from <FORMAT>` - Source format
- `--to <FORMAT>` - Target format

**Examples:**
```bash
# Turtle to N-Triples
oxirs migrate --source data.ttl --target data.nt --from turtle --to ntriples

# N-Quads to TriG
oxirs migrate --source data.nq --target data.trig --from nquads --to trig

# RDF/XML to JSON-LD
oxirs migrate --source schema.rdf --target schema.jsonld --from rdfxml --to jsonld
```

**Features:**
- Streaming architecture (no intermediate storage)
- Memory-efficient for large files
- Progress tracking
- Error statistics

---

## Query Tools

### `oxirs arq`

Advanced SPARQL query processor (Jena arq equivalent).

**Usage:**
```bash
oxirs arq [OPTIONS]
```

**Options:**
- `--query <STRING>` - SPARQL query string
- `--query-file <FILE>` - Query from file
- `--data <FILE>` - Data file(s) (repeatable)
- `--namedgraph <URI>` - Named graph data (repeatable)
- `--results <FORMAT>` - Results format: `table`, `csv`, `tsv`, `json`, `xml` (default: table)
- `--dataset <PATH>` - TDB dataset location
- `--explain` - Show query execution plan
- `--optimize` - Enable query optimization
- `--time` - Show execution time

**Examples:**
```bash
# Query data files directly
oxirs arq --query "SELECT * WHERE { ?s ?p ?o }" --data file1.ttl --data file2.ttl

# Query with named graphs
oxirs arq --query-file query.sparql --data data.ttl --namedgraph http://example.org/g1

# Query optimization and explain
oxirs arq --query-file complex.sparql --dataset ./data/mydata --explain --optimize --time
```

---

### `oxirs explain`

Query execution plan analysis.

**Usage:**
```bash
oxirs explain <DATASET> <QUERY> [OPTIONS]
```

**Arguments:**
- `<DATASET>` - Dataset name
- `<QUERY>` - SPARQL query

**Options:**
- `-f, --file` - Query from file
- `-m, --mode <MODE>` - Analysis mode: `explain`, `analyze`, `full` (default: explain)

**Examples:**
```bash
# Basic explain
oxirs explain mydata "SELECT * WHERE { ?s ?p ?o }"

# Full analysis with statistics
oxirs explain mydata query.sparql --file --mode full

# Analyze mode (with execution)
oxirs explain mydata complex.sparql --file --mode analyze
```

**Modes:**
- `explain` - Show query plan only
- `analyze` - Execute and show runtime statistics
- `full` - Detailed analysis with optimization hints

**Output:**
- PostgreSQL-style EXPLAIN format
- Estimated cardinalities
- Join order and algorithms
- Index usage
- Optimization suggestions

---

### `oxirs template`

SPARQL query template management.

**Usage:**
```bash
oxirs template <SUBCOMMAND>
```

**Subcommands:**
- `list [--category <CAT>]` - List available templates
- `show <NAME>` - Show template details
- `render <NAME> --param <K=V>...` - Render with parameters

**Examples:**
```bash
# List all templates
oxirs template list

# List by category
oxirs template list --category analytics

# Show template
oxirs template show basic-select

# Render with parameters
oxirs template render basic-select --param limit=100 --param offset=0
```

**Built-in Templates:**
- `basic-select` - Simple SELECT query
- `construct-graph` - CONSTRUCT query template
- `aggregate-count` - COUNT aggregation
- `property-path` - Property path queries
- `federated-query` - Federation template
- `analytics-stats` - Statistical analytics
- And more...

---

### `oxirs history`

Query history management.

**Usage:**
```bash
oxirs history <SUBCOMMAND>
```

**Subcommands:**
- `list [--limit <N>] [--dataset <NAME>]` - List history
- `show <ID>` - Show query details
- `replay <ID> [--output <FORMAT>]` - Replay query
- `search <TEXT>` - Search history
- `clear` - Clear history
- `stats` - Show statistics

**Examples:**
```bash
# List recent queries
oxirs history list --limit 20

# Show specific query
oxirs history show 42

# Replay query with different output
oxirs history replay 42 --output json

# Search history
oxirs history search "SELECT.*foaf"

# Statistics
oxirs history stats
```

**Storage:**
- Location: `~/.local/share/oxirs/query_history.json`
- Persisted across sessions
- Includes execution time and results

---

## Storage Management

### `oxirs tdbloader`

Bulk data loading into TDB.

**Usage:**
```bash
oxirs tdbloader <LOCATION> <FILE...> [OPTIONS]
```

**Arguments:**
- `<LOCATION>` - TDB dataset location
- `<FILE>` - Input files

**Options:**
- `-g, --graph <URI>` - Target graph
- `--progress` - Show progress
- `--stats` - Show statistics

**Examples:**
```bash
# Load data with progress
oxirs tdbloader ./data/mydata file1.nt file2.nt --progress

# Load to named graph
oxirs tdbloader ./data/mydata *.ttl --graph http://example.org/data --stats
```

---

### `oxirs tdbdump`

Export TDB dataset.

**Usage:**
```bash
oxirs tdbdump <LOCATION> [OPTIONS]
```

**Arguments:**
- `<LOCATION>` - TDB dataset location

**Options:**
- `-o, --output <FILE>` - Output file (default: stdout)
- `-f, --format <FORMAT>` - Output format: `nquads`, `turtle`, `ntriples`, etc. (default: nquads)
- `-g, --graph <URI>` - Dump specific graph only

**Examples:**
```bash
# Dump to N-Quads
oxirs tdbdump ./data/mydata --output backup.nq

# Dump specific graph
oxirs tdbdump ./data/mydata --graph http://example.org/g1 --format turtle
```

---

### `oxirs tdbstats`

Show dataset statistics.

**Usage:**
```bash
oxirs tdbstats <LOCATION> [OPTIONS]
```

**Options:**
- `--detailed` - Show detailed statistics
- `--format <FORMAT>` - Output format: `text`, `json` (default: text)

**Examples:**
```bash
# Basic stats
oxirs tdbstats ./data/mydata

# Detailed JSON
oxirs tdbstats ./data/mydata --detailed --format json
```

---

### `oxirs tdbbackup`

Backup TDB dataset.

**Usage:**
```bash
oxirs tdbbackup <SOURCE> <TARGET> [OPTIONS]
```

**Options:**
- `--compress` - Compress backup
- `--incremental` - Incremental backup

**Examples:**
```bash
# Full backup
oxirs tdbbackup ./data/mydata ./backups/mydata-20260104

# Compressed incremental backup
oxirs tdbbackup ./data/mydata ./backups/mydata-inc --compress --incremental
```

---

### `oxirs tdbcompact`

Compact TDB dataset (reclaim space).

**Usage:**
```bash
oxirs tdbcompact <LOCATION> [OPTIONS]
```

**Options:**
- `--delete-old` - Delete old log files after compaction

**Examples:**
```bash
# Compact dataset
oxirs tdbcompact ./data/mydata

# Compact and cleanup
oxirs tdbcompact ./data/mydata --delete-old
```

---

## Validation & Reasoning

### `oxirs shacl`

SHACL validation.

**Usage:**
```bash
oxirs shacl --shapes <FILE> [OPTIONS]
```

**Options:**
- `--data <FILE>` - Data file to validate
- `--dataset <PATH>` - Dataset to validate
- `--shapes <FILE>` - SHACL shapes file (required)
- `--format <FORMAT>` - Output format: `text`, `turtle`, `json`, `xml` (default: text)
- `-o, --output <FILE>` - Output file

**Examples:**
```bash
# Validate file against shapes
oxirs shacl --data data.ttl --shapes shapes.ttl

# Validate dataset
oxirs shacl --dataset ./data/mydata --shapes shapes.ttl --format json

# Save validation report
oxirs shacl --data data.ttl --shapes shapes.ttl --output report.ttl
```

---

### `oxirs infer`

Inference and reasoning.

**Usage:**
```bash
oxirs infer <DATA> [OPTIONS]
```

**Arguments:**
- `<DATA>` - Input data file

**Options:**
- `--ontology <FILE>` - Ontology/schema file
- `--profile <PROFILE>` - Reasoning profile: `rdfs`, `owl-rl`, `custom` (default: rdfs)
- `-o, --output <FILE>` - Output file
- `--format <FORMAT>` - Output format (default: turtle)

**Examples:**
```bash
# RDFS inference
oxirs infer data.ttl --profile rdfs --output inferred.ttl

# OWL-RL reasoning with ontology
oxirs infer data.ttl --ontology ontology.ttl --profile owl-rl
```

---

## SAMM & Industry 4.0

### `oxirs aspect`

SAMM Aspect Model tools (ESMF SDK compatible).

**Usage:**
```bash
oxirs aspect <SUBCOMMAND>
```

**Subcommands:**
- `validate <FILE>` - Validate Aspect Model
- `to-openapi <FILE>` - Generate OpenAPI spec
- `to-schema <FILE>` - Generate JSON Schema
- `to-html <FILE>` - Generate HTML documentation
- `to-png <FILE>` - Generate PNG diagram
- `to-svg <FILE>` - Generate SVG diagram
- `to-sql <FILE>` - Generate SQL schema
- `to-aas <FILE>` - Convert to AAS
- `prettyprint <FILE>` - Format Aspect Model
- `migrate <FILE> --target <VERSION>` - Migrate to SAMM version

**Examples:**
```bash
# Validate Aspect Model
oxirs aspect validate Movement.ttl

# Generate OpenAPI
oxirs aspect to-openapi Movement.ttl --output movement-api.yaml

# Generate documentation
oxirs aspect to-html Movement.ttl --output movement.html

# Migrate to SAMM 2.1.0
oxirs aspect migrate Movement.ttl --target 2.1.0
```

---

### `oxirs aas`

Asset Administration Shell tools.

**Usage:**
```bash
oxirs aas <SUBCOMMAND>
```

**Subcommands:**
- `validate <FILE>` - Validate AAS
- `to-rdf <FILE>` - Convert to RDF
- `to-json <FILE>` - Convert to JSON
- `to-xml <FILE>` - Convert to XML

**Examples:**
```bash
# Validate AAS
oxirs aas validate asset.json

# Convert to RDF
oxirs aas to-rdf asset.json --output asset.ttl

# Convert to XML
oxirs aas to-xml asset.json --output asset.xml
```

---

## Utility Commands

### `oxirs config`

Configuration management.

**Usage:**
```bash
oxirs config <SUBCOMMAND>
```

**Subcommands:**
- `init [--output <FILE>]` - Generate default config
- `validate <FILE>` - Validate config file
- `show [<FILE>]` - Show current config

**Examples:**
```bash
# Generate default config
oxirs config init --output oxirs.toml

# Validate config
oxirs config validate oxirs.toml

# Show config
oxirs config show oxirs.toml
```

---

### `oxirs interactive`

Start interactive REPL mode.

**Usage:**
```bash
oxirs interactive [OPTIONS]
```

**Options:**
- `-d, --dataset <NAME>` - Initial dataset
- `--history <FILE>` - History file location

**Examples:**
```bash
# Start REPL
oxirs interactive

# Start with dataset
oxirs interactive --dataset mydata
```

**Features:**
- Command history with search
- Tab completion
- Multi-line queries
- Syntax highlighting
- Query templates
- Result pagination

---

### `oxirs benchmark`

Performance benchmarking.

**Usage:**
```bash
oxirs benchmark <DATASET> [OPTIONS]
```

**Options:**
- `--suite <NAME>` - Benchmark suite
- `--iterations <N>` - Number of iterations (default: 10)
- `-o, --output <FILE>` - Output file (JSON)

**Examples:**
```bash
# Run standard benchmark
oxirs benchmark mydata --iterations 100

# Custom suite
oxirs benchmark mydata --suite custom-queries --output results.json
```

---

## Advanced Features

### `oxirs performance`

Performance monitoring and profiling.

**Usage:**
```bash
oxirs performance <SUBCOMMAND>
```

**Subcommands:**
- `profile <QUERY>` - Profile query execution
- `benchmark <QUERY>` - Benchmark query
- `monitor --dataset <NAME>` - Monitor dataset operations

**Examples:**
```bash
# Profile query
oxirs performance profile "SELECT * WHERE { ?s ?p ?o }" --dataset mydata

# Monitor operations
oxirs performance monitor --dataset mydata
```

---

### `oxirs cache`

Query result caching.

**Usage:**
```bash
oxirs cache <SUBCOMMAND>
```

**Subcommands:**
- `clear` - Clear cache
- `stats` - Show cache statistics
- `warmup <QUERY...>` - Warmup cache

**Examples:**
```bash
# Clear cache
oxirs cache clear

# Show stats
oxirs cache stats

# Warmup common queries
oxirs cache warmup query1.sparql query2.sparql
```

---

## Developer Tools

### `oxirs docs`

Generate comprehensive CLI documentation.

**Usage:**
```bash
oxirs docs [OPTIONS]
```

**Options:**
- `-f, --format <FORMAT>` - Output format: `markdown` (default), `html`, `man`, `text`
- `-o, --output <FILE>` - Output file path (stdout if not specified)
- `--command <COMMAND>` - Generate documentation for specific command

**Examples:**
```bash
# Generate Markdown documentation to file
oxirs docs --format markdown --output CLI_REFERENCE.md

# Generate HTML documentation
oxirs docs --format html --output docs.html

# Generate man page
oxirs docs --format man --output oxirs.1

# Output to stdout
oxirs docs --format markdown
```

**Features:**
- Auto-discovers all commands and subcommands
- Includes arguments, options, examples for each command
- Multiple output formats for different use cases
- Command-specific documentation generation

---

### `oxirs tutorial`

Interactive tutorial mode for learning OxiRS.

**Usage:**
```bash
oxirs tutorial [OPTIONS]
```

**Options:**
- `-l, --lesson <LESSON>` - Start at specific lesson

**Examples:**
```bash
# Start interactive tutorial
oxirs tutorial

# Start with specific lesson (future enhancement)
oxirs tutorial --lesson "Basic SPARQL"
```

**Tutorial Topics:**
1. **Getting Started** - OxiRS basics, initialization, configuration
2. **Basic SPARQL** - First queries, data import, SELECT statements
3. **SPARQL Filters** - Advanced filtering techniques
4. **Output Formats** - Working with JSON, CSV, PDF exports

**Features:**
- Interactive menu navigation
- Step-by-step instructions with examples
- Hint system for each tutorial step
- Progress tracking and completion status
- Color-coded UI with emoji indicators
- Difficulty levels: Beginner, Intermediate, Advanced

---

## Global Options

All commands support these global options:

- `-v, --verbose` - Enable verbose logging
- `-q, --quiet` - Suppress output (quiet mode)
- `--no-color` - Disable colored output
- `-c, --config <FILE>` - Configuration file
- `--completion <SHELL>` - Generate shell completion (bash, zsh, fish, powershell)

**Examples:**
```bash
# Verbose mode
oxirs query mydata "SELECT * WHERE { ?s ?p ?o }" --verbose

# Generate bash completion
oxirs --completion bash > /etc/bash_completion.d/oxirs

# Use custom config
oxirs serve --config /etc/oxirs/production.toml
```

---

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Invalid arguments
- `3` - Dataset not found
- `4` - Query syntax error
- `5` - I/O error
- `6` - Permission denied

---

## Environment Variables

- `OXIRS_LOG_LEVEL` - Log level: `trace`, `debug`, `info`, `warn`, `error` (default: info)
- `OXIRS_LOG_FORMAT` - Log format: `text`, `json`, `pretty` (default: text)
- `OXIRS_LOG_FILE` - Log file path
- `OXIRS_PERF_THRESHOLD` - Performance warning threshold (ms)
- `NO_COLOR` - Disable colored output (any value)

---

## See Also

- [README.md](../README.md) - Overview and getting started
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference
- [INTERACTIVE.md](INTERACTIVE.md) - Interactive mode guide
- [BEST_PRACTICES.md](BEST_PRACTICES.md) - Best practices guide

---

**OxiRS CLI v0.1.0** - Production-ready command-line interface for semantic web operations with 452 tests passing (100%)
