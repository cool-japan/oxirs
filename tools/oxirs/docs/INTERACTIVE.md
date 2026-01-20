# OxiRS Interactive Mode Guide

**Version**: 0.1.0
**Last Updated**: January 7, 2026
**Status**: Production-Ready

## Overview

OxiRS Interactive Mode provides a powerful REPL (Read-Eval-Print-Loop) for working with RDF datasets and executing SPARQL queries interactively. It's designed for exploratory data analysis, debugging, and rapid prototyping.

## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
- [Features](#features)
- [Commands](#commands)
- [Query Execution](#query-execution)
- [History Management](#history-management)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Configuration](#configuration)
- [Tips & Tricks](#tips--tricks)

---

## Getting Started

### Starting Interactive Mode

```bash
# Start REPL
oxirs interactive

# Start with a dataset
oxirs interactive --dataset mydata

# Custom history file
oxirs interactive --history ~/.oxirs_history
```

### First Steps

```
OxiRS Interactive Mode v0.1.0
Type 'help' for commands, 'exit' to quit

oxirs> connect mydata
Connected to dataset: mydata (TDB2, 125,432 triples)

oxirs[mydata]> SELECT * WHERE { ?s ?p ?o } LIMIT 5
Executing query...
+-------------------+-------------------+-------------------+
| s                 | p                 | o                 |
+-------------------+-------------------+-------------------+
| ex:Alice          | rdf:type          | foaf:Person       |
| ex:Alice          | foaf:name         | "Alice"           |
| ex:Alice          | foaf:knows        | ex:Bob            |
| ex:Bob            | rdf:type          | foaf:Person       |
| ex:Bob            | foaf:name         | "Bob"             |
+-------------------+-------------------+-------------------+
5 results (23ms)

oxirs[mydata]>
```

---

## Basic Usage

### Connecting to Datasets

```
# Connect to dataset
oxirs> connect mydata

# Connect to dataset at custom location
oxirs> connect /path/to/dataset

# Show current connection
oxirs> status
Connected to: mydata
Location: ./data/mydata
Type: TDB2
Triples: 125,432
Graphs: 3

# Disconnect
oxirs> disconnect
```

### Executing Queries

```
# Simple SELECT query
oxirs[mydata]> SELECT * WHERE { ?s ?p ?o } LIMIT 10

# Multi-line query (press Enter for new line, Ctrl+D to execute)
oxirs[mydata]> SELECT ?name ?email WHERE {
...> ?person foaf:name ?name .
...> ?person foaf:mbox ?email .
...> }

# ASK query
oxirs[mydata]> ASK { ?s rdf:type foaf:Person }
true

# CONSTRUCT query
oxirs[mydata]> CONSTRUCT { ?s foaf:knows ?o } WHERE { ?s foaf:knows ?o }
```

### Update Operations

```
# Insert data
oxirs[mydata]> INSERT DATA {
...>   <http://example.org/alice> foaf:knows <http://example.org/charlie> .
...> }
Update successful (1 triple inserted)

# Delete data
oxirs[mydata]> DELETE DATA {
...>   <http://example.org/alice> foaf:knows <http://example.org/charlie> .
...> }
Update successful (1 triple deleted)
```

---

## Features

### 1. Tab Completion

Tab completion is available for:

- Commands: `connect`, `disconnect`, `help`, etc.
- Dataset names
- SPARQL keywords: `SELECT`, `WHERE`, `PREFIX`, etc.
- Common prefixes: `rdf:`, `rdfs:`, `foaf:`, `ex:`, etc.

**Usage:**
```
oxirs> conn<TAB>
connect

oxirs> SELE<TAB>
SELECT

oxirs> PREFIX foaf: <http://xmlns.com/foaf/0.1/>
oxirs> ?person foaf:na<TAB>
?person foaf:name
```

### 2. Command History

All queries and commands are saved to history.

```
# Navigate history
Up Arrow    - Previous command
Down Arrow  - Next command
Ctrl+R      - Search history

# History management
oxirs> history           # Show last 20 commands
oxirs> history 50        # Show last 50 commands
oxirs> history clear     # Clear history
oxirs> !42              # Replay command #42
```

### 3. Multi-line Editing

Supports multi-line queries with proper indentation:

```
oxirs> PREFIX foaf: <http://xmlns.com/foaf/0.1/>
...> SELECT ?name WHERE {
...>   ?person foaf:name ?name .
...>   FILTER(LANG(?name) = "en")
...> }
```

**Controls:**
- `Enter` - New line
- `Ctrl+D` or `;;` - Execute query
- `Ctrl+C` - Cancel current input

### 4. Syntax Highlighting

SPARQL keywords, URIs, literals, and variables are color-coded:

- **Keywords**: Blue (SELECT, WHERE, FILTER, etc.)
- **URIs**: Green
- **Literals**: Yellow
- **Variables**: Cyan
- **Comments**: Gray

### 5. Query Templates

Pre-defined templates for common queries:

```
# List templates
oxirs> templates

Available templates:
  1. basic-select      - Simple SELECT query
  2. construct-graph   - CONSTRUCT query
  3. describe-resource - DESCRIBE query
  4. count-triples     - Count triples
  5. list-classes      - List all classes
  6. list-properties   - List all properties

# Use template
oxirs> template basic-select
SELECT * WHERE { ?s ?p ?o } LIMIT 10

# Use template with parameters
oxirs> template basic-select --limit 100
SELECT * WHERE { ?s ?p ?o } LIMIT 100
```

### 6. Result Formatting

Multiple output formats for query results:

```
# Change output format
oxirs[mydata]> set format table    # ASCII table (default)
oxirs[mydata]> set format json     # JSON
oxirs[mydata]> set format csv      # CSV
oxirs[mydata]> set format markdown # Markdown table

# Limit results
oxirs[mydata]> set limit 20       # Show max 20 results
oxirs[mydata]> set limit none     # Show all results

# Show/hide query time
oxirs[mydata]> set timing on
oxirs[mydata]> set timing off
```

### 7. Result Pagination

For large result sets:

```
oxirs[mydata]> SELECT * WHERE { ?s ?p ?o }
Showing results 1-100 of 15,432

[Results displayed]

Press 'n' for next, 'p' for previous, 'q' to quit
```

### 8. Explain and Profiling

```
# Explain query plan
oxirs[mydata]> explain SELECT * WHERE { ?s ?p ?o }
Query Plan:
  1. BGP (Basic Graph Pattern)
     - Estimated cardinality: 125,432
     - Access: Sequential scan
     - Cost: 125

# Profile query
oxirs[mydata]> profile SELECT ?name WHERE { ?person foaf:name ?name }
Execution Time: 45ms
Triples Scanned: 1,234
Results: 567
Memory: 2.3 MB
```

---

## Commands

### Connection Commands

```
connect <dataset>         Connect to dataset
disconnect                Disconnect from current dataset
status                    Show connection status
datasets                  List available datasets
```

### Query Commands

```
SELECT ...                Execute SELECT query
ASK ...                   Execute ASK query
CONSTRUCT ...             Execute CONSTRUCT query
DESCRIBE ...              Execute DESCRIBE query
INSERT ...                Execute INSERT update
DELETE ...                Execute DELETE update
```

### Utility Commands

```
help                      Show help
history [n]               Show command history
!<n>                      Replay command #n
clear                     Clear screen
set <key> <value>         Set configuration option
show <key>                Show configuration value
templates                 List query templates
template <name> [args]    Use query template
explain <query>           Explain query plan
profile <query>           Profile query execution
```

### Control Commands

```
exit / quit / Ctrl+D      Exit interactive mode
Ctrl+C                    Cancel current input
Ctrl+L                    Clear screen
```

---

## Query Execution

### PREFIX Declarations

Prefixes are remembered across the session:

```
oxirs> PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

oxirs> PREFIX ex: <http://example.org/>
PREFIX ex: <http://example.org/>

# Use prefixes in queries
oxirs> SELECT * WHERE { ?s foaf:name ?name }

# Show active prefixes
oxirs> show prefixes
Registered prefixes:
  rdf:   http://www.w3.org/1999/02/22-rdf-syntax-ns#
  rdfs:  http://www.w3.org/2000/01/rdf-schema#
  foaf:  http://xmlns.com/foaf/0.1/
  ex:    http://example.org/
```

### Query Execution Modes

```
# Normal execution
oxirs[mydata]> SELECT * WHERE { ?s ?p ?o } LIMIT 10

# Dry run (parse only, don't execute)
oxirs[mydata]> dryrun SELECT * WHERE { ?s ?p ?o }
Query syntax: OK
Estimated cost: 125

# Benchmark query
oxirs[mydata]> benchmark SELECT ?name WHERE { ?person foaf:name ?name }
Running 10 iterations...
Average time: 42ms
Min: 38ms, Max: 51ms, StdDev: 3.2ms
```

---

## History Management

### View History

```
# Last 20 commands
oxirs> history

# Last 50 commands
oxirs> history 50

# Search history
oxirs> history search SELECT

# Show history with timestamps
oxirs> history --timestamps
```

### Replay Commands

```
# Replay specific command
oxirs> !42

# Replay last command
oxirs> !!

# Replay last SELECT query
oxirs> !SELECT
```

### Export History

```
# Export to file
oxirs> history export queries.txt

# Export as SPARQL queries only
oxirs> history export --queries-only queries.sparql
```

---

## Keyboard Shortcuts

### Navigation

- `Ctrl+A` - Move to beginning of line
- `Ctrl+E` - Move to end of line
- `Ctrl+B` / `Left Arrow` - Move back one character
- `Ctrl+F` / `Right Arrow` - Move forward one character
- `Alt+B` - Move back one word
- `Alt+F` - Move forward one word

### Editing

- `Ctrl+D` - Delete character under cursor (or execute if line empty)
- `Ctrl+H` / `Backspace` - Delete character before cursor
- `Ctrl+W` - Delete word before cursor
- `Ctrl+K` - Kill (cut) to end of line
- `Ctrl+U` - Kill (cut) to beginning of line
- `Ctrl+Y` - Yank (paste) killed text

### History

- `Up Arrow` / `Ctrl+P` - Previous command
- `Down Arrow` / `Ctrl+N` - Next command
- `Ctrl+R` - Search history (reverse)
- `Ctrl+S` - Search history (forward)

### Control

- `Ctrl+C` - Cancel current input
- `Ctrl+D` - Execute query (empty line) or exit
- `Ctrl+L` - Clear screen
- `Tab` - Auto-completion

---

## Configuration

### Configuration Options

```
# Display settings
set format table|json|csv|markdown     # Output format
set limit <n>|none                    # Result limit
set timing on|off                     # Show query time
set colors on|off                     # Syntax highlighting
set pager on|off                      # Enable result pagination

# Query settings
set explain on|off                    # Auto-explain queries
set profile on|off                    # Auto-profile queries
set timeout <ms>                      # Query timeout

# History settings
set history <n>                       # History size
set save-history on|off               # Persist history

# Display current config
show config
```

### Configuration File

Create `~/.oxirs/interactive.toml`:

```toml
[display]
format = "table"
limit = 100
timing = true
colors = true
pager = true

[query]
explain = false
profile = false
timeout = 30000  # 30 seconds

[history]
size = 1000
save = true
file = "~/.oxirs/history"

[prefixes]
# Default prefixes
rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
rdfs = "http://www.w3.org/2000/01/rdf-schema#"
foaf = "http://xmlns.com/foaf/0.1/"
```

---

## Tips & Tricks

### 1. Quick Statistics

```
# Count triples
oxirs[mydata]> SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }

# List classes with counts
oxirs[mydata]> SELECT ?class (COUNT(?s) AS ?count) WHERE {
...>   ?s a ?class
...> } GROUP BY ?class ORDER BY DESC(?count)

# List properties
oxirs[mydata]> SELECT DISTINCT ?p WHERE { ?s ?p ?o }
```

### 2. Explore Schema

```
# List all classes
oxirs[mydata]> SELECT DISTINCT ?class WHERE { ?s a ?class }

# List properties of a class
oxirs[mydata]> SELECT DISTINCT ?p WHERE { ?s a foaf:Person . ?s ?p ?o }

# Sample instances
oxirs[mydata]> SELECT ?s WHERE { ?s a foaf:Person } LIMIT 5
```

### 3. Filter and Search

```
# Text search (case-insensitive)
oxirs[mydata]> SELECT * WHERE {
...>   ?s ?p ?o .
...>   FILTER(REGEX(STR(?o), "alice", "i"))
...> }

# Date range queries
oxirs[mydata]> SELECT * WHERE {
...>   ?s ex:created ?date .
...>   FILTER(?date >= "2026-01-01"^^xsd:date &&
...>          ?date <= "2026-01-31"^^xsd:date)
...> }
```

### 4. Export Results

```
# Export to CSV
oxirs[mydata]> set format csv
oxirs[mydata]> SELECT ?name ?email WHERE { ?p foaf:name ?name . ?p foaf:mbox ?email }
# Copy output or redirect

# Export to JSON
oxirs[mydata]> set format json
oxirs[mydata]> SELECT * WHERE { ?s ?p ?o } LIMIT 100 > results.json
```

### 5. Macros and Aliases

```
# Define macro
oxirs> macro count "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }"

# Use macro
oxirs[mydata]> @count
+-------+
| count |
+-------+
| 125432|
+-------+

# List macros
oxirs> macros
```

### 6. Batch Operations

```
# Load queries from file
oxirs[mydata]> load queries.sparql

# Execute multiple updates
oxirs[mydata]> batch updates.sparql
Executing 10 updates...
10/10 successful
```

---

## Troubleshooting

### Common Issues

**Issue**: Query syntax error
```
oxirs[mydata]> SLECT * WHERE { ?s ?p ?o }
Error: Parse error at line 1, column 1: Expected 'SELECT', found 'SLECT'
```

**Solution**: Check spelling and SPARQL syntax.

**Issue**: Connection timeout
```
oxirs> connect slowdataset
Error: Connection timeout after 30s
```

**Solution**: Increase timeout with `set timeout 60000`.

**Issue**: Out of memory for large results
```
oxirs[mydata]> SELECT * WHERE { ?s ?p ?o }
Error: Out of memory
```

**Solution**: Add LIMIT clause or enable pagination with `set pager on`.

---

## Advanced Features

### Custom Output Formatters

```
# Register custom formatter
oxirs> formatter add custom-json "jq '.results.bindings[]'"

# Use custom formatter
oxirs> set format custom-json
```

### Query Analysis

```
# Analyze query complexity
oxirs[mydata]> analyze SELECT ?name WHERE {
...>   ?p1 foaf:knows ?p2 .
...>   ?p2 foaf:knows ?p3 .
...>   ?p3 foaf:name ?name
...> }
Complexity: HIGH
Estimated time: 2-5 seconds
Optimization suggestions:
  - Add index on foaf:knows
  - Consider limiting results
```

### Integration with External Tools

```
# Pipe to external command (Unix)
oxirs[mydata]> SELECT * WHERE { ?s ?p ?o } | grep "Alice"

# Export to clipboard
oxirs[mydata]> SELECT * WHERE { ?s ?p ?o } | pbcopy  # macOS
oxirs[mydata]> SELECT * WHERE { ?s ?p ?o } | xclip   # Linux
```

---

## See Also

- [COMMAND_REFERENCE.md](COMMAND_REFERENCE.md) - Full command reference
- [README.md](../README.md) - Getting started guide
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference

---

**OxiRS Interactive Mode v0.1.0** - Powerful REPL for RDF and SPARQL
