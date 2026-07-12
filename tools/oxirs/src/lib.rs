//! # OxiRS CLI Tool
//!
//! [![Version](https://img.shields.io/badge/version-0.3.2-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs/badge.svg)](https://docs.rs/oxirs)
//!
//! **Status**: Production Release (v0.3.2)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! Command-line interface for OxiRS providing import, export, SPARQL queries,
//! benchmarking, and server management tools.
//!
//! ## Features
//!
//! - ✅ **Persistent RDF Storage**: Data automatically saved to disk in N-Quads format
//! - ✅ **SPARQL Queries**: Support for SELECT, ASK, CONSTRUCT, and DESCRIBE queries
//! - ✅ **Multi-format Import/Export**: Turtle, N-Triples, RDF/XML, JSON-LD, N-Quads, TriG
//! - ✅ **Interactive REPL**: Explore RDF data interactively
//! - 🚧 **Prefix Support**: Coming soon in next release
//!
//! ## Commands
//!
//! ### Core RDF Operations
//! - `init`: Initialize a new knowledge graph dataset
//! - `import`: Import RDF data from various formats (data persisted automatically)
//! - `query`: Execute SPARQL queries (SELECT, ASK, CONSTRUCT, DESCRIBE)
//! - `export`: Export RDF data to various formats
//! - `interactive`: Interactive REPL for SPARQL queries
//! - `serve`: Start the OxiRS SPARQL server
//! - `benchmark`: Run performance benchmarks
//!
//! ### Phase D: Industrial Connectivity
//! - `tsdb`: Time-series database operations with SPARQL temporal extensions
//! - `modbus`: Modbus TCP/RTU monitoring and RDF mapping
//! - `canbus`: CANbus/J1939 monitoring, DBC parsing, SAMM generation
//!
//! ### Storage Tools
//! - `tdbloader`, `tdbquery`, `tdbstats`, `tdbbackup`, `tdbcompact`
//!
//! ### Validation Tools
//! - `shacl`: SHACL shape validation
//! - `shex`: ShEx validation
//! - `infer`: Reasoning and inference
//!
//! ### SAMM/AAS Tools (Java ESMF SDK compatible)
//! - `aspect`: SAMM Aspect Model tools
//! - `aas`: Asset Administration Shell tools
//! - `package`: Package management
//!
//! ### Advanced Tools
//! - `graph-analytics`: RDF graph analytics using scirs2-graph
//! - Various utilities: `arq`, `riot`, `rdfcat`, etc.
//!
//! ## Quick Start
//!
//! ```bash
//! # 1. Initialize a new dataset
//! oxirs init mykg
//!
//! # 2. Import RDF data (automatically persisted to mykg/data.nq)
//! oxirs import mykg data.ttl --format turtle
//!
//! # 3. Query the data (data loaded from disk automatically)
//! oxirs query mykg "SELECT * WHERE { ?s ?p ?o } LIMIT 10"
//!
//! # 4. Query with specific patterns
//! oxirs query mykg "SELECT ?name WHERE { ?person <http://example.org/name> ?name }"
//!
//! # 5. Start SPARQL server
//! oxirs serve mykg/oxirs.toml --port 3030
//! ```
//!
//! ## Dataset Name Rules
//!
//! Dataset names must follow these rules:
//! - Only letters (a-z, A-Z), numbers (0-9), underscores (_), and hyphens (-)
//! - No dots (.), slashes (/), or other special characters
//! - Maximum length: 255 characters
//! - Cannot be empty
//!
//! Valid examples: `mykg`, `my_dataset`, `test-data-2024`
//! Invalid examples: `dataset.oxirs`, `my/data`, `data.ttl`
//!
//! ## SPARQL Query Examples
//!
//! ```bash
//! # Get all triples
//! oxirs query mykg "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
//!
//! # Filter by type
//! oxirs query mykg "SELECT ?s WHERE {
//!   ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person>
//! }"
//!
//! # ASK query (returns true/false)
//! oxirs query mykg "ASK { ?s <http://example.org/age> \"30\" }"
//!
//! # CONSTRUCT new triples
//! oxirs query mykg "CONSTRUCT { ?s <http://example.org/hasName> ?name }
//!                   WHERE { ?s <http://example.org/name> ?name }"
//! ```
//!
//! ## Phase D: Industrial Connectivity Examples (0.3.0)
//!
//! ### Time-Series Operations
//! ```bash
//! # Query time-series with aggregation
//! oxirs tsdb query mykg --series 1 --start 2026-01-01T00:00:00Z --end 2026-01-31T23:59:59Z --aggregate avg
//!
//! # Insert data point
//! oxirs tsdb insert mykg --series 1 --value 22.5
//!
//! # Show compression statistics
//! oxirs tsdb stats mykg --detailed
//!
//! # Export to CSV
//! oxirs tsdb export mykg --series 1 --output data.csv --format csv
//! ```
//!
//! ### Modbus Operations
//! ```bash
//! # Monitor Modbus TCP device (real-time)
//! oxirs modbus monitor-tcp --address 192.168.1.100:502 --start 40001 --count 10 --interval 1000
//!
//! # Read registers
//! oxirs modbus read --device 192.168.1.100:502 --address 40001 --count 5 --datatype float32
//!
//! # Generate RDF from Modbus data
//! oxirs modbus to-rdf --device 192.168.1.100:502 --config modbus_map.toml --output data.ttl
//!
//! # Start mock server for testing
//! oxirs modbus mock-server --port 5020
//! ```
//!
//! ### CANbus Operations
//! ```bash
//! # Monitor CAN interface
//! oxirs canbus monitor --interface can0 --dbc vehicle.dbc --j1939
//!
//! # Parse DBC file
//! oxirs canbus parse-dbc --file vehicle.dbc --detailed
//!
//! # Decode CAN frame
//! oxirs canbus decode --id 0x0CF00400 --data DEADBEEF --dbc vehicle.dbc
//!
//! # Generate SAMM Aspect Models from DBC
//! oxirs canbus to-samm --dbc vehicle.dbc --output ./models/
//!
//! # Generate RDF from live CAN data
//! oxirs canbus to-rdf --interface can0 --dbc vehicle.dbc --output can_data.ttl --count 1000
//! ```
//!
//! ## Data Persistence
//!
//! - Data is automatically saved to `<dataset>/data.nq` in N-Quads format
//! - On `oxirs import`, data is appended and persisted
//! - On `oxirs query`, data is loaded from disk automatically
//! - No manual save/load commands needed!

pub mod cli;
pub mod cli_actions;
pub mod commands;
pub mod config;
pub mod export;
pub mod lib_commands;
pub mod lib_dispatch;
pub mod profiling;
pub mod tools;

// Re-export action enums for convenience
pub use cli_actions::*;

// Re-export CLI types from lib_commands
pub use lib_commands::{Cli, Commands};

// Re-export the main run function from lib_dispatch
pub use lib_dispatch::run;
