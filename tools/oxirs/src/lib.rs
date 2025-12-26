//! # OxiRS CLI Tool
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--rc.1-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs/badge.svg)](https://docs.rs/oxirs)
//!
//! **Status**: Beta Release (v0.1.0-rc.1)
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
//! ### Phase D: Industrial Connectivity (0.1.0-rc.1)
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
//! ## Phase D: Industrial Connectivity Examples (0.1.0-rc.1)
//!
//! ### Time-Series Operations
//! ```bash
//! # Query time-series with aggregation
//! oxirs tsdb query mykg --series 1 --start 2025-12-01T00:00:00Z --end 2025-12-25T23:59:59Z --aggregate avg
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

use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub mod cli;
pub mod cli_actions;
pub mod commands;
pub mod config;
pub mod export;
pub mod profiling;
pub mod tools;

// Re-export action enums for convenience
pub use cli_actions::*;

/// OxiRS CLI application
#[derive(Parser)]
#[command(name = "oxirs")]
#[command(about = "OxiRS command-line interface")]
#[command(version)]
#[command(
    long_about = "OxiRS command-line interface for RDF processing, SPARQL operations, and semantic data management.\n\nComplete documentation at https://oxirs.io/docs/cli"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Configuration file
    #[arg(short, long, global = true)]
    pub config: Option<PathBuf>,

    /// Suppress output (quiet mode)
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,

    /// Interactive mode (where applicable)
    #[arg(short, long, global = true)]
    pub interactive: bool,

    /// Configuration profile to use
    #[arg(short = 'P', long, global = true)]
    pub profile: Option<String>,

    /// Generate shell completion
    #[arg(long, value_enum, hide = true)]
    pub completion: Option<clap_complete::Shell>,
}

/// Available CLI commands
#[derive(Subcommand)]
pub enum Commands {
    /// Initialize a new knowledge graph dataset
    Init {
        /// Dataset name
        name: String,
        /// Storage format (tdb2, memory)
        #[arg(long, default_value = "tdb2")]
        format: String,
        /// Dataset location
        #[arg(short, long)]
        location: Option<PathBuf>,
    },
    /// Start the OxiRS server
    Serve {
        /// Configuration file or dataset path
        config: PathBuf,
        /// Server port
        #[arg(short, long, default_value = "3030")]
        port: u16,
        /// Server host
        #[arg(long, default_value = "localhost")]
        host: String,
        /// Enable GraphQL endpoint
        #[arg(long)]
        graphql: bool,
    },
    /// Import RDF data
    Import {
        /// Target dataset (alphanumeric, _, - only; no dots or extensions)
        dataset: String,
        /// Input file path
        file: PathBuf,
        /// Input format (turtle, ntriples, rdfxml, jsonld)
        #[arg(short, long)]
        format: Option<String>,
        /// Named graph URI
        #[arg(short, long)]
        graph: Option<String>,
        /// Resume from previous checkpoint if interrupted
        #[arg(long)]
        resume: bool,
    },
    /// Export RDF data
    Export {
        /// Source dataset (alphanumeric, _, - only; no dots or extensions)
        dataset: String,
        /// Output file path
        file: PathBuf,
        /// Output format (turtle, ntriples, rdfxml, jsonld)
        #[arg(short, long, default_value = "turtle")]
        format: String,
        /// Named graph URI
        #[arg(short, long)]
        graph: Option<String>,
        /// Resume from previous checkpoint if interrupted
        #[arg(long)]
        resume: bool,
    },
    /// Execute SPARQL query
    Query {
        /// Target dataset (alphanumeric, _, - only; no dots or extensions)
        dataset: String,
        /// SPARQL query string or file
        query: String,
        /// Query is a file path
        #[arg(short, long)]
        file: bool,
        /// Output format (json, csv, tsv, table, xml, html, markdown, md)
        #[arg(short, long, default_value = "table")]
        output: String,
    },
    /// Execute SPARQL update
    Update {
        /// Target dataset (alphanumeric, _, - only; no dots or extensions)
        dataset: String,
        /// SPARQL update string or file
        update: String,
        /// Update is a file path
        #[arg(short, long)]
        file: bool,
    },
    /// Run performance benchmarks and generate benchmark datasets
    Benchmark {
        #[command(subcommand)]
        action: BenchmarkAction,
    },
    /// Migrate data between formats/databases
    Migrate {
        #[command(subcommand)]
        action: MigrateAction,
    },
    /// Generate synthetic RDF datasets for testing and benchmarking
    Generate {
        /// Output file path
        output: PathBuf,
        /// Dataset size (tiny/small/medium/large/xlarge or number)
        #[arg(short, long, default_value = "small")]
        size: String,
        /// Dataset type (rdf/graph/semantic/bibliographic/geographic/organizational)
        #[arg(short = 't', long, default_value = "rdf")]
        r#type: String,
        /// Output format (turtle, ntriples, rdfxml, jsonld, trig, nquads, n3)
        #[arg(short, long, default_value = "turtle")]
        format: String,
        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,
        /// SHACL/RDFS/OWL schema file for constrained generation
        #[arg(long)]
        schema: Option<PathBuf>,
    },
    /// Manage database indexes for query performance
    Index {
        #[command(subcommand)]
        action: IndexAction,
    },
    /// Export RDF graph visualization
    Visualize {
        /// Dataset name or path
        dataset: String,
        /// Output file path
        output: PathBuf,
        /// Visualization format (dot/graphviz, mermaid/mmd, cytoscape/json)
        #[arg(short, long, default_value = "dot")]
        format: String,
        /// Specific graph to export (omit for all graphs)
        #[arg(short, long)]
        graph: Option<String>,
        /// Maximum number of nodes to include
        #[arg(long, default_value = "1000")]
        max_nodes: Option<usize>,
    },
    /// Manage server configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    // === Data Processing Tools ===
    /// RDF parsing and serialization (Jena riot equivalent)
    Riot {
        /// Input file(s)
        #[arg(required = true)]
        input: Vec<PathBuf>,
        /// Output format (turtle, ntriples, rdfxml, jsonld, trig, nquads)
        #[arg(long, default_value = "turtle")]
        output: String,
        /// Output file (stdout if not specified)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Input format (auto-detect if not specified)
        #[arg(long)]
        syntax: Option<String>,
        /// Base URI for resolving relative URIs
        #[arg(long)]
        base: Option<String>,
        /// Validate syntax only
        #[arg(long)]
        validate: bool,
        /// Count triples/quads
        #[arg(long)]
        count: bool,
    },

    /// Concatenate and convert RDF files
    RdfCat {
        /// Input files
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Output format
        #[arg(short, long, default_value = "turtle")]
        format: String,
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Copy RDF datasets with format conversion
    RdfCopy {
        /// Source dataset/file
        source: PathBuf,
        /// Target dataset/file
        target: PathBuf,
        /// Source format
        #[arg(long)]
        source_format: Option<String>,
        /// Target format
        #[arg(long)]
        target_format: Option<String>,
    },

    /// Compare RDF datasets
    RdfDiff {
        /// First dataset/file
        first: PathBuf,
        /// Second dataset/file
        second: PathBuf,
        /// Output format for differences
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// Validate RDF syntax
    RdfParse {
        /// Input file
        file: PathBuf,
        /// Input format
        #[arg(short, long)]
        format: Option<String>,
        /// Base URI
        #[arg(short, long)]
        base: Option<String>,
    },

    // === Advanced Query Tools ===
    /// Advanced SPARQL query processor (Jena arq equivalent)
    Arq {
        /// SPARQL query string or file
        #[arg(long)]
        query: Option<String>,
        /// Query file
        #[arg(long)]
        query_file: Option<PathBuf>,
        /// Data file(s)
        #[arg(long, action = clap::ArgAction::Append)]
        data: Vec<PathBuf>,
        /// Named graph data
        #[arg(long, action = clap::ArgAction::Append)]
        namedgraph: Vec<String>,
        /// Results format (table, csv, tsv, json, xml)
        #[arg(long, default_value = "table")]
        results: String,
        /// Dataset location
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Explain query execution
        #[arg(long)]
        explain: bool,
        /// Optimize query
        #[arg(long)]
        optimize: bool,
        /// Time query execution
        #[arg(long)]
        time: bool,
    },

    /// Remote SPARQL query execution
    RSparql {
        /// SPARQL endpoint URL
        #[arg(long)]
        service: String,
        /// SPARQL query
        #[arg(long)]
        query: Option<String>,
        /// Query file
        #[arg(long)]
        query_file: Option<PathBuf>,
        /// Results format
        #[arg(long, default_value = "table")]
        results: String,
        /// HTTP timeout in seconds
        #[arg(long, default_value = "30")]
        timeout: u64,
    },

    /// Remote SPARQL update execution
    RUpdate {
        /// SPARQL endpoint URL
        #[arg(long)]
        service: String,
        /// SPARQL update
        #[arg(long)]
        update: Option<String>,
        /// Update file
        #[arg(long)]
        update_file: Option<PathBuf>,
        /// HTTP timeout in seconds
        #[arg(long, default_value = "30")]
        timeout: u64,
    },

    /// SPARQL query parsing and validation
    QParse {
        /// Query string or file
        query: String,
        /// Query is a file path
        #[arg(short, long)]
        file: bool,
        /// Print AST
        #[arg(long)]
        print_ast: bool,
        /// Print algebra
        #[arg(long)]
        print_algebra: bool,
    },

    /// SPARQL update parsing and validation
    UParse {
        /// Update string or file
        update: String,
        /// Update is a file path
        #[arg(short, long)]
        file: bool,
        /// Print AST
        #[arg(long)]
        print_ast: bool,
    },

    // === Storage Tools ===
    /// Bulk data loading
    TdbLoader {
        /// Target dataset location
        location: PathBuf,
        /// Input files
        files: Vec<PathBuf>,
        /// Graph URI for loading
        #[arg(short, long)]
        graph: Option<String>,
        /// Show progress
        #[arg(long)]
        progress: bool,
        /// Statistics reporting
        #[arg(long)]
        stats: bool,
    },

    /// Dataset export and dumping
    TdbDump {
        /// Source dataset location
        location: PathBuf,
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format
        #[arg(short, long, default_value = "nquads")]
        format: String,
        /// Graph URI to dump
        #[arg(short, long)]
        graph: Option<String>,
    },

    /// Direct TDB querying
    TdbQuery {
        /// Dataset location
        location: PathBuf,
        /// SPARQL query
        query: String,
        /// Query is a file path
        #[arg(short, long)]
        file: bool,
        /// Results format
        #[arg(long, default_value = "table")]
        results: String,
    },

    /// Direct TDB updates
    TdbUpdate {
        /// Dataset location
        location: PathBuf,
        /// SPARQL update
        update: String,
        /// Update is a file path
        #[arg(short, long)]
        file: bool,
    },

    /// Database statistics
    TdbStats {
        /// Dataset location
        location: PathBuf,
        /// Detailed statistics
        #[arg(long)]
        detailed: bool,
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Database backup utilities with encryption support
    TdbBackup {
        /// Source dataset location
        source: PathBuf,
        /// Backup location
        target: PathBuf,
        /// Compress backup
        #[arg(long)]
        compress: bool,
        /// Incremental backup
        #[arg(long)]
        incremental: bool,
        /// Encrypt backup with AES-256-GCM
        #[arg(long)]
        encrypt: bool,
        /// Password for encryption (prompted if not provided)
        #[arg(long, requires = "encrypt")]
        password: Option<String>,
        /// Keyfile for encryption (alternative to password)
        #[arg(long, requires = "encrypt", conflicts_with = "password")]
        keyfile: Option<PathBuf>,
        /// Generate a new encryption keyfile
        #[arg(long, conflicts_with_all = ["encrypt", "password"])]
        generate_keyfile: Option<PathBuf>,
    },

    /// Database compaction
    TdbCompact {
        /// Dataset location
        location: PathBuf,
        /// Delete logs after compaction
        #[arg(long)]
        delete_old: bool,
    },

    /// Point-in-Time Recovery (PITR) operations
    Pitr {
        #[command(subcommand)]
        action: PitrAction,
    },

    // === Validation Tools ===
    /// SHACL validation
    Shacl {
        /// Data to validate
        #[arg(long)]
        data: Option<PathBuf>,
        /// Dataset location
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// SHACL shapes file
        #[arg(long)]
        shapes: PathBuf,
        /// Output format (text, turtle, json)
        #[arg(long, default_value = "text")]
        format: String,
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// ShEx validation
    Shex {
        /// Data to validate
        #[arg(long)]
        data: Option<PathBuf>,
        /// Dataset location
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// ShEx schema file
        #[arg(long)]
        schema: PathBuf,
        /// Shape map file
        #[arg(long)]
        shape_map: Option<PathBuf>,
        /// Output format
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Inference and reasoning
    Infer {
        /// Input data
        data: PathBuf,
        /// Ontology/schema file
        #[arg(long)]
        ontology: Option<PathBuf>,
        /// Reasoning profile (rdfs, owl-rl, custom)
        #[arg(long, default_value = "rdfs")]
        profile: String,
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format
        #[arg(long, default_value = "turtle")]
        format: String,
    },

    /// Schema generation from RDF
    SchemaGen {
        /// Input RDF data
        data: PathBuf,
        /// Schema type (shacl, shex, owl)
        #[arg(long, default_value = "shacl")]
        schema_type: String,
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Include statistics
        #[arg(long)]
        stats: bool,
    },

    /// SAMM Aspect Model tools (Java ESMF SDK compatible)
    Aspect {
        #[command(subcommand)]
        action: AspectAction,
    },

    /// Asset Administration Shell (AAS) tools (Java ESMF SDK compatible)
    Aas {
        #[command(subcommand)]
        action: AasAction,
    },

    /// Package management tools (Java ESMF SDK compatible)
    Package {
        #[command(subcommand)]
        action: PackageAction,
    },

    // === Utility Tools ===
    /// IRI validation and processing
    Iri {
        /// IRI to validate/process
        iri: String,
        /// Resolve relative IRI
        #[arg(long)]
        resolve: Option<String>,
        /// Check if IRI is valid
        #[arg(long)]
        validate: bool,
        /// Normalize IRI
        #[arg(long)]
        normalize: bool,
    },

    /// Language tag validation
    LangTag {
        /// Language tag to validate
        tag: String,
        /// Check if tag is well-formed
        #[arg(long)]
        validate: bool,
        /// Normalize tag
        #[arg(long)]
        normalize: bool,
    },

    /// UUID generation for blank nodes
    JUuid {
        /// Number of UUIDs to generate
        #[arg(short, long, default_value = "1")]
        count: usize,
        /// Output format (uuid, urn, bnode)
        #[arg(short, long, default_value = "uuid")]
        format: String,
    },

    /// UTF-8 encoding utilities
    Utf8 {
        /// Input file or string
        input: String,
        /// Input is a file path
        #[arg(short, long)]
        file: bool,
        /// Check UTF-8 validity
        #[arg(long)]
        validate: bool,
        /// Fix UTF-8 encoding issues
        #[arg(long)]
        fix: bool,
    },

    /// URL encoding
    WwwEnc {
        /// String to encode
        input: String,
        /// Encoding type (url, form)
        #[arg(long, default_value = "url")]
        encoding: String,
    },

    /// URL decoding
    WwwDec {
        /// String to decode
        input: String,
        /// Decoding type (url, form)
        #[arg(long, default_value = "url")]
        decoding: String,
    },

    /// Result set processing
    RSet {
        /// Input results file
        input: PathBuf,
        /// Input format (csv, tsv, json, xml)
        #[arg(long)]
        input_format: Option<String>,
        /// Output format (csv, tsv, json, xml, table)
        #[arg(long, default_value = "table")]
        output_format: String,
        /// Output file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Start interactive REPL mode
    Interactive {
        /// Initial dataset to connect to
        #[arg(short, long)]
        dataset: Option<String>,
        /// History file location
        #[arg(long)]
        history: Option<PathBuf>,
    },

    /// Performance monitoring and profiling
    Performance {
        #[command(subcommand)]
        action: commands::performance::PerformanceCommand,
    },

    /// Query explanation and analysis
    Explain {
        /// Target dataset
        dataset: String,
        /// SPARQL query string or file
        query: String,
        /// Query is a file path
        #[arg(short, long)]
        file: bool,
        /// Analysis mode (explain, analyze, full)
        #[arg(short, long, default_value = "explain")]
        mode: String,
        /// Generate graphical query plan (Graphviz DOT format)
        #[arg(short, long)]
        graphviz: Option<PathBuf>,
    },

    /// Query optimization analyzer
    Optimize {
        /// SPARQL query string or file
        query: String,
        /// Query is a file path
        #[arg(short, long)]
        file: bool,
    },

    /// SPARQL query template management
    Template {
        #[command(subcommand)]
        action: TemplateAction,
    },

    /// Query history management
    History {
        #[command(subcommand)]
        action: HistoryAction,
    },

    /// CI/CD integration tools
    Cicd {
        #[command(subcommand)]
        action: CicdAction,
    },

    /// Command alias management
    Alias {
        #[command(subcommand)]
        action: AliasAction,
    },

    /// Query cache management
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },

    /// ReBAC relationship management
    Rebac(commands::rebac::RebacArgs),

    /// Generate CLI documentation
    Docs {
        /// Output format (markdown, html, man, text)
        #[arg(short, long, default_value = "markdown")]
        format: String,
        /// Output file path (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Generate documentation for specific command
        #[arg(long)]
        command: Option<String>,
    },

    /// Interactive tutorial mode for learning OxiRS
    Tutorial {
        /// Start at specific lesson
        #[arg(short, long)]
        lesson: Option<String>,
    },

    /// Advanced RDF graph analytics using scirs2-graph
    GraphAnalytics {
        /// Dataset name or path
        dataset: String,
        /// Analytics operation (pagerank, community, betweenness, closeness, degree, paths, stats)
        #[arg(short, long, default_value = "pagerank")]
        operation: String,
        /// Damping factor for PageRank
        #[arg(long, default_value = "0.85")]
        damping: f64,
        /// Maximum iterations for iterative algorithms
        #[arg(long, default_value = "100")]
        max_iter: usize,
        /// Convergence tolerance
        #[arg(long, default_value = "0.000001")]
        tolerance: f64,
        /// Source node URI for shortest paths
        #[arg(long)]
        source: Option<String>,
        /// Target node URI for shortest paths
        #[arg(long)]
        target: Option<String>,
        /// Top K results to display
        #[arg(short = 'k', long, default_value = "20")]
        top: usize,
    },

    // === Phase D: Industrial Connectivity (0.1.0-rc.1) ===
    /// Time-series database operations
    Tsdb {
        #[command(subcommand)]
        action: TsdbAction,
    },

    /// Modbus protocol monitoring and configuration
    Modbus {
        #[command(subcommand)]
        action: ModbusAction,
    },

    /// CANbus/J1939 monitoring and DBC parsing
    Canbus {
        #[command(subcommand)]
        action: CanbusAction,
    },
}

/// Run the CLI application
pub async fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    use cli::{completion, CliContext};

    // Handle shell completion generation
    if let Some(shell) = cli.completion {
        use clap::CommandFactory;
        let mut app = Cli::command();
        completion::print_completions(shell, &mut app);
        return Ok(());
    }

    // Create CLI context
    let ctx = CliContext::from_cli(cli.verbose, cli.quiet, cli.no_color);

    // Initialize structured logging
    let log_format = if std::env::var("OXIRS_LOG_FORMAT").as_deref() == Ok("json") {
        cli::LogFormat::Json
    } else if ctx.verbose {
        cli::LogFormat::Pretty
    } else {
        cli::LogFormat::Text
    };

    let log_config = cli::LogConfig {
        level: if ctx.verbose {
            "debug".to_string()
        } else if ctx.quiet {
            "error".to_string()
        } else {
            std::env::var("OXIRS_LOG_LEVEL").unwrap_or_else(|_| "info".to_string())
        },
        format: log_format,
        timestamps: !ctx.quiet,
        source_location: ctx.verbose,
        thread_ids: false,
        perf_threshold_ms: std::env::var("OXIRS_PERF_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok()),
        file: std::env::var("OXIRS_LOG_FILE").ok(),
    };

    cli::init_logging(&log_config).expect("Failed to initialize logging");

    // Show startup message if not quiet
    if ctx.should_show_output() {
        ctx.info(&format!("Oxirs CLI v{}", env!("CARGO_PKG_VERSION")));
    }

    match cli.command {
        Commands::Init {
            name,
            format,
            location,
        } => commands::init::run(name, format, location)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Serve {
            config,
            port,
            host,
            graphql,
        } => commands::serve::run(config, port, host, graphql)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Import {
            dataset,
            file,
            format,
            graph,
            resume,
        } => commands::import::run(dataset, file, format, graph, resume)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Export {
            dataset,
            file,
            format,
            graph,
            resume,
        } => commands::export::run(dataset, file, format, graph, resume)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Query {
            dataset,
            query,
            file,
            output,
        } => commands::query::run(dataset, query, file, output)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Update {
            dataset,
            update,
            file,
        } => commands::update::run(dataset, update, file)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Benchmark { action } => match action {
            BenchmarkAction::Run {
                dataset,
                suite,
                iterations,
                output,
                detailed,
                warmup,
            } => commands::benchmark::run(dataset, suite, iterations, output, detailed, warmup)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            BenchmarkAction::Generate {
                output,
                size,
                dataset_type,
                seed,
                triples,
                schema,
            } => commands::benchmark::generate(output, size, dataset_type, seed, triples, schema)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            BenchmarkAction::Analyze {
                input,
                output,
                format,
                suggestions,
                patterns,
            } => commands::benchmark::analyze(input, output, format, suggestions, patterns)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            BenchmarkAction::Compare {
                baseline,
                current,
                output,
                threshold,
                format,
            } => commands::benchmark::compare(baseline, current, output, threshold, format)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        },
        Commands::Migrate { action } => match action {
            MigrateAction::Format {
                source,
                target,
                from,
                to,
            } => commands::migrate::format(source, target, from, to)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromTdb1 {
                tdb_dir,
                dataset,
                skip_validation,
            } => commands::migrate::from_tdb1(tdb_dir, dataset, skip_validation)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromTdb2 {
                tdb_dir,
                dataset,
                skip_validation,
            } => commands::migrate::from_tdb2(tdb_dir, dataset, skip_validation)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromVirtuoso {
                connection,
                dataset,
                graphs,
            } => commands::migrate::from_virtuoso(connection, dataset, graphs)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromRdf4j { repo_dir, dataset } => {
                commands::migrate::from_rdf4j(repo_dir, dataset)
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            MigrateAction::FromBlazegraph {
                endpoint,
                dataset,
                namespace,
            } => commands::migrate::from_blazegraph(endpoint, dataset, namespace)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            MigrateAction::FromGraphdb {
                endpoint,
                dataset,
                repository,
            } => commands::migrate::from_graphdb(endpoint, dataset, repository)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        },
        Commands::Generate {
            output,
            size,
            r#type,
            format,
            seed,
            schema,
        } => commands::generate::run(output, size, r#type, format, seed, schema)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Index { action } => match action {
            IndexAction::List { dataset } => commands::index::list(dataset)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            IndexAction::Rebuild { dataset, index } => commands::index::rebuild(dataset, index)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            IndexAction::Stats { dataset, format } => commands::index::stats(dataset, format)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
            IndexAction::Optimize { dataset } => commands::index::optimize(dataset)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        },
        Commands::Visualize {
            dataset,
            output,
            format,
            graph,
            max_nodes,
        } => commands::visualize::export(dataset, output, format, graph, max_nodes)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Config { action } => commands::config::run(action)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),

        // Data Processing Tools
        Commands::Riot {
            input,
            output,
            out,
            syntax,
            base,
            validate,
            count,
        } => tools::riot::run(input, output, out, syntax, base, validate, count).await,
        Commands::RdfCat {
            files,
            format,
            output,
        } => tools::rdfcat::run(files, format, output).await,
        Commands::RdfCopy {
            source,
            target,
            source_format,
            target_format,
        } => tools::rdfcopy::run(source, target, source_format, target_format).await,
        Commands::RdfDiff {
            first,
            second,
            format,
        } => tools::rdfdiff::run(first, second, format).await,
        Commands::RdfParse { file, format, base } => tools::rdfparse::run(file, format, base).await,

        // Advanced Query Tools
        Commands::Arq {
            query,
            query_file,
            data,
            namedgraph,
            results,
            dataset,
            explain,
            optimize,
            time,
        } => {
            tools::arq::run(tools::arq::ArqConfig {
                query,
                query_file,
                data,
                namedgraph,
                results_format: results,
                dataset,
                explain,
                optimize,
                time,
            })
            .await
        }
        Commands::RSparql {
            service,
            query,
            query_file,
            results,
            timeout,
        } => tools::rsparql::run(service, query, query_file, results, timeout).await,
        Commands::RUpdate {
            service,
            update,
            update_file,
            timeout,
        } => tools::rupdate::run(service, update, update_file, timeout).await,
        Commands::QParse {
            query,
            file,
            print_ast,
            print_algebra,
        } => tools::qparse::run(query, file, print_ast, print_algebra).await,
        Commands::UParse {
            update,
            file,
            print_ast,
        } => tools::uparse::run(update, file, print_ast).await,

        // Storage Tools
        Commands::TdbLoader {
            location,
            files,
            graph,
            progress,
            stats,
        } => tools::tdbloader::run(location, files, graph, progress, stats).await,
        Commands::TdbDump {
            location,
            output,
            format,
            graph,
        } => tools::tdbdump::run(location, output, format, graph).await,
        Commands::TdbQuery {
            location,
            query,
            file,
            results,
        } => tools::tdbquery::run(location, query, file, results).await,
        Commands::TdbUpdate {
            location,
            update,
            file,
        } => tools::tdbupdate::run(location, update, file).await,
        Commands::TdbStats {
            location,
            detailed,
            format,
        } => tools::tdbstats::run(location, detailed, format).await,
        Commands::TdbBackup {
            source,
            target,
            compress,
            incremental,
            encrypt,
            password,
            keyfile,
            generate_keyfile,
        } => {
            use tools::backup_encryption;

            // Handle keyfile generation
            if let Some(keyfile_path) = generate_keyfile {
                println!("Generating encryption keyfile...");
                backup_encryption::generate_keyfile(&keyfile_path)?;
                println!(
                    "Keyfile generated successfully at: {}",
                    keyfile_path.display()
                );
                println!(
                    "⚠️  Keep this keyfile secure! Loss of the keyfile means loss of data access."
                );
                return Ok(());
            }

            // Clone target for encryption if needed
            let target_for_encryption = target.clone();

            // Run backup
            tools::tdbbackup::run(source, target, compress, incremental).await?;

            // Encrypt backup if requested
            if encrypt {
                use dialoguer::Password;

                println!("\nEncrypting backup...");
                let backup_file = &target_for_encryption;
                let encrypted_file = backup_file.with_extension("oxirs.enc");

                let encryption_config = if let Some(ref pwd) = password {
                    backup_encryption::EncryptionConfig {
                        password: Some(pwd.clone()),
                        keyfile: None,
                        verify: true,
                    }
                } else if let Some(ref kf) = keyfile {
                    backup_encryption::EncryptionConfig {
                        password: None,
                        keyfile: Some(kf.clone()),
                        verify: true,
                    }
                } else {
                    // Prompt for password
                    let pwd = Password::new()
                        .with_prompt("Enter encryption password")
                        .with_confirmation("Confirm password", "Passwords don't match")
                        .interact()?;

                    backup_encryption::EncryptionConfig {
                        password: Some(pwd),
                        keyfile: None,
                        verify: true,
                    }
                };

                backup_encryption::encrypt_backup(
                    backup_file,
                    &encrypted_file,
                    &encryption_config,
                )?;
                println!(
                    "✓ Backup encrypted successfully: {}",
                    encrypted_file.display()
                );
            }
            Ok(())
        }
        Commands::TdbCompact {
            location,
            delete_old,
        } => tools::tdbcompact::run(location, delete_old).await,

        Commands::Pitr { action } => {
            use chrono::{DateTime, Utc};
            use tools::pitr::{PitrConfig, TransactionLog};

            match action {
                PitrAction::Init {
                    dataset,
                    max_log_size,
                    auto_archive,
                } => {
                    println!("Initializing PITR for dataset: {}", dataset.display());
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: max_log_size * 1_048_576, // Convert MB to bytes
                        auto_archive,
                    };
                    let _log = TransactionLog::new(config)?;
                    println!("✓ PITR initialized successfully");
                }
                PitrAction::Checkpoint { dataset, name } => {
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let log = TransactionLog::new(config)?;
                    let checkpoint_path = log.create_checkpoint(&name)?;
                    println!("✓ Checkpoint created: {}", checkpoint_path.display());
                }
                PitrAction::List { dataset, format } => {
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let log = TransactionLog::new(config)?;
                    let checkpoints = log.list_checkpoints()?;

                    if format == "json" {
                        println!("{}", serde_json::to_string_pretty(&checkpoints)?);
                    } else {
                        println!("Available Checkpoints:");
                        println!("{:-<80}", "");
                        for cp in checkpoints {
                            println!("Name: {}", cp.name);
                            println!("  Timestamp: {}", cp.timestamp.to_rfc3339());
                            println!("  Last Transaction ID: {}", cp.last_transaction_id);
                            println!("  Log Files: {}", cp.log_files.len());
                            println!();
                        }
                    }
                }
                PitrAction::RecoverTimestamp {
                    dataset,
                    timestamp,
                    output,
                } => {
                    let target_time: DateTime<Utc> = timestamp.parse()?;
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let log = TransactionLog::new(config)?;
                    let count = log.recover_to_timestamp(target_time, &output)?;
                    println!("✓ Recovered {} transactions to {}", count, output.display());
                }
                PitrAction::RecoverTransaction {
                    dataset,
                    transaction_id,
                    output,
                } => {
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let log = TransactionLog::new(config)?;
                    let count = log.recover_to_transaction(transaction_id, &output)?;
                    println!("✓ Recovered {} transactions to {}", count, output.display());
                }
                PitrAction::Archive { dataset } => {
                    let config = PitrConfig {
                        log_dir: dataset.join("pitr/logs"),
                        archive_dir: dataset.join("pitr/archive"),
                        max_log_size: 100_000_000,
                        auto_archive: false,
                    };
                    let mut log = TransactionLog::new(config)?;
                    let archived = log.archive_logs()?;
                    println!("✓ Archived {} log files", archived);
                }
            }
            Ok(())
        }

        // Validation Tools
        Commands::Shacl {
            data,
            dataset,
            shapes,
            format,
            output,
        } => tools::shacl::run(data, dataset, shapes, format, output).await,
        Commands::Shex {
            data,
            dataset,
            schema,
            shape_map,
            format,
        } => tools::shex::run(data, dataset, schema, shape_map, format).await,
        Commands::Infer {
            data,
            ontology,
            profile,
            output,
            format,
        } => tools::infer::run(data, ontology, profile, output, format).await,
        Commands::SchemaGen {
            data,
            schema_type,
            output,
            stats,
        } => tools::schemagen::run(data, schema_type, output, stats).await,
        Commands::Aspect { action } => commands::aspect::run(action)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Aas { action } => commands::aas::run(action)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Package { action } => commands::package::run(action)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),

        // Utility Tools
        Commands::Iri {
            iri,
            resolve,
            validate,
            normalize,
        } => tools::iri::run(iri, resolve, validate, normalize).await,
        Commands::LangTag {
            tag,
            validate,
            normalize,
        } => tools::langtag::run(tag, validate, normalize).await,
        Commands::JUuid { count, format } => tools::juuid::run(count, format).await,
        Commands::Utf8 {
            input,
            file,
            validate,
            fix,
        } => tools::utf8::run(input, file, validate, fix).await,
        Commands::WwwEnc { input, encoding } => tools::wwwenc::run(input, encoding).await,
        Commands::WwwDec { input, decoding } => tools::wwwdec::run(input, decoding).await,
        Commands::RSet {
            input,
            input_format,
            output_format,
            output,
        } => tools::rset::run(input, input_format, output_format, output).await,
        Commands::Interactive {
            dataset,
            history: _,
        } => {
            ctx.info("Starting interactive SPARQL shell...");
            commands::interactive::execute(dataset, cli.config)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        }
        Commands::Performance { action } => {
            let config = config::Config::default();
            action
                .execute(&config)
                .await
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        }
        Commands::Explain {
            dataset,
            query,
            file,
            mode,
            graphviz,
        } => {
            let analysis_mode = match mode.to_lowercase().as_str() {
                "explain" => commands::explain::AnalysisMode::Explain,
                "analyze" => commands::explain::AnalysisMode::Analyze,
                "full" => commands::explain::AnalysisMode::Full,
                _ => {
                    eprintln!(
                        "Invalid mode '{}'. Valid modes: explain, analyze, full",
                        mode
                    );
                    return Err("Invalid analysis mode".into());
                }
            };
            commands::explain::explain_query_with_options(
                dataset,
                query,
                file,
                analysis_mode,
                graphviz,
            )
            .await
            .map_err(|e| e.into())
        }
        Commands::Optimize { query, file } => {
            commands::query_optimizer::optimize_command(query, file)
                .await
                .map_err(|e| e.into())
        }
        Commands::Template { action } => {
            use std::collections::HashMap;
            match action {
                TemplateAction::List { category } => commands::templates::list_command(category)
                    .await
                    .map_err(|e| e.into()),
                TemplateAction::Show { name } => commands::templates::show_command(name)
                    .await
                    .map_err(|e| e.into()),
                TemplateAction::Render { name, param } => {
                    let mut params = HashMap::new();
                    for p in param {
                        let parts: Vec<&str> = p.splitn(2, '=').collect();
                        if parts.len() != 2 {
                            eprintln!("Invalid parameter format: '{}'. Expected key=value", p);
                            return Err("Invalid parameter format".into());
                        }
                        params.insert(parts[0].to_string(), parts[1].to_string());
                    }
                    commands::templates::render_command(name, params)
                        .await
                        .map_err(|e| e.into())
                }
            }
        }
        Commands::History { action } => match action {
            HistoryAction::List { limit, dataset } => {
                commands::history::commands::list_command(limit, dataset)
                    .await
                    .map_err(|e| e.into())
            }
            HistoryAction::Show { id } => commands::history::commands::show_command(id)
                .await
                .map_err(|e| e.into()),
            HistoryAction::Replay { id, output } => {
                commands::history::commands::replay_command(id, output)
                    .await
                    .map_err(|e| e.into())
            }
            HistoryAction::Search { query } => commands::history::commands::search_command(query)
                .await
                .map_err(|e| e.into()),
            HistoryAction::Clear => commands::history::commands::clear_command()
                .await
                .map_err(|e| e.into()),
            HistoryAction::Stats => commands::history::commands::stats_command()
                .await
                .map_err(|e| e.into()),
            HistoryAction::Analytics { dataset } => {
                commands::history::commands::analytics_command(dataset)
                    .await
                    .map_err(|e| e.into())
            }
        },
        Commands::Cicd { action } => match action {
            CicdAction::Report {
                input,
                output,
                format,
            } => commands::cicd::generate_test_report(input, output, format)
                .await
                .map_err(|e| e.into()),
            CicdAction::Docker { output } => commands::cicd::generate_docker_files(output)
                .await
                .map_err(|e| e.into()),
            CicdAction::Github { output } => commands::cicd::generate_github_workflow(output)
                .await
                .map_err(|e| e.into()),
            CicdAction::Gitlab { output } => commands::cicd::generate_gitlab_ci(output)
                .await
                .map_err(|e| e.into()),
        },
        Commands::Alias { action } => match action {
            AliasAction::List => commands::alias::list().await.map_err(|e| e.into()),
            AliasAction::Show { name } => commands::alias::show(name.clone())
                .await
                .map_err(|e| e.into()),
            AliasAction::Add { name, command } => {
                commands::alias::add(name.clone(), command.clone())
                    .await
                    .map_err(|e| e.into())
            }
            AliasAction::Remove { name } => commands::alias::remove(name.clone())
                .await
                .map_err(|e| e.into()),
            AliasAction::Reset => commands::alias::reset().await.map_err(|e| e.into()),
        },

        Commands::Cache { action } => match action {
            CacheAction::Stats => commands::cache::commands::stats_command()
                .await
                .map_err(|e| e.into()),
            CacheAction::Clear => commands::cache::commands::clear_command()
                .await
                .map_err(|e| e.into()),
            CacheAction::Config { ttl, max_size } => {
                commands::cache::commands::config_command(ttl, max_size)
                    .await
                    .map_err(|e| e.into())
            }
        },

        Commands::Rebac(args) => commands::rebac::execute(args).await.map_err(|e| e.into()),

        Commands::Docs {
            format,
            output,
            command,
        } => {
            use cli::doc_generator::{DocFormat, DocGenerator};
            use std::io::Write;

            let doc_format: DocFormat = format
                .parse()
                .map_err(|e: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

            let generator = DocGenerator::new();

            if let Some(cmd_name) = command {
                ctx.info(&format!(
                    "Generating documentation for command: {}",
                    cmd_name
                ));
                // Generate single command docs (future enhancement)
                ctx.warn(
                    "Single command documentation not yet implemented. Generating all commands.",
                );
            }

            let content = generator
                .generate(doc_format)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

            if let Some(output_path) = output {
                let mut file = std::fs::File::create(&output_path)?;
                file.write_all(content.as_bytes())?;
                ctx.success(&format!(
                    "Documentation written to: {}",
                    output_path.display()
                ));
            } else {
                println!("{}", content);
            }

            Ok(())
        }

        Commands::Tutorial { lesson } => {
            use cli::tutorial::TutorialManager;

            let mut manager = TutorialManager::new();

            if let Some(lesson_name) = lesson {
                ctx.info(&format!("Starting tutorial with lesson: {}", lesson_name));
                ctx.warn(
                    "Specific lesson selection not yet implemented. Starting interactive tutorial.",
                );
            }

            manager.start().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, format!("Tutorial error: {}", e))
            })?;

            Ok(())
        }

        Commands::GraphAnalytics {
            dataset,
            operation,
            damping,
            max_iter,
            tolerance,
            source,
            target,
            top,
        } => {
            use commands::graph_analytics::{
                execute_graph_analytics, AnalyticsConfig, AnalyticsOperation,
            };
            use std::path::Path;

            // Parse operation
            let op: AnalyticsOperation = operation
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

            // Build configuration
            let config = AnalyticsConfig {
                operation: op,
                damping_factor: damping,
                max_iterations: max_iter,
                tolerance,
                source_node: source.clone(),
                target_node: target.clone(),
                top_k: top,
                katz_alpha: 0.1,            // Default Katz centrality alpha parameter
                katz_beta: 1.0,             // Default Katz centrality beta parameter
                k_core_value: None,         // Auto-detect all cores
                enable_simd: true,          // Auto-enable SIMD optimizations
                enable_parallel: true,      // Auto-enable parallel processing
                enable_gpu: false,          // GPU is opt-in (requires hardware)
                enable_cache: true,         // Enable caching for better performance
                export_path: None,          // No export by default
                enable_benchmarking: false, // Disable benchmarking by default
            };

            // Execute analytics
            let dataset_path = Path::new(dataset.as_str());
            execute_graph_analytics(dataset_path, &config)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

            Ok(())
        }

        // === Phase D: Industrial Connectivity CLI Handlers (0.1.0-rc.1) ===
        Commands::Tsdb { action } => commands::tsdb::execute(action, &ctx)
            .await
            .map_err(|e| e.into()),

        Commands::Modbus { action } => commands::modbus::execute(action, &ctx)
            .await
            .map_err(|e| e.into()),

        Commands::Canbus { action } => commands::canbus::execute(action, &ctx)
            .await
            .map_err(|e| e.into()),
    }
}
