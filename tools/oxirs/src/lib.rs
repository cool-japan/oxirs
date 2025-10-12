//! # OxiRS CLI Tool
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--alpha.2-orange)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs/badge.svg)](https://docs.rs/oxirs)
//!
//! **Status**: Alpha Release (v0.1.0-alpha.3)
//! ⚠️ APIs may change. Not recommended for production use.
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
//! - `init`: Initialize a new knowledge graph dataset
//! - `import`: Import RDF data from various formats (data persisted automatically)
//! - `query`: Execute SPARQL queries (SELECT, ASK, CONSTRUCT, DESCRIBE)
//! - `export`: Export RDF data to various formats
//! - `interactive`: Interactive REPL for SPARQL queries
//! - `serve`: Start the OxiRS SPARQL server
//! - `benchmark`: Run performance benchmarks
//! - Various tools: `tdbloader`, `tdbquery`, `arq`, `sparql`, etc.
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
//! ## Data Persistence
//!
//! - Data is automatically saved to `<dataset>/data.nq` in N-Quads format
//! - On `oxirs import`, data is appended and persisted
//! - On `oxirs query`, data is loaded from disk automatically
//! - No manual save/load commands needed!

use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub mod benchmark;
pub mod cli;
pub mod commands;
pub mod config;
pub mod export;
pub mod import;
pub mod server;
pub mod tools;

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
        /// Output format (json, csv, tsv, table)
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
    /// Run performance benchmarks
    Benchmark {
        /// Target dataset (alphanumeric, _, - only; no dots or extensions)
        dataset: String,
        /// Benchmark suite (sp2bench, watdiv, ldbc)
        #[arg(short, long, default_value = "sp2bench")]
        suite: String,
        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,
        /// Output report file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Migrate data between formats/versions
    Migrate {
        /// Source dataset path
        source: PathBuf,
        /// Target dataset path
        target: PathBuf,
        /// Source format
        #[arg(long)]
        from: String,
        /// Target format
        #[arg(long)]
        to: String,
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

    /// Database backup utilities
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
    },

    /// Database compaction
    TdbCompact {
        /// Dataset location
        location: PathBuf,
        /// Delete logs after compaction
        #[arg(long)]
        delete_old: bool,
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
}

/// Configuration management actions
#[derive(Subcommand)]
pub enum ConfigAction {
    /// Generate a default configuration file
    Init {
        /// Output file path
        #[arg(short, long, default_value = "oxirs.toml")]
        output: PathBuf,
    },
    /// Validate a configuration file
    Validate {
        /// Configuration file path
        config: PathBuf,
    },
    /// Show current configuration
    Show {
        /// Configuration file path
        config: Option<PathBuf>,
    },
}

/// SPARQL query template actions
#[derive(Subcommand)]
pub enum TemplateAction {
    /// List all available templates
    List {
        /// Filter by category (basic, advanced, analytics, graph, federation, paths, aggregation)
        #[arg(long)]
        category: Option<String>,
    },
    /// Show template details
    Show {
        /// Template name
        name: String,
    },
    /// Render a template with parameters
    Render {
        /// Template name
        name: String,
        /// Template parameters in key=value format (repeatable)
        #[arg(short, long)]
        param: Vec<String>,
    },
}

/// Query history actions
#[derive(Subcommand)]
pub enum HistoryAction {
    /// List query history
    List {
        /// Maximum number of entries to show
        #[arg(short, long, default_value = "20")]
        limit: Option<usize>,
        /// Filter by dataset
        #[arg(short, long)]
        dataset: Option<String>,
    },
    /// Show full query details
    Show {
        /// History entry ID
        id: usize,
    },
    /// Replay a query from history
    Replay {
        /// History entry ID
        id: usize,
        /// Output format
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Search query history
    Search {
        /// Query text to search for
        query: String,
    },
    /// Clear query history
    Clear,
    /// Show history statistics
    Stats,
}

/// SAMM Aspect Model actions (Java ESMF SDK compatible)
#[derive(Subcommand)]
pub enum AspectAction {
    /// Validate a SAMM Aspect model
    Validate {
        /// Aspect model file (Turtle format)
        file: PathBuf,
        /// Show detailed validation output
        #[arg(short, long)]
        detailed: bool,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Pretty-print an Aspect model
    Prettyprint {
        /// Aspect model file (Turtle format)
        file: PathBuf,
        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format (turtle, rdfxml, jsonld)
        #[arg(short, long, default_value = "turtle")]
        format: String,
        /// Include comments
        #[arg(long)]
        comments: bool,
    },
    /// Generate artifacts from Aspect model
    To {
        /// Aspect model file (Turtle format)
        file: PathBuf,
        /// Target format (rust, python, java, scala, typescript, graphql, markdown, html,
        /// jsonschema, openapi, asyncapi, jsonld, payload, aas, sql, diagram)
        format: String,
        /// Output file or directory
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Include examples in output
        #[arg(long)]
        examples: bool,
        /// Format variant (for aas: xml/json/aasx, for sql: postgresql/mysql/sqlite,
        /// for diagram: dot/svg/png)
        #[arg(short = 'f', long = "format")]
        format_variant: Option<String>,
    },
    /// Edit Aspect model (move elements or create new version)
    Edit {
        #[command(subcommand)]
        action: EditAction,
    },
    /// Show where model elements are used
    Usage {
        /// Aspect model file or URN
        input: String,
        /// Models root directory (required when using URN)
        #[arg(long = "models-root")]
        models_root: Option<PathBuf>,
    },
}

/// Edit actions for Aspect models (Java ESMF SDK compatible)
#[derive(Subcommand)]
pub enum EditAction {
    /// Move element to different namespace
    Move {
        /// Aspect model file (Turtle format)
        file: PathBuf,
        /// Element URN to move
        element: String,
        /// Target namespace (optional)
        namespace: Option<String>,
        /// Don't write changes, only show report
        #[arg(long)]
        dry_run: bool,
        /// Include detailed content changes (with --dry-run)
        #[arg(long)]
        details: bool,
        /// Overwrite existing files
        #[arg(long)]
        force: bool,
        /// Copy file header from source
        #[arg(long)]
        copy_file_header: bool,
    },
    /// Create new version of Aspect model
    Newversion {
        /// Aspect model file (Turtle format)
        file: PathBuf,
        /// Update major version
        #[arg(long, conflicts_with_all = ["minor", "micro"])]
        major: bool,
        /// Update minor version
        #[arg(long, conflicts_with_all = ["major", "micro"])]
        minor: bool,
        /// Update micro version
        #[arg(long, conflicts_with_all = ["major", "minor"])]
        micro: bool,
        /// Don't write changes, only show report
        #[arg(long)]
        dry_run: bool,
        /// Include detailed content changes (with --dry-run)
        #[arg(long)]
        details: bool,
        /// Overwrite existing files
        #[arg(long)]
        force: bool,
    },
}

/// Asset Administration Shell (AAS) actions (Java ESMF SDK compatible)
#[derive(Subcommand)]
pub enum AasAction {
    /// Convert AAS Submodel Templates to Aspect Models
    ToAspect {
        /// AAS file (XML, JSON, or AASX format)
        file: PathBuf,
        /// Output directory for generated Aspect Models
        #[arg(short = 'd', long = "output-directory")]
        output_directory: Option<PathBuf>,
        /// Select specific submodel template(s) to convert (repeatable)
        #[arg(short = 's', long = "submodel-template")]
        submodel_templates: Vec<usize>,
    },
    /// List submodel templates in AAS file
    List {
        /// AAS file (XML, JSON, or AASX format)
        file: PathBuf,
    },
}

/// Package management actions (Java ESMF SDK compatible)
#[derive(Subcommand)]
pub enum PackageAction {
    /// Import namespace package (ZIP)
    Import {
        /// Namespace package ZIP file
        file: PathBuf,
        /// Directory to import into (required)
        #[arg(long = "models-root", required = true)]
        models_root: PathBuf,
        /// Don't write changes, print report only
        #[arg(long)]
        dry_run: bool,
        /// Include details about model content changes (with --dry-run)
        #[arg(long)]
        details: bool,
        /// Overwrite existing files
        #[arg(long)]
        force: bool,
    },
    /// Export Aspect Model or namespace as ZIP package
    Export {
        /// Aspect Model file or namespace URN
        input: String,
        /// Output ZIP file path (required)
        #[arg(short = 'o', long = "output", required = true)]
        output: PathBuf,
        /// Namespace version filter (for URN exports)
        #[arg(long)]
        version: Option<String>,
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
        } => commands::import::run(dataset, file, format, graph)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Export {
            dataset,
            file,
            format,
            graph,
        } => commands::export::run(dataset, file, format, graph)
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
        Commands::Benchmark {
            dataset,
            suite,
            iterations,
            output,
        } => commands::benchmark::run(dataset, suite, iterations, output)
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        Commands::Migrate {
            source,
            target,
            from,
            to,
        } => commands::migrate::run(source, target, from, to)
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
        } => tools::tdbbackup::run(source, target, compress, incremental).await,
        Commands::TdbCompact {
            location,
            delete_old,
        } => tools::tdbcompact::run(location, delete_old).await,

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
            commands::explain::explain_query(dataset, query, file, analysis_mode)
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
        },
    }
}
