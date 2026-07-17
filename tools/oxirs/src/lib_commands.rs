//! CLI command and subcommand definitions for OxiRS.
//!
//! Contains `Cli` (the root parser) and `Commands` (all top-level subcommands).

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::cli_actions::*;

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
        /// Maximum input file size in bytes (0 = unlimited)
        #[arg(long, default_value_t = 10 * 1024 * 1024 * 1024)]
        max_file_size: u64,
        /// Storage backend: "memory" (in-RAM N-Quads log, default) or
        /// "tdb2" (on-disk oxirs-tdb store; keeps RAM use bounded for large
        /// bulk loads)
        #[arg(long, alias = "backend", default_value = "memory")]
        dataset_type: String,
    },
    /// Batch import multiple RDF files into a dataset in parallel
    Batch {
        /// Target dataset (alphanumeric, _, - only; no dots or extensions)
        dataset: String,
        /// Input file paths
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Input format (turtle, ntriples, rdfxml, jsonld) - auto-detected per file if omitted
        #[arg(short, long)]
        format: Option<String>,
        /// Named graph URI
        #[arg(short, long)]
        graph: Option<String>,
        /// Number of files to process concurrently
        #[arg(short, long, default_value = "4")]
        parallel: usize,
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
        /// Results format (json, xml, csv, tsv, text)
        #[arg(long, default_value = "text")]
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
        #[arg(short = 'n', long, default_value = "1")]
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
        action: crate::commands::performance::PerformanceCommand,
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
    Rebac(crate::commands::rebac::RebacArgs),

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

    // === Phase D: Industrial Connectivity ===
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

    /// SPARQL query profiler
    Profile {
        #[command(subcommand)]
        action: ProfilerAction,
    },

    /// LRU result cache management
    ResultCache {
        #[command(subcommand)]
        action: ResultCacheAction,
    },

    /// Streaming SPARQL query results
    Stream {
        #[command(subcommand)]
        action: StreamAction,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Cli`'s derived clap parser recurses through an unusually large
    /// `Commands` enum (50+ variants, several with nested subcommand enums
    /// of their own); in an unoptimized (debug/test) build this can exceed
    /// the default 2-8 MiB thread stack. Run parsing on a thread with a
    /// generous stack instead of fighting the default test-harness stack.
    fn parse_cli_with_large_stack(args: &'static [&'static str]) -> Cli {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || Cli::parse_from(args))
            .expect("failed to spawn CLI-parsing thread")
            .join()
            .expect("CLI-parsing thread panicked")
    }

    /// Regression test: the fully-implemented parallel `batch` import
    /// command must be reachable from the CLI (previously there was no
    /// `Commands::Batch` variant at all, so `oxirs batch ...` could not be
    /// parsed despite `commands::batch::import_batch` being fully wired).
    #[test]
    fn test_batch_command_is_reachable_from_cli() {
        let cli = parse_cli_with_large_stack(&[
            "oxirs",
            "batch",
            "mydataset",
            "a.ttl",
            "b.ttl",
            "--parallel",
            "2",
        ]);
        match cli.command {
            Commands::Batch {
                dataset,
                files,
                parallel,
                ..
            } => {
                assert_eq!(dataset, "mydataset");
                assert_eq!(files.len(), 2);
                assert_eq!(parallel, 2);
            }
            _ => panic!("expected Commands::Batch, got a different variant"),
        }
    }

    /// Regression test: `oxirs import` must accept a configurable
    /// `--max-file-size` (0 = unlimited) instead of a hardcoded 1 GiB cap
    /// with no CLI override.
    #[test]
    fn test_import_max_file_size_flag_is_configurable() {
        let cli = parse_cli_with_large_stack(&[
            "oxirs",
            "import",
            "mydataset",
            "data.ttl",
            "--max-file-size",
            "0",
        ]);
        match cli.command {
            Commands::Import { max_file_size, .. } => {
                assert_eq!(max_file_size, 0, "0 must be accepted to mean unlimited");
            }
            _ => panic!("expected Commands::Import, got a different variant"),
        }

        // Default must be raised well above the old hardcoded 1 GiB cap.
        let cli = parse_cli_with_large_stack(&["oxirs", "import", "mydataset", "data.ttl"]);
        match cli.command {
            Commands::Import { max_file_size, .. } => {
                assert!(
                    max_file_size > 1_073_741_824,
                    "default max_file_size should exceed the old 1 GiB hard cap"
                );
            }
            _ => panic!("expected Commands::Import, got a different variant"),
        }
    }
}
