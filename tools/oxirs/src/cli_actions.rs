//! CLI action enums for various oxirs commands
//!
//! This module contains all the action enum definitions used by the main Commands enum.
//! These were extracted from lib.rs to keep the main module under 2000 lines.

use clap::Subcommand;
use std::path::PathBuf;

/// CI/CD integration actions
#[derive(Subcommand)]
pub enum CicdAction {
    /// Generate test report from benchmark results
    Report {
        /// Input benchmark results file (JSON)
        input: PathBuf,
        /// Output report file
        #[arg(short, long)]
        output: PathBuf,
        /// Report format (junit, tap, json)
        #[arg(short, long, default_value = "junit")]
        format: String,
    },
    /// Generate Docker integration files
    Docker {
        /// Output directory for Docker files
        #[arg(short, long, default_value = ".")]
        output: PathBuf,
    },
    /// Generate GitHub Actions workflow
    Github {
        /// Output file path
        #[arg(short, long, default_value = ".github/workflows/ci.yml")]
        output: PathBuf,
    },
    /// Generate GitLab CI configuration
    Gitlab {
        /// Output file path
        #[arg(short, long, default_value = ".gitlab-ci.yml")]
        output: PathBuf,
    },
}

/// Cache management actions
#[derive(Subcommand)]
pub enum CacheAction {
    /// Show cache statistics
    Stats,
    /// Clear the query cache
    Clear,
    /// Configure cache settings
    Config {
        /// TTL in seconds
        #[arg(long)]
        ttl: Option<u64>,
        /// Maximum cache size
        #[arg(long)]
        max_size: Option<usize>,
    },
}

/// Alias management actions
#[derive(Subcommand)]
pub enum AliasAction {
    /// List all aliases
    List,
    /// Show a specific alias
    Show {
        /// Alias name
        name: String,
    },
    /// Add or update an alias
    Add {
        /// Alias name
        name: String,
        /// Command to alias
        command: String,
    },
    /// Remove an alias
    Remove {
        /// Alias name
        name: String,
    },
    /// Reset aliases to defaults
    Reset,
}

/// Index management actions
#[derive(Subcommand)]
pub enum IndexAction {
    /// List all indexes in a dataset
    List {
        /// Dataset name or path
        dataset: String,
    },
    /// Rebuild indexes for better performance
    Rebuild {
        /// Dataset name or path
        dataset: String,
        /// Specific index name to rebuild (omit to rebuild all)
        #[arg(long)]
        index: Option<String>,
    },
    /// Show detailed index statistics
    Stats {
        /// Dataset name or path
        dataset: String,
        /// Output format (text, json, csv)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Optimize indexes to reduce fragmentation
    Optimize {
        /// Dataset name or path
        dataset: String,
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
    /// Show comprehensive query analytics
    Analytics {
        /// Filter by dataset
        #[arg(short, long)]
        dataset: Option<String>,
    },
}

/// Migration actions for converting between databases and formats
#[derive(Subcommand)]
pub enum MigrateAction {
    /// Convert RDF data between formats (turtle, ntriples, etc.)
    Format {
        /// Source file path
        source: PathBuf,
        /// Target file path
        target: PathBuf,
        /// Source format
        #[arg(long)]
        from: String,
        /// Target format
        #[arg(long)]
        to: String,
    },
    /// Migrate from Apache Jena TDB1 database to OxiRS
    FromTdb1 {
        /// TDB1 database directory
        tdb_dir: PathBuf,
        /// Target OxiRS dataset name
        dataset: String,
        /// Skip validation (faster but less safe)
        #[arg(long)]
        skip_validation: bool,
    },
    /// Migrate from Apache Jena TDB2 database to OxiRS
    FromTdb2 {
        /// TDB2 database directory
        tdb_dir: PathBuf,
        /// Target OxiRS dataset name
        dataset: String,
        /// Skip validation (faster but less safe)
        #[arg(long)]
        skip_validation: bool,
    },
    /// Migrate from Virtuoso database to OxiRS
    FromVirtuoso {
        /// Virtuoso connection string
        connection: String,
        /// Target OxiRS dataset name
        dataset: String,
        /// Graph URIs to migrate (comma-separated, or 'all')
        #[arg(long, default_value = "all")]
        graphs: String,
    },
    /// Migrate from RDF4J repository to OxiRS
    FromRdf4j {
        /// RDF4J repository directory
        repo_dir: PathBuf,
        /// Target OxiRS dataset name
        dataset: String,
    },
    /// Migrate from Blazegraph database to OxiRS
    FromBlazegraph {
        /// Blazegraph SPARQL endpoint URL
        endpoint: String,
        /// Target OxiRS dataset name
        dataset: String,
        /// Namespace to migrate
        #[arg(long, default_value = "kb")]
        namespace: String,
    },
    /// Migrate from Ontotext GraphDB to OxiRS
    FromGraphdb {
        /// GraphDB SPARQL endpoint URL
        endpoint: String,
        /// Target OxiRS dataset name
        dataset: String,
        /// Repository name
        #[arg(long)]
        repository: String,
    },
}

/// Point-in-Time Recovery (PITR) actions
#[derive(Subcommand)]
pub enum PitrAction {
    /// Initialize transaction logging for a dataset
    Init {
        /// Dataset directory
        dataset: PathBuf,
        /// Maximum log file size in MB
        #[arg(long, default_value = "100")]
        max_log_size: u64,
        /// Enable auto-archival of old logs
        #[arg(long)]
        auto_archive: bool,
    },
    /// Create a named checkpoint
    Checkpoint {
        /// Dataset directory
        dataset: PathBuf,
        /// Checkpoint name
        name: String,
    },
    /// List available checkpoints
    List {
        /// Dataset directory
        dataset: PathBuf,
        /// Output format (text, json)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    /// Recover to a specific point in time
    RecoverTimestamp {
        /// Dataset directory
        dataset: PathBuf,
        /// Target timestamp (ISO 8601 format: 2024-01-01T12:00:00Z)
        timestamp: String,
        /// Output directory for recovered data
        output: PathBuf,
    },
    /// Recover to a specific transaction ID
    RecoverTransaction {
        /// Dataset directory
        dataset: PathBuf,
        /// Target transaction ID
        transaction_id: u64,
        /// Output directory for recovered data
        output: PathBuf,
    },
    /// Archive transaction logs
    Archive {
        /// Dataset directory
        dataset: PathBuf,
    },
}

/// Benchmark actions for performance testing and dataset generation
#[derive(Subcommand)]
pub enum BenchmarkAction {
    /// Run benchmark suite on a dataset
    Run {
        /// Target dataset (alphanumeric, _, - only; no dots or extensions)
        dataset: String,
        /// Benchmark suite (sp2bench, watdiv, ldbc, bsbm, custom)
        #[arg(short, long, default_value = "sp2bench")]
        suite: String,
        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,
        /// Output report file
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Enable detailed timing information
        #[arg(long)]
        detailed: bool,
        /// Warmup iterations before benchmarking
        #[arg(long, default_value = "3")]
        warmup: usize,
    },
    /// Generate synthetic benchmark datasets
    Generate {
        /// Output dataset path
        output: PathBuf,
        /// Dataset size (tiny, small, medium, large, xlarge)
        #[arg(short, long, default_value = "small")]
        size: String,
        /// Dataset type (rdf, graph, semantic)
        #[arg(short = 't', long, default_value = "rdf")]
        dataset_type: String,
        /// Random seed for reproducibility
        #[arg(long)]
        seed: Option<u64>,
        /// Number of triples to generate
        #[arg(long)]
        triples: Option<usize>,
        /// Schema file for constrained generation
        #[arg(long)]
        schema: Option<PathBuf>,
    },
    /// Analyze query workload from log files
    Analyze {
        /// Query log file or dataset
        input: PathBuf,
        /// Output analysis report
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Report format (text, json, html)
        #[arg(short, long, default_value = "text")]
        format: String,
        /// Include query optimization suggestions
        #[arg(long)]
        suggestions: bool,
        /// Analyze patterns and frequencies
        #[arg(long)]
        patterns: bool,
    },
    /// Compare benchmark results for regression detection
    Compare {
        /// Baseline benchmark results file
        baseline: PathBuf,
        /// Current benchmark results file
        current: PathBuf,
        /// Output comparison report
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Regression threshold percentage
        #[arg(long, default_value = "10.0")]
        threshold: f64,
        /// Report format (text, json, html)
        #[arg(short, long, default_value = "text")]
        format: String,
    },
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
