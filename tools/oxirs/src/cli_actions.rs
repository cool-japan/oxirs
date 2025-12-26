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
    /// Convert DTDL to SAMM Aspect model
    From {
        /// DTDL Interface file (JSON format)
        file: PathBuf,
        /// Output file (Turtle format)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format (ttl, json, xml)
        #[arg(short, long, default_value = "ttl")]
        format: String,
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

// === Phase D: Industrial Connectivity CLI Actions (0.1.0-rc.1) ===

/// Time-series database operations
#[derive(Subcommand)]
pub enum TsdbAction {
    /// Query time-series data with SPARQL temporal extensions
    Query {
        /// Dataset name
        dataset: String,
        /// Series ID to query
        #[arg(short = 'S', long)]
        series: Option<u64>,
        /// Start time (ISO 8601 format)
        #[arg(long)]
        start: Option<String>,
        /// End time (ISO 8601 format)
        #[arg(long)]
        end: Option<String>,
        /// SPARQL query with temporal functions (ts:window, ts:resample, ts:interpolate)
        #[arg(long)]
        sparql: Option<String>,
        /// Aggregation function (avg, min, max, sum, count)
        #[arg(short, long)]
        aggregate: Option<String>,
        /// Output format (table, json, csv)
        #[arg(short = 'f', long, default_value = "table")]
        format: String,
    },
    /// Insert time-series data points
    Insert {
        /// Dataset name
        dataset: String,
        /// Series ID
        #[arg(short, long)]
        series: u64,
        /// Timestamp (ISO 8601 format, default: now)
        #[arg(short, long)]
        timestamp: Option<String>,
        /// Value to insert
        #[arg(short, long)]
        value: f64,
        /// Batch insert from CSV file (columns: timestamp,value)
        #[arg(long)]
        from_csv: Option<PathBuf>,
    },
    /// Show compression statistics
    Stats {
        /// Dataset name
        dataset: String,
        /// Series ID (all series if omitted)
        #[arg(short, long)]
        series: Option<u64>,
        /// Show detailed statistics
        #[arg(long)]
        detailed: bool,
    },
    /// Compact time-series storage
    Compact {
        /// Dataset name
        dataset: String,
        /// Series ID (all series if omitted)
        #[arg(short, long)]
        series: Option<u64>,
        /// Force compaction even if not needed
        #[arg(long)]
        force: bool,
    },
    /// Manage retention policies
    Retention {
        #[command(subcommand)]
        action: RetentionAction,
    },
    /// Export time-series to CSV or Parquet
    Export {
        /// Dataset name
        dataset: String,
        /// Series ID
        #[arg(short, long)]
        series: u64,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Export format (csv, parquet)
        #[arg(short, long, default_value = "csv")]
        format: String,
        /// Start time (ISO 8601 format)
        #[arg(long)]
        start: Option<String>,
        /// End time (ISO 8601 format)
        #[arg(long)]
        end: Option<String>,
    },
    /// Benchmark time-series performance
    Benchmark {
        /// Dataset name
        dataset: String,
        /// Number of data points to write
        #[arg(long, default_value = "100000")]
        points: usize,
        /// Number of series to create
        #[arg(long, default_value = "1")]
        series_count: usize,
    },
}

/// Retention policy management
#[derive(Subcommand)]
pub enum RetentionAction {
    /// List retention policies
    List {
        /// Dataset name
        dataset: String,
    },
    /// Add retention policy
    Add {
        /// Dataset name
        dataset: String,
        /// Policy name
        #[arg(short, long)]
        name: String,
        /// Retention duration (e.g., 7d, 90d, 1y)
        #[arg(short, long)]
        duration: String,
        /// Downsampling resolution (e.g., 1m, 1h, 1d)
        #[arg(long)]
        downsample: Option<String>,
        /// Downsampling aggregation (avg, min, max, sum, first, last)
        #[arg(long, default_value = "avg")]
        aggregation: String,
    },
    /// Remove retention policy
    Remove {
        /// Dataset name
        dataset: String,
        /// Policy name
        #[arg(short, long)]
        name: String,
    },
    /// Run retention enforcement manually
    Enforce {
        /// Dataset name
        dataset: String,
        /// Dry run (show what would be deleted)
        #[arg(long)]
        dry_run: bool,
    },
}

/// Modbus protocol operations
#[derive(Subcommand)]
pub enum ModbusAction {
    /// Monitor Modbus TCP device
    MonitorTcp {
        /// Device IP address and port (e.g., 192.168.1.100:502)
        #[arg(short, long)]
        address: String,
        /// Modbus unit ID
        #[arg(short, long, default_value = "1")]
        unit_id: u8,
        /// Register start address
        #[arg(long)]
        start: u16,
        /// Number of registers to read
        #[arg(long, default_value = "10")]
        count: u16,
        /// Polling interval in milliseconds
        #[arg(long, default_value = "1000")]
        interval: u64,
        /// Output format (table, json, csv)
        #[arg(short, long, default_value = "table")]
        format: String,
        /// Output to file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Monitor Modbus RTU device (serial)
    MonitorRtu {
        /// Serial port (e.g., /dev/ttyUSB0, COM3)
        #[arg(short, long)]
        port: String,
        /// Baud rate
        #[arg(short, long, default_value = "9600")]
        baud: u32,
        /// Modbus unit ID
        #[arg(short, long, default_value = "1")]
        unit_id: u8,
        /// Register start address
        #[arg(long)]
        start: u16,
        /// Number of registers to read
        #[arg(long, default_value = "10")]
        count: u16,
        /// Polling interval in milliseconds
        #[arg(long, default_value = "1000")]
        interval: u64,
    },
    /// Read Modbus registers
    Read {
        /// Device configuration (TCP address or RTU port)
        #[arg(short, long)]
        device: String,
        /// Register type (holding, input, coil, discrete)
        #[arg(short = 't', long, default_value = "holding")]
        register_type: String,
        /// Register start address
        #[arg(long)]
        address: u16,
        /// Number of registers to read
        #[arg(long, default_value = "1")]
        count: u16,
        /// Data type interpretation (int16, uint16, int32, uint32, float32, bit)
        #[arg(long)]
        datatype: Option<String>,
    },
    /// Write Modbus registers
    Write {
        /// Device configuration (TCP address or RTU port)
        #[arg(short, long)]
        device: String,
        /// Register address
        #[arg(long)]
        address: u16,
        /// Value to write
        #[arg(long)]
        value: String,
        /// Data type (int16, uint16, int32, uint32, float32)
        #[arg(long, default_value = "uint16")]
        datatype: String,
    },
    /// Generate RDF triples from Modbus data
    ToRdf {
        /// Device configuration (TCP address or RTU port)
        #[arg(short, long)]
        device: String,
        /// Register mapping configuration file (TOML)
        #[arg(short, long)]
        config: PathBuf,
        /// Output RDF file
        #[arg(short, long)]
        output: PathBuf,
        /// RDF format (turtle, ntriples, jsonld)
        #[arg(short, long, default_value = "turtle")]
        format: String,
        /// Number of readings to collect
        #[arg(short, long, default_value = "1")]
        count: usize,
    },
    /// Start Modbus mock server for testing
    MockServer {
        /// Server port
        #[arg(short, long, default_value = "5020")]
        port: u16,
        /// Mock data configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
}

/// CANbus protocol operations
#[derive(Subcommand)]
pub enum CanbusAction {
    /// Monitor CAN interface
    Monitor {
        /// CAN interface name (e.g., can0, vcan0)
        #[arg(short, long)]
        interface: String,
        /// Filter by CAN ID (decimal or hex with 0x prefix)
        #[arg(long)]
        filter: Option<String>,
        /// DBC file for signal decoding
        #[arg(long)]
        dbc: Option<PathBuf>,
        /// Output format (table, json, csv)
        #[arg(short, long, default_value = "table")]
        format: String,
        /// Output to file instead of stdout
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Show only J1939 messages
        #[arg(long)]
        j1939: bool,
    },
    /// Parse DBC file
    ParseDbc {
        /// DBC file path (Vector CANdb++ format)
        #[arg(short = 'd', long)]
        file: PathBuf,
        /// Output format (json, yaml, table)
        #[arg(short = 'f', long, default_value = "table")]
        format: String,
        /// Show detailed signal information
        #[arg(long)]
        detailed: bool,
    },
    /// Decode CAN frame using DBC
    Decode {
        /// CAN ID (decimal or hex with 0x prefix)
        #[arg(long)]
        id: String,
        /// CAN data (hex bytes, e.g., DEADBEEF)
        #[arg(long)]
        data: String,
        /// DBC file for decoding
        #[arg(long)]
        dbc: PathBuf,
        /// Output format (table, json)
        #[arg(short, long, default_value = "table")]
        format: String,
    },
    /// Send CAN frame
    Send {
        /// CAN interface name (e.g., can0, vcan0)
        #[arg(short, long)]
        interface: String,
        /// CAN ID (decimal or hex with 0x prefix)
        #[arg(long)]
        id: String,
        /// CAN data (hex bytes, e.g., DEADBEEF)
        #[arg(long)]
        data: String,
        /// Extended frame (29-bit ID)
        #[arg(long)]
        extended: bool,
    },
    /// Generate SAMM Aspect Model from DBC
    ToSamm {
        /// DBC file path
        #[arg(short, long)]
        dbc: PathBuf,
        /// Output directory for Aspect Models
        #[arg(short, long)]
        output: PathBuf,
        /// Base namespace URI
        #[arg(long, default_value = "urn:samm:org.example.can")]
        namespace: String,
        /// Generate separate Aspect per message
        #[arg(long)]
        per_message: bool,
    },
    /// Generate RDF triples from CAN data
    ToRdf {
        /// CAN interface name
        #[arg(short, long)]
        interface: String,
        /// DBC file for signal mapping
        #[arg(long)]
        dbc: PathBuf,
        /// Output RDF file
        #[arg(short, long)]
        output: PathBuf,
        /// RDF format (turtle, ntriples, jsonld)
        #[arg(short, long, default_value = "turtle")]
        format: String,
        /// Number of frames to collect
        #[arg(short, long, default_value = "100")]
        count: usize,
    },
    /// Replay CAN log file
    Replay {
        /// CAN log file (candump format)
        #[arg(short, long)]
        file: PathBuf,
        /// CAN interface to replay on
        #[arg(short, long)]
        interface: String,
        /// Playback speed multiplier
        #[arg(long, default_value = "1.0")]
        speed: f64,
        /// Loop playback
        #[arg(long)]
        r#loop: bool,
    },
}
