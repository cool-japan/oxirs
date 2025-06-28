//! # Oxide - OxiRS CLI Tool
//!
//! Command-line interface for OxiRS providing import, export, migration,
//! benchmarking, and server management tools.
//!
//! ## Commands
//!
//! - `init`: Initialize a new knowledge graph dataset
//! - `serve`: Start the OxiRS server
//! - `import`: Import RDF data from various formats
//! - `export`: Export RDF data to various formats
//! - `query`: Execute SPARQL queries
//! - `update`: Execute SPARQL updates
//! - `benchmark`: Run performance benchmarks
//! - `migrate`: Migrate data between formats/versions
//! - `config`: Manage server configuration
//!
//! ## Examples
//!
//! ```bash
//! # Initialize a new dataset
//! oxide init mykg --format tdb2
//!
//! # Import data
//! oxide import mykg data.ttl --format turtle
//!
//! # Start server
//! oxide serve mykg.toml --port 3030
//!
//! # Run benchmarks
//! oxide benchmark mykg --suite sp2bench
//! ```

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

/// Oxide CLI application
#[derive(Parser)]
#[command(name = "oxide")]
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
        /// Target dataset
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
        /// Source dataset
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
        /// Target dataset
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
        /// Target dataset
        dataset: String,
        /// SPARQL update string or file
        update: String,
        /// Update is a file path
        #[arg(short, long)]
        file: bool,
    },
    /// Run performance benchmarks
    Benchmark {
        /// Target dataset
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
    let log_format = if std::env::var("OXIDE_LOG_FORMAT").as_deref() == Ok("json") {
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
            std::env::var("OXIDE_LOG_LEVEL").unwrap_or_else(|_| "info".to_string())
        },
        format: log_format,
        timestamps: !ctx.quiet,
        source_location: ctx.verbose,
        thread_ids: false,
        perf_threshold_ms: std::env::var("OXIDE_PERF_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok()),
        file: std::env::var("OXIDE_LOG_FILE").ok(),
    };

    cli::init_logging(&log_config).expect("Failed to initialize logging");

    // Show startup message if not quiet
    if ctx.should_show_output() {
        ctx.info(&format!("Oxide CLI v{}", env!("CARGO_PKG_VERSION")));
    }

    match cli.command {
        Commands::Init {
            name,
            format,
            location,
        } => commands::init::run(name, format, location).await,
        Commands::Serve {
            config,
            port,
            host,
            graphql,
        } => commands::serve::run(config, port, host, graphql).await,
        Commands::Import {
            dataset,
            file,
            format,
            graph,
        } => commands::import::run(dataset, file, format, graph).await,
        Commands::Export {
            dataset,
            file,
            format,
            graph,
        } => commands::export::run(dataset, file, format, graph).await,
        Commands::Query {
            dataset,
            query,
            file,
            output,
        } => commands::query::run(dataset, query, file, output).await,
        Commands::Update {
            dataset,
            update,
            file,
        } => commands::update::run(dataset, update, file).await,
        Commands::Benchmark {
            dataset,
            suite,
            iterations,
            output,
        } => commands::benchmark::run(dataset, suite, iterations, output).await,
        Commands::Migrate {
            source,
            target,
            from,
            to,
        } => commands::migrate::run(source, target, from, to).await,
        Commands::Config { action } => commands::config::run(action).await,

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
            dataset: _,
            history: _,
        } => {
            use cli::InteractiveMode;

            ctx.info("Starting interactive mode...");
            let mut interactive = InteractiveMode::new()?;
            interactive.run().await
        }
    }
}
