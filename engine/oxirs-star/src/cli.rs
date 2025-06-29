//! Command-line interface tools for RDF-star validation and debugging.
//!
//! This module provides comprehensive CLI utilities for working with RDF-star data,
//! including validation, conversion, analysis, and debugging tools.

use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Arg, ArgMatches, Command};
use serde_json;
use tracing::{debug, error, info, warn};

use crate::model::{StarGraph, StarQuad, StarTerm, StarTriple};
use crate::parser::{ParseError, StarFormat, StarParser};
use crate::serializer::{SerializationOptions, StarSerializer};
use crate::store::StarStore;
use crate::troubleshooting::{
    DiagnosticAnalyzer, MigrationAssistant, MigrationSourceFormat, TroubleshootingGuide,
};
use crate::{StarConfig, StarError, StarResult};

/// CLI application for RDF-star tools
pub struct StarCli {
    config: StarConfig,
    verbose: bool,
    quiet: bool,
}

impl StarCli {
    /// Create a new CLI application
    pub fn new() -> Self {
        Self {
            config: StarConfig::default(),
            verbose: false,
            quiet: false,
        }
    }

    /// Run the CLI application with command-line arguments
    pub fn run(&mut self, args: Vec<String>) -> Result<()> {
        let app = self.build_cli();
        let matches = app.try_get_matches_from(args)?;

        self.verbose = matches.get_flag("verbose");
        self.quiet = matches.get_flag("quiet");

        self.setup_logging();

        match matches.subcommand() {
            Some(("validate", sub_matches)) => self.validate_command(sub_matches),
            Some(("convert", sub_matches)) => self.convert_command(sub_matches),
            Some(("analyze", sub_matches)) => self.analyze_command(sub_matches),
            Some(("debug", sub_matches)) => self.debug_command(sub_matches),
            Some(("benchmark", sub_matches)) => self.benchmark_command(sub_matches),
            Some(("query", sub_matches)) => self.query_command(sub_matches),
            Some(("troubleshoot", sub_matches)) => self.troubleshoot_command(sub_matches),
            Some(("migrate", sub_matches)) => self.migrate_command(sub_matches),
            Some(("doctor", sub_matches)) => self.doctor_command(sub_matches),
            _ => {
                eprintln!("No command specified. Use --help for usage information.");
                std::process::exit(1);
            }
        }
    }

    /// Build the CLI application structure
    fn build_cli(&self) -> Command {
        Command::new("oxirs-star")
            .version("1.0.0")
            .about("RDF-star validation, conversion, and debugging tools")
            .author("OxiRS Team")
            .arg(
                Arg::new("verbose")
                    .short('v')
                    .long("verbose")
                    .help("Enable verbose output")
                    .action(clap::ArgAction::SetTrue),
            )
            .arg(
                Arg::new("quiet")
                    .short('q')
                    .long("quiet")
                    .help("Suppress all output except errors")
                    .action(clap::ArgAction::SetTrue),
            )
            .subcommand(
                Command::new("validate")
                    .about("Validate RDF-star files")
                    .arg(
                        Arg::new("input")
                            .help("Input file path")
                            .required(true)
                            .value_name("FILE"),
                    )
                    .arg(
                        Arg::new("format")
                            .short('f')
                            .long("format")
                            .help("Input format (auto-detect if not specified)")
                            .value_name("FORMAT"),
                    )
                    .arg(
                        Arg::new("strict")
                            .long("strict")
                            .help("Use strict validation mode")
                            .action(clap::ArgAction::SetTrue),
                    )
                    .arg(
                        Arg::new("report")
                            .short('r')
                            .long("report")
                            .help("Output detailed validation report")
                            .value_name("OUTPUT_FILE"),
                    ),
            )
            .subcommand(
                Command::new("convert")
                    .about("Convert between RDF-star formats")
                    .arg(
                        Arg::new("input")
                            .help("Input file path")
                            .required(true)
                            .value_name("INPUT_FILE"),
                    )
                    .arg(
                        Arg::new("output")
                            .help("Output file path")
                            .required(true)
                            .value_name("OUTPUT_FILE"),
                    )
                    .arg(
                        Arg::new("from")
                            .long("from")
                            .help("Input format")
                            .value_name("FORMAT"),
                    )
                    .arg(
                        Arg::new("to")
                            .long("to")
                            .help("Output format")
                            .required(true)
                            .value_name("FORMAT"),
                    )
                    .arg(
                        Arg::new("pretty")
                            .long("pretty")
                            .help("Enable pretty printing")
                            .action(clap::ArgAction::SetTrue),
                    ),
            )
            .subcommand(
                Command::new("analyze")
                    .about("Analyze RDF-star data structure and statistics")
                    .arg(
                        Arg::new("input")
                            .help("Input file path")
                            .required(true)
                            .value_name("FILE"),
                    )
                    .arg(
                        Arg::new("output")
                            .short('o')
                            .long("output")
                            .help("Output analysis report to file")
                            .value_name("OUTPUT_FILE"),
                    )
                    .arg(
                        Arg::new("json")
                            .long("json")
                            .help("Output analysis in JSON format")
                            .action(clap::ArgAction::SetTrue),
                    ),
            )
            .subcommand(
                Command::new("debug")
                    .about("Debug RDF-star parsing issues")
                    .arg(
                        Arg::new("input")
                            .help("Input file path")
                            .required(true)
                            .value_name("FILE"),
                    )
                    .arg(
                        Arg::new("line")
                            .short('l')
                            .long("line")
                            .help("Focus on specific line number")
                            .value_name("LINE_NUMBER"),
                    )
                    .arg(
                        Arg::new("context")
                            .short('c')
                            .long("context")
                            .help("Number of context lines to show around errors")
                            .value_name("LINES")
                            .default_value("3"),
                    ),
            )
            .subcommand(
                Command::new("benchmark")
                    .about("Benchmark parsing and serialization performance")
                    .arg(
                        Arg::new("input")
                            .help("Input file path")
                            .required(true)
                            .value_name("FILE"),
                    )
                    .arg(
                        Arg::new("iterations")
                            .short('n')
                            .long("iterations")
                            .help("Number of benchmark iterations")
                            .value_name("N")
                            .default_value("10"),
                    )
                    .arg(
                        Arg::new("warmup")
                            .long("warmup")
                            .help("Number of warmup iterations")
                            .value_name("N")
                            .default_value("3"),
                    ),
            )
            .subcommand(
                Command::new("query")
                    .about("Execute SPARQL-star queries")
                    .arg(
                        Arg::new("data")
                            .help("RDF-star data file")
                            .required(true)
                            .value_name("DATA_FILE"),
                    )
                    .arg(
                        Arg::new("query")
                            .help("SPARQL-star query file or inline query")
                            .required(true)
                            .value_name("QUERY"),
                    )
                    .arg(
                        Arg::new("format")
                            .short('f')
                            .long("format")
                            .help("Output format for results")
                            .value_name("FORMAT")
                            .default_value("table"),
                    ),
            )
            .subcommand(
                Command::new("troubleshoot")
                    .about("Get troubleshooting help for specific errors")
                    .arg(
                        Arg::new("error")
                            .help("Error message or type to troubleshoot")
                            .required(true)
                            .value_name("ERROR"),
                    )
                    .arg(
                        Arg::new("output")
                            .short('o')
                            .long("output")
                            .help("Output troubleshooting report to file")
                            .value_name("OUTPUT_FILE"),
                    ),
            )
            .subcommand(
                Command::new("migrate")
                    .about("Migrate data from other RDF stores to RDF-star")
                    .arg(
                        Arg::new("source")
                            .help("Source data file")
                            .required(true)
                            .value_name("SOURCE_FILE"),
                    )
                    .arg(
                        Arg::new("output")
                            .help("Output RDF-star file")
                            .required(true)
                            .value_name("OUTPUT_FILE"),
                    )
                    .arg(
                        Arg::new("source-format")
                            .long("source-format")
                            .help("Source format (standard-rdf, jena, rdflib, etc.)")
                            .value_name("FORMAT")
                            .default_value("standard-rdf"),
                    )
                    .arg(
                        Arg::new("plan")
                            .long("plan")
                            .help("Generate migration plan without executing")
                            .action(clap::ArgAction::SetTrue),
                    ),
            )
            .subcommand(
                Command::new("doctor")
                    .about("Comprehensive diagnostic analysis of RDF-star data")
                    .arg(
                        Arg::new("input")
                            .help("Input file to diagnose")
                            .required(true)
                            .value_name("FILE"),
                    )
                    .arg(
                        Arg::new("report")
                            .short('r')
                            .long("report")
                            .help("Output detailed diagnostic report")
                            .value_name("REPORT_FILE"),
                    )
                    .arg(
                        Arg::new("fix")
                            .long("fix")
                            .help("Attempt to automatically fix issues")
                            .action(clap::ArgAction::SetTrue),
                    ),
            )
    }

    /// Setup logging based on verbosity flags
    fn setup_logging(&self) {
        if self.quiet {
            return;
        }

        let level = if self.verbose {
            tracing::Level::DEBUG
        } else {
            tracing::Level::INFO
        };

        tracing_subscriber::fmt()
            .with_max_level(level)
            .with_target(false)
            .init();
    }

    /// Validate RDF-star files
    fn validate_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches.get_one::<String>("input").unwrap();
        let format = matches.get_one::<String>("format");
        let strict = matches.get_flag("strict");
        let report_path = matches.get_one::<String>("report");

        info!("Validating RDF-star file: {}", input_path);

        let start_time = Instant::now();
        let validation_result = self.validate_file(input_path, format, strict)?;
        let duration = start_time.elapsed();

        if !self.quiet {
            println!("Validation completed in {:?}", duration);
            self.print_validation_result(&validation_result);
        }

        if let Some(report_path) = report_path {
            self.write_validation_report(&validation_result, report_path)?;
        }

        if validation_result.is_valid {
            Ok(())
        } else {
            std::process::exit(1);
        }
    }

    /// Convert between RDF-star formats
    fn convert_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches.get_one::<String>("input").unwrap();
        let output_path = matches.get_one::<String>("output").unwrap();
        let from_format = matches.get_one::<String>("from");
        let to_format = matches.get_one::<String>("to").unwrap();
        let pretty = matches.get_flag("pretty");

        info!("Converting {} to {}", input_path, output_path);

        let start_time = Instant::now();
        self.convert_file(input_path, output_path, from_format, to_format, pretty)?;
        let duration = start_time.elapsed();

        if !self.quiet {
            println!("Conversion completed in {:?}", duration);
        }

        Ok(())
    }

    /// Analyze RDF-star data structure
    fn analyze_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches.get_one::<String>("input").unwrap();
        let output_path = matches.get_one::<String>("output");
        let json_output = matches.get_flag("json");

        info!("Analyzing RDF-star file: {}", input_path);

        let start_time = Instant::now();
        let analysis = self.analyze_file(input_path)?;
        let duration = start_time.elapsed();

        if !self.quiet {
            println!("Analysis completed in {:?}", duration);
        }

        if json_output {
            let json_output = serde_json::to_string_pretty(&analysis)?;
            if let Some(output_path) = output_path {
                fs::write(output_path, json_output)?;
            } else {
                println!("{}", json_output);
            }
        } else {
            if let Some(output_path) = output_path {
                let report = self.format_analysis_report(&analysis);
                fs::write(output_path, report)?;
            } else {
                self.print_analysis_result(&analysis);
            }
        }

        Ok(())
    }

    /// Debug parsing issues
    fn debug_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches.get_one::<String>("input").unwrap();
        let target_line = matches
            .get_one::<String>("line")
            .map(|s| s.parse::<usize>().unwrap_or(0));
        let context_lines: usize = matches
            .get_one::<String>("context")
            .unwrap()
            .parse()
            .unwrap_or(3);

        info!("Debugging RDF-star file: {}", input_path);

        self.debug_file(input_path, target_line, context_lines)?;

        Ok(())
    }

    /// Benchmark performance
    fn benchmark_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches.get_one::<String>("input").unwrap();
        let iterations: usize = matches
            .get_one::<String>("iterations")
            .unwrap()
            .parse()
            .unwrap_or(10);
        let warmup: usize = matches
            .get_one::<String>("warmup")
            .unwrap()
            .parse()
            .unwrap_or(3);

        info!("Benchmarking file: {}", input_path);

        let results = self.benchmark_file(input_path, iterations, warmup)?;
        self.print_benchmark_results(&results);

        Ok(())
    }

    /// Execute SPARQL-star queries
    fn query_command(&self, matches: &ArgMatches) -> Result<()> {
        let data_path = matches.get_one::<String>("data").unwrap();
        let query_input = matches.get_one::<String>("query").unwrap();
        let output_format = matches.get_one::<String>("format").unwrap();

        info!("Executing SPARQL-star query on: {}", data_path);

        self.execute_query(data_path, query_input, output_format)?;

        Ok(())
    }

    /// Validate a single file
    fn validate_file(
        &self,
        path: &str,
        format: Option<&String>,
        strict: bool,
    ) -> Result<ValidationResult> {
        let content =
            fs::read_to_string(path).with_context(|| format!("Failed to read file: {}", path))?;

        let detected_format = if let Some(fmt) = format {
            fmt.parse::<StarFormat>()
                .map_err(|_| anyhow::anyhow!("Invalid format: {}", fmt))?
        } else {
            self.detect_format(path, &content)?
        };

        let mut parser = StarParser::new();
        if strict {
            parser.set_strict_mode(true);
        }

        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut triple_count = 0;
        let mut quoted_triple_count = 0;

        let parse_result = parser.parse_str(&content, detected_format);

        match parse_result {
            Ok(graph) => {
                triple_count = graph.len();
                // Count quoted triples
                for triple in &graph {
                    if self.has_quoted_terms(triple) {
                        quoted_triple_count += 1;
                    }
                }
            }
            Err(e) => {
                errors.push(format!("Parse error: {}", e));
            }
        }

        // Get detailed errors from parser
        let parse_errors = parser.get_errors();
        for error in parse_errors {
            match &error {
                StarError::ParseError { message, line, .. } => {
                    if let Some(line_num) = line {
                        errors.push(format!("Line {}: {}", line_num, message));
                    } else {
                        errors.push(message.clone());
                    }
                }
                _ => {
                    errors.push(error.to_string());
                }
            }
        }

        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            triple_count,
            quoted_triple_count,
            format: detected_format,
        })
    }

    /// Convert between formats
    fn convert_file(
        &self,
        input: &str,
        output: &str,
        from: Option<&String>,
        to: &str,
        pretty: bool,
    ) -> Result<()> {
        let content = fs::read_to_string(input)?;

        let from_format = if let Some(fmt) = from {
            fmt.parse::<StarFormat>()?
        } else {
            self.detect_format(input, &content)?
        };

        let to_format = to.parse::<StarFormat>()?;

        // Parse input
        let mut parser = StarParser::new();
        let graph = parser.parse_str(&content, from_format)?;

        // Serialize output
        let mut serializer = StarSerializer::new();
        let mut options = SerializationOptions::default();
        if pretty {
            options.pretty_print = true;
        }

        let output_content = serializer.serialize_graph(&graph, to_format, &options)?;
        fs::write(output, output_content)?;

        Ok(())
    }

    /// Analyze file structure
    fn analyze_file(&self, path: &str) -> Result<AnalysisResult> {
        let content = fs::read_to_string(path)?;
        let format = self.detect_format(path, &content)?;

        let mut parser = StarParser::new();
        let graph = parser.parse_str(&content, format)?;

        let mut analysis = AnalysisResult {
            format,
            total_triples: graph.len(),
            quoted_triples: 0,
            subjects: std::collections::HashSet::new(),
            predicates: std::collections::HashSet::new(),
            objects: std::collections::HashSet::new(),
            max_nesting_depth: 0,
            namespaces: std::collections::HashSet::new(),
        };

        // Analyze each triple
        for triple in &graph {
            self.analyze_triple(triple, &mut analysis, 0);
        }

        Ok(analysis)
    }

    /// Debug parsing issues in a file
    fn debug_file(
        &self,
        path: &str,
        target_line: Option<usize>,
        context_lines: usize,
    ) -> Result<()> {
        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();

        println!("Debugging file: {}", path);
        println!("Total lines: {}", lines.len());
        println!();

        let format = self.detect_format(path, &content)?;
        println!("Detected format: {:?}", format);
        println!();

        let mut parser = StarParser::new();
        parser.set_error_recovery(true);

        let parse_result = parser.parse_str(&content, format);
        let errors = parser.get_errors();

        if errors.is_empty() {
            println!("✓ No parsing errors found");
        } else {
            println!("✗ Found {} parsing errors:", errors.len());
            println!();

            for (i, error) in errors.iter().enumerate() {
                println!("Error {}:", i + 1);

                match error {
                    StarError::ParseError {
                        message,
                        line,
                        column,
                        ..
                    } => {
                        if let (Some(line), Some(column)) = (line, column) {
                            println!("  Line {}, Column {}: {}", line, column, message);

                            // Show context lines
                            let start_line = line.saturating_sub(context_lines + 1);
                            let end_line = (*line + context_lines).min(lines.len());

                            println!("  Context lines:");
                            for line_num in start_line..end_line {
                                let marker = if line_num + 1 == *line { ">>>" } else { "   " };
                                println!(
                                    "  {} {:4}: {}",
                                    marker,
                                    line_num + 1,
                                    lines.get(line_num).unwrap_or(&"")
                                );
                            }
                        } else {
                            println!("  {}", message);
                        }
                    }
                    _ => {
                        println!("  {}", error);
                    }
                }

                println!();
            }
        }

        // If target line specified, show analysis for that line
        if let Some(line_num) = target_line {
            if line_num > 0 && line_num <= lines.len() {
                println!("Analysis for line {}:", line_num);
                println!("  Content: {}", lines[line_num - 1]);
                // Additional line-specific analysis could be added here
            }
        }

        if let Ok(graph) = parse_result {
            println!("Successfully parsed {} triples", graph.len());
        }

        Ok(())
    }

    /// Benchmark file parsing performance
    fn benchmark_file(
        &self,
        path: &str,
        iterations: usize,
        warmup: usize,
    ) -> Result<BenchmarkResults> {
        let content = fs::read_to_string(path)?;
        let format = self.detect_format(path, &content)?;
        let file_size = content.len();

        println!(
            "Benchmarking {} ({} bytes, {} iterations + {} warmup)",
            path, file_size, iterations, warmup
        );

        // Warmup runs
        for _ in 0..warmup {
            let mut parser = StarParser::new();
            let _ = parser.parse_str(&content, format);
        }

        // Benchmark runs
        let mut parse_times = Vec::with_capacity(iterations);
        let mut serialize_times = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            // Parse benchmark
            let start = Instant::now();
            let mut parser = StarParser::new();
            let graph = parser.parse_str(&content, format)?;
            let parse_duration = start.elapsed();
            parse_times.push(parse_duration);

            // Serialize benchmark
            let start = Instant::now();
            let mut serializer = StarSerializer::new();
            let _output =
                serializer.serialize_graph(&graph, format, &SerializationOptions::default())?;
            let serialize_duration = start.elapsed();
            serialize_times.push(serialize_duration);
        }

        Ok(BenchmarkResults {
            file_size,
            iterations,
            parse_times,
            serialize_times,
        })
    }

    /// Execute SPARQL-star query
    fn execute_query(&self, data_path: &str, query_input: &str, output_format: &str) -> Result<()> {
        use crate::query::{QueryEngine, QueryResult};

        // Load data
        let content = fs::read_to_string(data_path)?;
        let format = self.detect_format(data_path, &content)?;

        let mut parser = StarParser::new();
        let graph = parser.parse_str(&content, format)?;

        let mut store = StarStore::new();
        for triple in &graph {
            store.insert(&triple)?;
        }

        // Load query
        let query_text = if Path::new(query_input).exists() {
            fs::read_to_string(query_input)?
        } else {
            query_input.to_string()
        };

        info!("Executing SPARQL-star query on {} triples", store.len());

        // Execute query using the query engine
        let mut engine = QueryEngine::new();
        let start_time = Instant::now();

        match engine.execute(&query_text, &store) {
            Ok(result) => {
                let duration = start_time.elapsed();

                if !self.quiet {
                    println!("Query executed in {:?}", duration);
                }

                self.format_query_results(&result, output_format)?;
            }
            Err(e) => {
                error!("Query execution failed: {}", e);
                return Err(anyhow!("Query execution failed: {}", e));
            }
        }

        Ok(())
    }

    // Helper methods

    fn detect_format(&self, path: &str, content: &str) -> Result<StarFormat> {
        let path_lower = path.to_lowercase();

        if path_lower.ends_with(".ttls") || path_lower.ends_with(".turtle-star") {
            return Ok(StarFormat::TurtleStar);
        }
        if path_lower.ends_with(".nts") || path_lower.ends_with(".ntriples-star") {
            return Ok(StarFormat::NTriplesStar);
        }
        if path_lower.ends_with(".trigs") || path_lower.ends_with(".trig-star") {
            return Ok(StarFormat::TrigStar);
        }
        if path_lower.ends_with(".nqs") || path_lower.ends_with(".nquads-star") {
            return Ok(StarFormat::NQuadsStar);
        }
        if path_lower.ends_with(".jlds")
            || path_lower.ends_with(".jsonld-star")
            || path_lower.ends_with(".json")
        {
            return Ok(StarFormat::JsonLdStar);
        }

        // Try to detect from content
        if content.trim_start().starts_with("{") || content.trim_start().starts_with("[") {
            // Looks like JSON - assume JSON-LD-star
            Ok(StarFormat::JsonLdStar)
        } else if content.contains("<<") && content.contains(">>") {
            if content.contains("GRAPH") || content.contains("{") {
                Ok(StarFormat::TrigStar)
            } else {
                Ok(StarFormat::TurtleStar)
            }
        } else {
            Ok(StarFormat::TurtleStar) // Default
        }
    }

    fn has_quoted_terms(&self, triple: &StarTriple) -> bool {
        matches!(triple.subject, StarTerm::QuotedTriple(_))
            || matches!(triple.object, StarTerm::QuotedTriple(_))
    }

    fn analyze_triple(&self, triple: &StarTriple, analysis: &mut AnalysisResult, depth: usize) {
        analysis.max_nesting_depth = analysis.max_nesting_depth.max(depth);

        if self.has_quoted_terms(triple) {
            analysis.quoted_triples += 1;
        }

        // Analyze terms
        self.analyze_term(&triple.subject, analysis, depth + 1);
        analysis.predicates.insert(triple.predicate.to_string());
        self.analyze_term(&triple.object, analysis, depth + 1);
    }

    fn analyze_term(&self, term: &StarTerm, analysis: &mut AnalysisResult, depth: usize) {
        match term {
            StarTerm::QuotedTriple(quoted) => {
                analysis.max_nesting_depth = analysis.max_nesting_depth.max(depth);
                self.analyze_triple(quoted, analysis, depth);
            }
            StarTerm::NamedNode(node) => {
                analysis.subjects.insert(node.iri.clone());
                if let Some(namespace) = self.extract_namespace(&node.iri) {
                    analysis.namespaces.insert(namespace);
                }
            }
            _ => {
                analysis.objects.insert(term.to_string());
            }
        }
    }

    fn extract_namespace(&self, iri: &str) -> Option<String> {
        if let Some(pos) = iri.rfind(['#', '/']) {
            Some(iri[..=pos].to_string())
        } else {
            None
        }
    }

    fn print_validation_result(&self, result: &ValidationResult) {
        if result.is_valid {
            println!("✓ Validation successful");
        } else {
            println!("✗ Validation failed");
        }

        println!("Format: {:?}", result.format);
        println!("Total triples: {}", result.triple_count);
        println!("Quoted triples: {}", result.quoted_triple_count);

        if !result.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &result.warnings {
                println!("  ⚠ {}", warning);
            }
        }

        if !result.errors.is_empty() {
            println!("\nErrors:");
            for error in &result.errors {
                println!("  ✗ {}", error);
            }
        }
    }

    fn write_validation_report(&self, result: &ValidationResult, path: &str) -> Result<()> {
        let report = serde_json::to_string_pretty(result)?;
        fs::write(path, report)?;
        println!("Validation report written to: {}", path);
        Ok(())
    }

    fn print_analysis_result(&self, analysis: &AnalysisResult) {
        println!("Analysis Results:");
        println!("================");
        println!("Format: {:?}", analysis.format);
        println!("Total triples: {}", analysis.total_triples);
        println!("Quoted triples: {}", analysis.quoted_triples);
        println!("Unique subjects: {}", analysis.subjects.len());
        println!("Unique predicates: {}", analysis.predicates.len());
        println!("Unique objects: {}", analysis.objects.len());
        println!("Max nesting depth: {}", analysis.max_nesting_depth);
        println!("Namespaces: {}", analysis.namespaces.len());

        if !analysis.namespaces.is_empty() {
            println!("\nNamespaces found:");
            for ns in &analysis.namespaces {
                println!("  {}", ns);
            }
        }
    }

    fn format_analysis_report(&self, analysis: &AnalysisResult) -> String {
        format!(
            "RDF-star Analysis Report\n\
             ========================\n\
             Format: {:?}\n\
             Total triples: {}\n\
             Quoted triples: {}\n\
             Unique subjects: {}\n\
             Unique predicates: {}\n\
             Unique objects: {}\n\
             Max nesting depth: {}\n\
             Namespaces: {}\n",
            analysis.format,
            analysis.total_triples,
            analysis.quoted_triples,
            analysis.subjects.len(),
            analysis.predicates.len(),
            analysis.objects.len(),
            analysis.max_nesting_depth,
            analysis.namespaces.len()
        )
    }

    fn print_benchmark_results(&self, results: &BenchmarkResults) {
        let avg_parse: f64 = results
            .parse_times
            .iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>()
            / results.iterations as f64;
        let avg_serialize: f64 = results
            .serialize_times
            .iter()
            .map(|d| d.as_secs_f64())
            .sum::<f64>()
            / results.iterations as f64;

        println!("Benchmark Results:");
        println!("==================");
        println!("File size: {} bytes", results.file_size);
        println!("Iterations: {}", results.iterations);
        println!("Average parse time: {:.3}ms", avg_parse * 1000.0);
        println!("Average serialize time: {:.3}ms", avg_serialize * 1000.0);
        println!(
            "Parse throughput: {:.2} MB/s",
            (results.file_size as f64) / (1024.0 * 1024.0 * avg_parse)
        );
        println!(
            "Serialize throughput: {:.2} MB/s",
            (results.file_size as f64) / (1024.0 * 1024.0 * avg_serialize)
        );
    }

    /// Run troubleshooting diagnostics
    fn troubleshoot_command(&self, matches: &ArgMatches) -> Result<()> {
        let error_input = matches.get_one::<String>("error").unwrap();
        let output_path = matches.get_one::<String>("output");

        let guide = TroubleshootingGuide::new();
        let analyzer = DiagnosticAnalyzer::new();

        info!("Analyzing error: {}", error_input);

        // Analyze the error
        let diagnosis = analyzer.analyze_error(error_input)?;

        // Get troubleshooting recommendations
        let recommendations = guide.get_recommendations(&diagnosis)?;

        // System health check
        let health_report = self.run_system_diagnostics()?;

        let report = format!(
            "RDF-star Troubleshooting Report\n"
                + "================================\n\n"
                + "Error Analysis:\n"
                + "---------------\n"
                + "Error Type: {}\n"
                + "Severity: {:?}\n"
                + "Description: {}\n\n"
                + "Recommendations:\n"
                + "----------------\n{}"
                + "\n\nSystem Health:\n"
                + "---------------\n{}",
            diagnosis.error_type,
            diagnosis.severity,
            diagnosis.description,
            recommendations
                .iter()
                .map(|r| format!("• {}\n", r))
                .collect::<String>(),
            health_report
        );

        if let Some(output) = output_path {
            fs::write(output, &report)?;
            println!("Troubleshooting report written to: {}", output);
        } else {
            println!("{}", report);
        }

        Ok(())
    }

    fn migrate_command(&self, matches: &ArgMatches) -> Result<()> {
        let source_file = matches.get_one::<String>("source").unwrap();
        let output_file = matches.get_one::<String>("output").unwrap();
        let source_format = matches.get_one::<String>("source-format").unwrap();
        let plan_only = matches.get_flag("plan");

        info!(
            "Starting RDF-star migration from {} to {}",
            source_file, output_file
        );

        let mut assistant = MigrationAssistant::new();
        let migration_format = source_format
            .parse::<MigrationSourceFormat>()
            .map_err(|_| anyhow!("Unsupported source format: {}", source_format))?;

        // Analyze source data
        let analysis = assistant.analyze_source(source_file, migration_format)?;

        if !self.quiet {
            println!("Source Analysis:");
            println!("  Format: {:?}", migration_format);
            println!("  Triples: {}", analysis.triple_count);
            println!(
                "  Estimated quoted triples after migration: {}",
                analysis.estimated_star_triples
            );
            println!("  Complexity: {:?}", analysis.complexity);
        }

        // Generate migration plan
        let plan = assistant.create_migration_plan(&analysis)?;

        if plan_only {
            println!("Migration Plan:");
            println!("===============");
            for (i, step) in plan.steps.iter().enumerate() {
                println!("{}. {}", i + 1, step.description);
                if let Some(notes) = &step.notes {
                    println!("   Notes: {}", notes);
                }
            }
            return Ok(());
        }

        // Execute migration
        let start_time = Instant::now();
        let result = assistant.execute_migration(source_file, output_file, &plan)?;
        let duration = start_time.elapsed();

        if !self.quiet {
            println!("Migration completed in {:?}", duration);
            println!("Results:");
            println!("  Input triples: {}", result.input_triples);
            println!("  Output triples: {}", result.output_triples);
            println!(
                "  Quoted triples created: {}",
                result.quoted_triples_created
            );
            println!("  Warnings: {}", result.warnings.len());

            if !result.warnings.is_empty() {
                println!("\nWarnings:");
                for warning in &result.warnings {
                    println!("  ⚠ {}", warning);
                }
            }
        }

        Ok(())
    }

    fn doctor_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_file = matches.get_one::<String>("input").unwrap();
        let report_path = matches.get_one::<String>("report");
        let auto_fix = matches.get_flag("fix");

        info!(
            "Running comprehensive diagnostic analysis on: {}",
            input_file
        );

        let analyzer = DiagnosticAnalyzer::new();
        let start_time = Instant::now();

        // Comprehensive file analysis
        let diagnostic_result = analyzer.run_comprehensive_analysis(input_file)?;
        let duration = start_time.elapsed();

        let mut issues_found = 0;
        let mut fixes_applied = 0;

        // System health check
        let system_health = self.run_system_diagnostics()?;

        // Performance analysis
        let perf_analysis = self.run_performance_analysis(input_file)?;

        // Generate comprehensive report
        let report = format!(
            "RDF-star Diagnostic Report\n"
                + "===========================\n\n"
                + "File: {}\n"
                + "Analysis Duration: {:?}\n\n"
                + "Structural Analysis:\n"
                + "--------------------\n"
                + "Total Triples: {}\n"
                + "Quoted Triples: {}\n"
                + "Max Nesting Depth: {}\n"
                + "Syntax Errors: {}\n"
                + "Semantic Issues: {}\n\n"
                + "Quality Assessment:\n"
                + "-------------------\n"
                + "Overall Score: {}/100\n"
                + "Readability: {}/10\n"
                + "Efficiency: {}/10\n"
                + "Compliance: {}/10\n\n"
                + "Issues Found:\n"
                + "-------------\n{}"
                + "\nPerformance Analysis:\n"
                + "---------------------\n{}"
                + "\nSystem Health:\n"
                + "---------------\n{}",
            input_file,
            duration,
            diagnostic_result.total_triples,
            diagnostic_result.quoted_triples,
            diagnostic_result.max_nesting_depth,
            diagnostic_result.syntax_errors.len(),
            diagnostic_result.semantic_issues.len(),
            diagnostic_result.quality_score,
            diagnostic_result.readability_score,
            diagnostic_result.efficiency_score,
            diagnostic_result.compliance_score,
            diagnostic_result
                .issues
                .iter()
                .map(|issue| format!(
                    "• {} ({}): {}\n",
                    issue.severity, issue.category, issue.description
                ))
                .collect::<String>(),
            perf_analysis,
            system_health
        );

        // Apply automatic fixes if requested
        if auto_fix {
            let fixes = analyzer.apply_automatic_fixes(input_file, &diagnostic_result.issues)?;
            fixes_applied = fixes.len();

            if !self.quiet && fixes_applied > 0 {
                println!("Applied {} automatic fixes:", fixes_applied);
                for fix in &fixes {
                    println!("  ✓ {}", fix);
                }
            }
        }

        issues_found = diagnostic_result.issues.len();

        if let Some(report_file) = report_path {
            fs::write(report_file, &report)?;
            println!("Diagnostic report written to: {}", report_file);
        } else if !self.quiet {
            println!("{}", report);
        }

        // Summary
        if !self.quiet {
            println!("\nDiagnostic Summary:");
            println!("===================");
            println!("Issues found: {}", issues_found);
            if auto_fix {
                println!("Fixes applied: {}", fixes_applied);
            }
            println!(
                "Overall health: {}",
                if issues_found == 0 {
                    "Excellent"
                } else if issues_found < 5 {
                    "Good"
                } else if issues_found < 10 {
                    "Fair"
                } else {
                    "Needs attention"
                }
            );
        }

        Ok(())
    }
}

impl Default for StarCli {
    fn default() -> Self {
        Self::new()
    }
}

// Result structures

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ValidationResult {
    is_valid: bool,
    errors: Vec<String>,
    warnings: Vec<String>,
    triple_count: usize,
    quoted_triple_count: usize,
    format: StarFormat,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct AnalysisResult {
    format: StarFormat,
    total_triples: usize,
    quoted_triples: usize,
    subjects: std::collections::HashSet<String>,
    predicates: std::collections::HashSet<String>,
    objects: std::collections::HashSet<String>,
    max_nesting_depth: usize,
    namespaces: std::collections::HashSet<String>,
}

#[derive(Debug)]
struct BenchmarkResults {
    file_size: usize,
    iterations: usize,
    parse_times: Vec<std::time::Duration>,
    serialize_times: Vec<std::time::Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_creation() {
        let cli = StarCli::new();
        assert!(!cli.verbose);
        assert!(!cli.quiet);
    }

    #[test]
    fn test_format_detection() {
        let cli = StarCli::new();

        // Test file extension detection
        assert_eq!(
            cli.detect_format("test.ttls", "").unwrap(),
            StarFormat::TurtleStar
        );

        assert_eq!(
            cli.detect_format("test.nts", "").unwrap(),
            StarFormat::NTriplesStar
        );

        // Test content-based detection
        assert_eq!(
            cli.detect_format("test.txt", "<< :s :p :o >> :meta :value .")
                .unwrap(),
            StarFormat::TurtleStar
        );
    }

    #[test]
    fn test_namespace_extraction() {
        let cli = StarCli::new();

        assert_eq!(
            cli.extract_namespace("http://example.org/person#name"),
            Some("http://example.org/person#".to_string())
        );

        assert_eq!(
            cli.extract_namespace("http://example.org/data/"),
            Some("http://example.org/data/".to_string())
        );
    }
}
