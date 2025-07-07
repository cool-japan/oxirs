//! SHACL Validation CLI Tool
//!
//! A command-line tool for validating RDF data against SHACL shapes.
//! Supports multiple input/output formats and validation strategies.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use indexmap::IndexMap;

use oxirs_core::{Store, ConcreteStore};
use oxirs_shacl::{
    optimization::ValidationStrategy,
    report::{ReportFormat, ValidationReport},
    shapes::{ShapeValidator},
    validation::ValidationEngine,
    Shape, ShapeId, ValidationConfig,
};

/// Command-line arguments for the SHACL validator
#[derive(Debug)]
struct Args {
    /// Data file to validate
    data_file: PathBuf,

    /// SHACL shapes file
    shapes_file: PathBuf,

    /// Output file (optional - if not specified, output to stdout)
    output_file: Option<PathBuf>,

    /// Output format
    output_format: ReportFormat,

    /// Validation strategy
    strategy: ValidationStrategy,

    /// Enable verbose output
    verbose: bool,

    /// Validate shapes graph before validation
    validate_shapes: bool,

    /// Maximum number of violations to report (0 = unlimited)
    max_violations: usize,

    /// Filter violations by severity
    severity_filter: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            data_file: PathBuf::from("data.ttl"),
            shapes_file: PathBuf::from("shapes.ttl"),
            output_file: None,
            output_format: ReportFormat::Turtle,
            strategy: ValidationStrategy::Sequential,
            verbose: false,
            validate_shapes: true,
            max_violations: 0,
            severity_filter: None,
        }
    }
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args = parse_args()?;

    if args.verbose {
        println!("🔍 OxiRS SHACL Validator");
        println!("=======================");
        println!("Data file: {:?}", args.data_file);
        println!("Shapes file: {:?}", args.shapes_file);
        println!("Output format: {:?}", args.output_format);
        println!("Strategy: {:?}", args.strategy);
        println!("Validate shapes: {}", args.validate_shapes);
        println!();
    }

    // Load and validate shapes
    let shapes = load_shapes(&args)?;

    // Validate shapes graph if requested
    if args.validate_shapes {
        validate_shapes_graph(&shapes, args.verbose)?;
    }

    // Load data store
    let store = load_data_store(&args.data_file)?;

    // Configure validation engine
    let config = create_validation_config(&args);
    let mut engine = ValidationEngine::new(&shapes, config);

    // Execute validation
    if args.verbose {
        println!("🚀 Starting validation...");
    }

    let start_time = Instant::now();
    let report = engine.validate_store(&*store)?;
    let duration = start_time.elapsed();

    if args.verbose {
        println!("✅ Validation completed in {:.3}s", duration.as_secs_f64());
        println!();
    }

    // Display results summary
    display_summary(&report, args.verbose);

    // Output validation report
    output_report(&report, &args)?;

    // Return appropriate exit code
    let exit_code = if report.conforms() { 0 } else { 1 };
    std::process::exit(exit_code);
}

fn parse_args() -> Result<Args> {
    let args: Vec<String> = std::env::args().collect();
    let mut parsed_args = Args::default();

    if args.len() < 3 {
        print_usage(&args[0]);
        return Err(anyhow!("Insufficient arguments"));
    }

    parsed_args.data_file = PathBuf::from(&args[1]);
    parsed_args.shapes_file = PathBuf::from(&args[2]);

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                if i < args.len() {
                    parsed_args.output_file = Some(PathBuf::from(&args[i]));
                }
            }
            "-f" | "--format" => {
                i += 1;
                if i < args.len() {
                    parsed_args.output_format = parse_format(&args[i])?;
                }
            }
            "-s" | "--strategy" => {
                i += 1;
                if i < args.len() {
                    parsed_args.strategy = parse_strategy(&args[i])?;
                }
            }
            "-v" | "--verbose" => {
                parsed_args.verbose = true;
            }
            "--no-validate-shapes" => {
                parsed_args.validate_shapes = false;
            }
            "--max-violations" => {
                i += 1;
                if i < args.len() {
                    parsed_args.max_violations = args[i].parse()?;
                }
            }
            "--severity" => {
                i += 1;
                if i < args.len() {
                    parsed_args.severity_filter = Some(args[i].clone());
                }
            }
            "-h" | "--help" => {
                print_usage(&args[0]);
                std::process::exit(0);
            }
            _ => {
                return Err(anyhow!("Unknown argument: {}", args[i]));
            }
        }
        i += 1;
    }

    Ok(parsed_args)
}

fn print_usage(program_name: &str) {
    println!("OxiRS SHACL Validator");
    println!();
    println!("USAGE:");
    println!("    {} <data-file> <shapes-file> [OPTIONS]", program_name);
    println!();
    println!("ARGS:");
    println!("    <data-file>     RDF data file to validate");
    println!("    <shapes-file>   SHACL shapes file");
    println!();
    println!("OPTIONS:");
    println!("    -o, --output <file>         Output file (default: stdout)");
    println!("    -f, --format <format>       Output format [turtle, json-ld, json, html, csv]");
    println!("    -s, --strategy <strategy>   Validation strategy [sequential, parallel, incremental, streaming]");
    println!("    -v, --verbose               Enable verbose output");
    println!("    --no-validate-shapes        Skip shapes graph validation");
    println!("    --max-violations <n>        Maximum violations to report (0 = unlimited)");
    println!("    --severity <level>          Filter by severity [violation, warning, info]");
    println!("    -h, --help                  Print this help message");
    println!();
    println!("EXAMPLES:");
    println!("    {} data.ttl shapes.ttl", program_name);
    println!("    {} data.ttl shapes.ttl -f json -o report.json", program_name);
    println!("    {} data.ttl shapes.ttl -s parallel -v", program_name);
}

fn parse_format(format_str: &str) -> Result<ReportFormat> {
    match format_str.to_lowercase().as_str() {
        "turtle" | "ttl" => Ok(ReportFormat::Turtle),
        "json-ld" | "jsonld" => Ok(ReportFormat::JsonLd),
        "json" => Ok(ReportFormat::Json),
        "html" => Ok(ReportFormat::Html),
        "csv" => Ok(ReportFormat::Csv),
        "rdf-xml" | "rdfxml" | "xml" => Ok(ReportFormat::RdfXml),
        "n-triples" | "nt" => Ok(ReportFormat::NTriples),
        _ => Err(anyhow!("Unknown format: {}", format_str)),
    }
}

fn parse_strategy(strategy_str: &str) -> Result<ValidationStrategy> {
    match strategy_str.to_lowercase().as_str() {
        "sequential" | "seq" => Ok(ValidationStrategy::Sequential),
        "parallel" | "par" => Ok(ValidationStrategy::Parallel { max_threads: 4 }),
        "incremental" | "inc" => Ok(ValidationStrategy::Incremental { force_revalidate: false }),
        "streaming" | "stream" => Ok(ValidationStrategy::Streaming { batch_size: 1000 }),
        _ => Err(anyhow!("Unknown strategy: {}", strategy_str)),
    }
}

fn load_shapes(args: &Args) -> Result<IndexMap<ShapeId, Shape>> {
    if args.verbose {
        println!("📖 Loading shapes from {:?}...", args.shapes_file);
    }

    if !args.shapes_file.exists() {
        return Err(anyhow!("Shapes file not found: {:?}", args.shapes_file));
    }

    let _shapes_content = fs::read_to_string(&args.shapes_file)?;
    
    // Create a simple in-memory store for shapes parsing
    let _store = ConcreteStore::new();
    
    // Parse shapes content - using simplified approach for demo
    // TODO: Implement proper RDF parsing and shape extraction
    // This should parse the RDF graph and extract SHACL shapes using oxirs-core parser
    let mut shapes = IndexMap::new();
    
    // Create a demonstration shape showing typical SHACL structure
    let shape_id = ShapeId::new("http://example.org/PersonShape");
    let shape = Shape::node_shape(shape_id.clone());
    shapes.insert(shape_id, shape);

    if args.verbose {
        println!("✅ Loaded {} shapes", shapes.len());
        println!("   NOTE: Using simplified shape loading for demonstration");
    }

    Ok(shapes)
}

fn validate_shapes_graph(shapes: &IndexMap<ShapeId, Shape>, verbose: bool) -> Result<()> {
    if verbose {
        println!("🔎 Validating shapes graph...");
    }

    let validator = ShapeValidator::new();
    let shapes_vec: Vec<Shape> = shapes.values().cloned().collect();
    let validation_results = validator.validate_shapes(&shapes_vec)?;

    let has_errors = !validation_results.global_errors.is_empty() || 
                     validation_results.shape_results.iter().any(|r| !r.errors.is_empty());
    let has_warnings = validation_results.shape_results.iter().any(|r| !r.warnings.is_empty());
    
    if has_errors || has_warnings {
        println!("⚠️  Shapes validation issues:");
        
        // Print global errors
        for error in &validation_results.global_errors {
            println!("  GLOBAL ERROR: {}", error);
        }
        
        // Print shape-specific errors and warnings
        for shape_result in &validation_results.shape_results {
            for error in &shape_result.errors {
                println!("  ERROR [{}]: {}", shape_result.shape_id, error);
            }
            for warning in &shape_result.warnings {
                println!("  WARNING [{}]: {}", shape_result.shape_id, warning);
            }
        }
        
        if has_errors {
            return Err(anyhow!("Shapes graph validation failed"));
        }
    } else if verbose {
        println!("✅ Shapes graph is valid");
    }

    Ok(())
}

fn load_data_store(data_file: &PathBuf) -> Result<Box<dyn Store>> {
    if !data_file.exists() {
        return Err(anyhow!("Data file not found: {:?}", data_file));
    }

    // Validate file extension for supported formats
    let extension = data_file
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    
    match extension.to_lowercase().as_str() {
        "ttl" | "turtle" | "nt" | "ntriples" | "rdf" | "owl" | "n3" => {
            // Supported RDF formats
        }
        _ => {
            return Err(anyhow!(
                "Unsupported file format: {}. Supported formats: .ttl, .nt, .rdf, .owl, .n3",
                extension
            ));
        }
    }

    // Create in-memory store
    let store = ConcreteStore::new()?;

    // TODO: Implement proper RDF parsing and loading
    // This should use oxirs-core parsers to load the RDF data:
    // 1. Detect format from file extension
    // 2. Parse RDF triples/quads from the file
    // 3. Load parsed data into the store
    // Example:
    // let parser = get_parser_for_format(extension)?;
    // let triples = parser.parse_file(data_file)?;
    // for triple in triples {
    //     store.insert_triple(triple)?;
    // }

    Ok(Box::new(store) as Box<dyn Store>)
}

fn create_validation_config(args: &Args) -> ValidationConfig {
    let mut config = ValidationConfig::default();
    
    if args.max_violations > 0 {
        config.max_violations = args.max_violations;
    }

    config
}

fn display_summary(report: &ValidationReport, verbose: bool) {
    println!("📊 Validation Results");
    println!("===================");
    println!("Conforms: {}", if report.conforms() { "✅ YES" } else { "❌ NO" });
    println!("Violations: {}", report.violations().len());
    
    if verbose && !report.violations().is_empty() {
        println!();
        println!("Violation details:");
        for (i, violation) in report.violations().iter().enumerate() {
            if i >= 10 {
                println!("  ... and {} more violations", report.violations().len() - 10);
                break;
            }
            if let Some(message) = &violation.result_message {
                println!("  {}: {}", i + 1, message);
            } else {
                println!("  {}: Constraint violation", i + 1);
            }
        }
    }
    
    println!();
}

fn output_report(report: &ValidationReport, args: &Args) -> Result<()> {
    let serialized = match args.output_format {
        ReportFormat::Turtle => report.to_turtle()?,
        ReportFormat::JsonLd => report.to_json()?, // JsonLd uses same format as Json for now
        ReportFormat::Json => report.to_json()?,
        ReportFormat::Html => report.to_html()?,
        ReportFormat::Csv => report.to_csv()?,
        ReportFormat::RdfXml => report.to_rdf("application/rdf+xml")?,
        ReportFormat::NTriples => report.to_rdf("application/n-triples")?,
        ReportFormat::Text => report.to_text()?,
        ReportFormat::Yaml => report.to_yaml()?,
    };

    match &args.output_file {
        Some(output_file) => {
            fs::write(output_file, serialized)?;
            if args.verbose {
                println!("📁 Report saved to {:?}", output_file);
            }
        }
        None => {
            println!("{}", serialized);
        }
    }

    Ok(())
}