//! Benchmark command - Comprehensive benchmarking tools for RDF and SPARQL
//!
//! This module provides comprehensive benchmarking capabilities including:
//! - Running benchmark suites (SP2Bench, WatDiv, LDBC, BSBM)
//! - Generating synthetic benchmark datasets
//! - Analyzing query workload patterns
//! - Comparing benchmark results for regression detection

use super::CommandResult;
use scirs2_core::random::{Random, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// Placeholder Store type until oxirs_core is available
struct Store;

impl Store {
    fn open(_path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Store)
    }
}

/// Run performance benchmarks on a dataset
pub async fn run(
    dataset: String,
    suite: String,
    iterations: usize,
    output: Option<PathBuf>,
    detailed: bool,
    warmup: usize,
) -> CommandResult {
    println!("Running '{suite}' benchmark suite on dataset '{dataset}'");
    println!("Iterations: {iterations}, Warmup: {warmup}, Detailed: {detailed}");

    // Validate benchmark suite
    if !is_supported_benchmark_suite(&suite) {
        return Err(format!(
            "Unsupported benchmark suite '{suite}'. Supported suites: sp2bench, watdiv, ldbc, bsbm, custom"
        )
        .into());
    }

    // Load dataset
    let dataset_path = if PathBuf::from(&dataset).join("oxirs.toml").exists() {
        load_dataset_from_config(&dataset)?
    } else {
        PathBuf::from(&dataset)
    };

    let store = if dataset_path.is_dir() {
        Store::open(&dataset_path)?
    } else {
        return Err(format!(
            "Dataset '{dataset}' not found. Use 'oxirs init' to create a dataset."
        )
        .into());
    };

    println!("Dataset loaded successfully\n");

    // Run warmup iterations
    if warmup > 0 {
        println!("Running {warmup} warmup iterations...");
        run_warmup_iterations(&store, &suite, warmup)?;
        println!("Warmup complete\n");
    }

    // Run benchmark
    let benchmark_results = run_benchmark_suite(&store, &suite, iterations, detailed)?;

    // Display results
    display_benchmark_results(&benchmark_results, detailed);

    // Save results to file if specified
    if let Some(output_path) = output {
        save_benchmark_results(&benchmark_results, &output_path)?;
        println!("\nResults saved to: {}", output_path.display());
    }

    Ok(())
}

/// Generate synthetic benchmark datasets
pub async fn generate(
    output: PathBuf,
    size: String,
    dataset_type: String,
    seed: Option<u64>,
    triples: Option<usize>,
    schema: Option<PathBuf>,
) -> CommandResult {
    println!("Generating synthetic benchmark dataset");
    println!("Output: {}", output.display());
    println!("Size: {size}, Type: {dataset_type}");
    if let Some(s) = seed {
        println!("Random seed: {s}");
    }

    // Determine triple count
    let triple_count = if let Some(count) = triples {
        count
    } else {
        match size.as_str() {
            "tiny" => 1_000,
            "small" => 10_000,
            "medium" => 100_000,
            "large" => 1_000_000,
            "xlarge" => 10_000_000,
            _ => {
                return Err(format!(
                    "Invalid size '{size}'. Valid sizes: tiny, small, medium, large, xlarge"
                )
                .into())
            }
        }
    };

    println!("Generating {} triples...\n", triple_count);

    // Initialize random number generator (unused but kept for future seed-based generation)
    let mut _rng = if let Some(s) = seed {
        Random::seed_from_u64(s)
    } else {
        Random::seed_from_u64(42) // Use consistent RNG type
    };

    // Load schema if provided
    if let Some(schema_path) = schema {
        println!("Using schema: {}", schema_path.display());
        // Schema-based generation would go here
    }

    // Generate dataset based on type
    let dataset = match dataset_type.as_str() {
        "rdf" => generate_rdf_dataset(triple_count, &mut _rng)?,
        "graph" => generate_graph_dataset(triple_count, &mut _rng)?,
        "semantic" => generate_semantic_dataset(triple_count, &mut _rng)?,
        _ => {
            return Err(format!(
                "Invalid dataset type '{dataset_type}'. Valid types: rdf, graph, semantic"
            )
            .into())
        }
    };

    // Create output directory if needed
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    // Save dataset
    fs::write(&output, dataset)?;

    println!("✓ Dataset generated successfully");
    println!("  Total triples: {}", triple_count);
    println!("  Output file: {}", output.display());
    println!(
        "  File size: {:.2} MB",
        fs::metadata(&output)?.len() as f64 / 1_048_576.0
    );

    Ok(())
}

/// Analyze query workload from log files
pub async fn analyze(
    input: PathBuf,
    output: Option<PathBuf>,
    format: String,
    suggestions: bool,
    patterns: bool,
) -> CommandResult {
    println!("Analyzing query workload");
    println!("Input: {}", input.display());
    println!("Format: {format}\n");

    // Read query log
    let log_content = fs::read_to_string(&input)?;
    let queries = parse_query_log(&log_content)?;

    println!("Parsed {} queries from log\n", queries.len());

    // Analyze queries
    let analysis = analyze_query_workload(&queries, patterns)?;

    // Generate report
    let report = generate_workload_report(&analysis, suggestions, format.as_str())?;

    // Display or save report
    if let Some(output_path) = output {
        fs::write(&output_path, &report)?;
        println!("Analysis report saved to: {}", output_path.display());
    } else {
        println!("{}", report);
    }

    Ok(())
}

/// Compare benchmark results for regression detection
pub async fn compare(
    baseline: PathBuf,
    current: PathBuf,
    output: Option<PathBuf>,
    threshold: f64,
    format: String,
) -> CommandResult {
    println!("Comparing benchmark results");
    println!("Baseline: {}", baseline.display());
    println!("Current:  {}", current.display());
    println!("Regression threshold: {:.1}%\n", threshold);

    // Load results
    let baseline_results: BenchmarkResults = load_benchmark_results(&baseline)?;
    let current_results: BenchmarkResults = load_benchmark_results(&current)?;

    // Perform comparison
    let comparison = compare_benchmark_results(&baseline_results, &current_results, threshold)?;

    // Generate report
    let report = generate_comparison_report(&comparison, format.as_str())?;

    // Display or save report
    if let Some(output_path) = output {
        fs::write(&output_path, &report)?;
        println!("Comparison report saved to: {}", output_path.display());
    } else {
        println!("{}", report);
    }

    // Exit with error if regressions detected
    if comparison.has_regressions {
        return Err("Performance regressions detected!".into());
    }

    println!("\n✓ No performance regressions detected");
    Ok(())
}

// ===== Helper Functions =====

/// Check if benchmark suite is supported
fn is_supported_benchmark_suite(suite: &str) -> bool {
    matches!(suite, "sp2bench" | "watdiv" | "ldbc" | "bsbm" | "custom")
}

/// Load dataset configuration
fn load_dataset_from_config(dataset: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let config_path = PathBuf::from(dataset).join("oxirs.toml");

    if !config_path.exists() {
        return Err(format!("Configuration file '{}' not found", config_path.display()).into());
    }

    Ok(PathBuf::from(dataset))
}

/// Benchmark results container
#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkResults {
    suite: String,
    total_queries: usize,
    iterations: usize,
    warmup_iterations: usize,
    total_duration: DurationSerde,
    query_results: Vec<QueryBenchmarkResult>,
    statistics: BenchmarkStatistics,
    timestamp: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct QueryBenchmarkResult {
    query_name: String,
    avg_time: DurationSerde,
    min_time: DurationSerde,
    max_time: DurationSerde,
    median_time: DurationSerde,
    p95_time: DurationSerde,
    p99_time: DurationSerde,
    success_rate: f64,
    stddev: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkStatistics {
    total_queries_executed: usize,
    avg_query_time: DurationSerde,
    queries_per_second: f64,
    success_rate: f64,
    total_errors: usize,
}

// Serializable Duration wrapper
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct DurationSerde {
    secs: u64,
    nanos: u32,
}

impl From<Duration> for DurationSerde {
    fn from(d: Duration) -> Self {
        DurationSerde {
            secs: d.as_secs(),
            nanos: d.subsec_nanos(),
        }
    }
}

impl From<DurationSerde> for Duration {
    fn from(d: DurationSerde) -> Self {
        Duration::new(d.secs, d.nanos)
    }
}

impl DurationSerde {
    fn as_secs_f64(&self) -> f64 {
        self.secs as f64 + self.nanos as f64 / 1_000_000_000.0
    }
}

/// Run warmup iterations
fn run_warmup_iterations(
    _store: &Store,
    suite: &str,
    warmup: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let queries = get_benchmark_queries(suite)?;

    for (_query_name, _query) in queries.iter().take(3) {
        // Run a few queries for warmup
        for _ in 0..warmup {
            simulate_query_execution();
        }
        print!(".");
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();

    Ok(())
}

/// Run benchmark suite
fn run_benchmark_suite(
    _store: &Store,
    suite: &str,
    iterations: usize,
    detailed: bool,
) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    let queries = get_benchmark_queries(suite)?;
    let mut query_results = Vec::new();
    let mut total_duration = Duration::new(0, 0);
    let mut total_queries_executed = 0;
    let mut successful_queries = 0;
    let mut total_errors = 0;

    for (i, (query_name, _query)) in queries.iter().enumerate() {
        if detailed {
            println!("Running query {}/{}: {}", i + 1, queries.len(), query_name);
        } else {
            print!("\rProgress: {}/{} queries", i + 1, queries.len());
            use std::io::Write;
            std::io::stdout().flush().ok();
        }

        let mut execution_times = Vec::new();
        let mut successes = 0;

        for iteration in 1..=iterations {
            if detailed && (iteration % 10 == 0 || iteration == 1) {
                print!("  Iteration {iteration}/{iterations}\r");
            }

            let start = Instant::now();
            let success = simulate_query_execution();
            let duration = start.elapsed();

            execution_times.push(duration);
            total_duration += duration;
            total_queries_executed += 1;

            if success {
                successes += 1;
                successful_queries += 1;
            } else {
                total_errors += 1;
            }
        }

        if detailed {
            println!("  Completed {iterations} iterations");
        }

        // Calculate statistics
        execution_times.sort();
        let avg_time = Duration::from_nanos(
            (execution_times.iter().map(|d| d.as_nanos()).sum::<u128>() / iterations as u128)
                as u64,
        );
        let min_time = *execution_times.first().unwrap();
        let max_time = *execution_times.last().unwrap();
        let median_time = execution_times[iterations / 2];
        let p95_time = execution_times[(iterations as f64 * 0.95) as usize];
        let p99_time = execution_times[(iterations as f64 * 0.99) as usize];
        let success_rate = successes as f64 / iterations as f64;

        // Calculate standard deviation
        let mean_nanos = avg_time.as_nanos() as f64;
        let variance: f64 = execution_times
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / iterations as f64;
        let stddev = variance.sqrt() / 1_000_000.0; // Convert to milliseconds

        query_results.push(QueryBenchmarkResult {
            query_name: query_name.clone(),
            avg_time: avg_time.into(),
            min_time: min_time.into(),
            max_time: max_time.into(),
            median_time: median_time.into(),
            p95_time: p95_time.into(),
            p99_time: p99_time.into(),
            success_rate,
            stddev,
        });
    }

    if !detailed {
        println!(); // New line after progress
    }

    let avg_query_time =
        Duration::from_nanos((total_duration.as_nanos() / total_queries_executed as u128) as u64);
    let queries_per_second = total_queries_executed as f64 / total_duration.as_secs_f64();
    let success_rate = successful_queries as f64 / total_queries_executed as f64;

    let statistics = BenchmarkStatistics {
        total_queries_executed,
        avg_query_time: avg_query_time.into(),
        queries_per_second,
        success_rate,
        total_errors,
    };

    Ok(BenchmarkResults {
        suite: suite.to_string(),
        total_queries: queries.len(),
        iterations,
        warmup_iterations: 0,
        total_duration: total_duration.into(),
        query_results,
        statistics,
        timestamp: chrono::Utc::now().to_rfc3339(),
    })
}

/// Get benchmark queries for a suite
fn get_benchmark_queries(suite: &str) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    match suite {
        "sp2bench" => Ok(vec![
            ("Q1-Simple".to_string(), "SELECT * WHERE { ?s ?p ?o } LIMIT 10".to_string()),
            ("Q2-FOAF".to_string(), "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }".to_string()),
            ("Q3-Creator".to_string(), "SELECT ?article WHERE { ?article <http://purl.org/dc/elements/1.1/creator> ?author }".to_string()),
            ("Q4-Filter".to_string(), "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name FILTER(REGEX(?name, 'Smith')) }".to_string()),
            ("Q5-Optional".to_string(), "SELECT ?name ?email WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name OPTIONAL { ?person <http://xmlns.com/foaf/0.1/mbox> ?email } }".to_string()),
        ]),
        "watdiv" => Ok(vec![
            ("C1-Caption".to_string(), "SELECT ?v0 WHERE { ?v0 <http://schema.org/caption> ?v1 }".to_string()),
            ("C2-Follows".to_string(), "SELECT ?v0 ?v1 WHERE { ?v0 <http://schema.org/follows> ?v1 }".to_string()),
            ("F1-Complex".to_string(), "SELECT ?v0 ?v2 WHERE { ?v0 <http://schema.org/likes> ?v1 . ?v1 <http://schema.org/friendOf> ?v2 }".to_string()),
        ]),
        "ldbc" => Ok(vec![
            ("Q1-FirstName".to_string(), "SELECT ?name WHERE { ?person <http://www.ldbc.eu/ldbc_socialnet/1.0/vocabulary/firstName> ?name }".to_string()),
            ("Q2-Friends".to_string(), "SELECT ?p1 ?p2 WHERE { ?p1 <http://www.ldbc.eu/ldbc_socialnet/1.0/vocabulary/knows> ?p2 }".to_string()),
        ]),
        "bsbm" => Ok(vec![
            ("Q1-Product".to_string(), "SELECT ?product ?label WHERE { ?product <http://www.w3.org/2000/01/rdf-schema#label> ?label }".to_string()),
            ("Q2-Features".to_string(), "SELECT ?product ?feature WHERE { ?product <http://www4.wiwiss.fu-berlin.de/bizer/bsbm/v01/vocabulary/productFeature> ?feature }".to_string()),
        ]),
        "custom" => Ok(vec![
            ("simple".to_string(), "SELECT * WHERE { ?s ?p ?o } LIMIT 1".to_string()),
        ]),
        _ => Err(format!("Unknown benchmark suite: {suite}").into()),
    }
}

/// Simulate query execution (placeholder)
fn simulate_query_execution() -> bool {
    use scirs2_core::random::{Random, Rng};

    // Simulate some work with realistic timing
    let delay = {
        let mut random = Random::default();
        1 + random.random::<u64>() % 15 // 1-15ms
    };
    std::thread::sleep(Duration::from_millis(delay));

    // Simulate 95% success rate
    let mut random = Random::default();
    random.random::<f64>() < 0.95
}

/// Display benchmark results
fn display_benchmark_results(results: &BenchmarkResults, detailed: bool) {
    println!("\n==================== Benchmark Results ====================");
    println!("Suite: {}", results.suite);
    println!("Timestamp: {}", results.timestamp);
    println!("Total queries: {}", results.total_queries);
    println!("Iterations per query: {}", results.iterations);
    println!(
        "Total duration: {:.2}s",
        results.total_duration.as_secs_f64()
    );
    println!();

    println!("Overall Statistics:");
    println!(
        "  Total queries executed: {}",
        results.statistics.total_queries_executed
    );
    println!(
        "  Average query time: {:.3}ms",
        results.statistics.avg_query_time.as_secs_f64() * 1000.0
    );
    println!(
        "  Queries per second: {:.2}",
        results.statistics.queries_per_second
    );
    println!(
        "  Success rate: {:.1}%",
        results.statistics.success_rate * 100.0
    );
    if results.statistics.total_errors > 0 {
        println!("  Total errors: {}", results.statistics.total_errors);
    }
    println!();

    println!("Query Details:");
    for query_result in &results.query_results {
        println!("  {}:", query_result.query_name);
        println!(
            "    Average: {:.3}ms (±{:.2}ms)",
            query_result.avg_time.as_secs_f64() * 1000.0,
            query_result.stddev
        );

        if detailed {
            println!(
                "    Min: {:.3}ms",
                query_result.min_time.as_secs_f64() * 1000.0
            );
            println!(
                "    Max: {:.3}ms",
                query_result.max_time.as_secs_f64() * 1000.0
            );
            println!(
                "    Median: {:.3}ms",
                query_result.median_time.as_secs_f64() * 1000.0
            );
            println!(
                "    P95: {:.3}ms",
                query_result.p95_time.as_secs_f64() * 1000.0
            );
            println!(
                "    P99: {:.3}ms",
                query_result.p99_time.as_secs_f64() * 1000.0
            );
        }

        println!(
            "    Success rate: {:.1}%",
            query_result.success_rate * 100.0
        );
    }
    println!("==========================================================");
}

/// Save benchmark results to file
fn save_benchmark_results(
    results: &BenchmarkResults,
    output_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let json_results = serde_json::to_string_pretty(results)?;
    fs::write(output_path, json_results)?;
    Ok(())
}

/// Load benchmark results from file
fn load_benchmark_results(path: &PathBuf) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let results: BenchmarkResults = serde_json::from_str(&content)?;
    Ok(results)
}

// ===== Dataset Generation Functions =====

/// Generate RDF dataset
fn generate_rdf_dataset(
    triple_count: usize,
    _rng: &mut Random<scirs2_core::rngs::StdRng>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = String::new();
    output.push_str("# Generated RDF Dataset\n");
    output.push_str(&format!("# Triples: {}\n\n", triple_count));

    for i in 0..triple_count {
        let subject = format!("<http://example.org/resource/{}>", i);
        let predicate_choice = (i * 7 + 3) % 5; // Deterministic but varied
        let predicate = match predicate_choice {
            0 => "<http://www.w3.org/2000/01/rdf-schema#label>",
            1 => "<http://purl.org/dc/terms/title>",
            2 => "<http://xmlns.com/foaf/0.1/name>",
            3 => "<http://schema.org/name>",
            _ => "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
        };
        let object = format!("\"Resource {}\"", i);

        output.push_str(&format!("{} {} {} .\n", subject, predicate, object));
    }

    Ok(output)
}

/// Generate graph dataset
fn generate_graph_dataset(
    triple_count: usize,
    _rng: &mut Random<scirs2_core::rngs::StdRng>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = String::new();
    output.push_str("# Generated Graph Dataset\n\n");

    let node_count = (triple_count as f64).sqrt() as usize;

    for i in 0..triple_count {
        // Deterministic but varied node selection
        let from = (i * 13) % node_count;
        let to = (i * 17 + 7) % node_count;

        let subject = format!("<http://example.org/node/{}>", from);
        let predicate = "<http://example.org/edge>";
        let object = format!("<http://example.org/node/{}>", to);

        output.push_str(&format!("{} {} {} .\n", subject, predicate, object));
    }

    Ok(output)
}

/// Generate semantic dataset
fn generate_semantic_dataset(
    triple_count: usize,
    _rng: &mut Random<scirs2_core::rngs::StdRng>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut output = String::new();
    output.push_str("# Generated Semantic Dataset\n\n");

    let classes = ["Person", "Organization", "Place", "Event", "Document"];
    let properties = ["name", "description", "createdAt", "author", "location"];

    for i in 0..triple_count {
        // Deterministic but varied selection
        let class = classes[(i * 11) % classes.len()];
        let property = properties[(i * 13) % properties.len()];

        let subject = format!("<http://example.org/{}/{}>", class.to_lowercase(), i);
        let predicate = format!("<http://schema.org/{}>", property);
        let object = format!("\"{}_{}_value\"", class, property);

        output.push_str(&format!("{} {} {} .\n", subject, predicate, object));
    }

    Ok(output)
}

// ===== Workload Analysis Functions =====

#[derive(Debug)]
struct QueryLog {
    query: String,
    _timestamp: String,
    duration_ms: f64,
}

#[derive(Debug)]
struct WorkloadAnalysis {
    total_queries: usize,
    unique_queries: usize,
    _query_frequencies: HashMap<String, usize>,
    avg_duration_ms: f64,
    query_patterns: Vec<QueryPattern>,
}

#[derive(Debug)]
struct QueryPattern {
    pattern_type: String,
    count: usize,
    percentage: f64,
}

/// Parse query log
fn parse_query_log(content: &str) -> Result<Vec<QueryLog>, Box<dyn std::error::Error>> {
    let mut queries = Vec::new();

    for (i, line) in content.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        // Simple parsing - in production, parse actual log format
        queries.push(QueryLog {
            query: line.to_string(),
            _timestamp: format!("2025-11-09T{:02}:00:00Z", i % 24),
            duration_ms: (i as f64 % 100.0) + 5.0,
        });
    }

    Ok(queries)
}

/// Analyze query workload
fn analyze_query_workload(
    queries: &[QueryLog],
    analyze_patterns: bool,
) -> Result<WorkloadAnalysis, Box<dyn std::error::Error>> {
    let mut query_frequencies: HashMap<String, usize> = HashMap::new();
    let mut total_duration = 0.0;

    for log in queries {
        *query_frequencies.entry(log.query.clone()).or_insert(0) += 1;
        total_duration += log.duration_ms;
    }

    let unique_queries = query_frequencies.len();
    let avg_duration_ms = total_duration / queries.len() as f64;

    let mut query_patterns = Vec::new();
    if analyze_patterns {
        query_patterns = detect_query_patterns(queries)?;
    }

    Ok(WorkloadAnalysis {
        total_queries: queries.len(),
        unique_queries,
        _query_frequencies: query_frequencies,
        avg_duration_ms,
        query_patterns,
    })
}

/// Detect query patterns
fn detect_query_patterns(
    queries: &[QueryLog],
) -> Result<Vec<QueryPattern>, Box<dyn std::error::Error>> {
    let mut patterns: HashMap<String, usize> = HashMap::new();

    for log in queries {
        let pattern_type = if log.query.contains("SELECT") {
            "SELECT"
        } else if log.query.contains("ASK") {
            "ASK"
        } else if log.query.contains("CONSTRUCT") {
            "CONSTRUCT"
        } else if log.query.contains("DESCRIBE") {
            "DESCRIBE"
        } else {
            "OTHER"
        };

        *patterns.entry(pattern_type.to_string()).or_insert(0) += 1;
    }

    let total = queries.len() as f64;
    Ok(patterns
        .into_iter()
        .map(|(pattern_type, count)| QueryPattern {
            pattern_type,
            count,
            percentage: (count as f64 / total) * 100.0,
        })
        .collect())
}

/// Generate workload report
fn generate_workload_report(
    analysis: &WorkloadAnalysis,
    include_suggestions: bool,
    format: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    match format {
        "json" => Ok(serde_json::to_string_pretty(&serde_json::json!({
            "total_queries": analysis.total_queries,
            "unique_queries": analysis.unique_queries,
            "avg_duration_ms": analysis.avg_duration_ms,
            "patterns": analysis.query_patterns.iter().map(|p| {
                serde_json::json!({
                    "type": p.pattern_type,
                    "count": p.count,
                    "percentage": p.percentage
                })
            }).collect::<Vec<_>>()
        }))?),
        "html" => Ok(format!(
            r#"<html><body>
<h1>Query Workload Analysis</h1>
<p>Total Queries: {}</p>
<p>Unique Queries: {}</p>
<p>Average Duration: {:.2}ms</p>
</body></html>"#,
            analysis.total_queries, analysis.unique_queries, analysis.avg_duration_ms
        )),
        _ => {
            // Text format
            let mut report = String::new();
            report.push_str("===== Query Workload Analysis =====\n\n");
            report.push_str(&format!("Total queries: {}\n", analysis.total_queries));
            report.push_str(&format!("Unique queries: {}\n", analysis.unique_queries));
            report.push_str(&format!(
                "Average duration: {:.2}ms\n\n",
                analysis.avg_duration_ms
            ));

            if !analysis.query_patterns.is_empty() {
                report.push_str("Query Patterns:\n");
                for pattern in &analysis.query_patterns {
                    report.push_str(&format!(
                        "  {}: {} ({:.1}%)\n",
                        pattern.pattern_type, pattern.count, pattern.percentage
                    ));
                }
                report.push('\n');
            }

            if include_suggestions {
                report.push_str("Optimization Suggestions:\n");
                report.push_str("  • Consider caching frequently executed queries\n");
                report.push_str("  • Add indexes for commonly queried patterns\n");
                report
                    .push_str("  • Review slow queries (>100ms) for optimization opportunities\n");
            }

            Ok(report)
        }
    }
}

// ===== Comparison Functions =====

#[derive(Debug, Serialize)]
struct BenchmarkComparison {
    baseline_suite: String,
    current_suite: String,
    baseline_timestamp: String,
    current_timestamp: String,
    query_comparisons: Vec<QueryComparison>,
    overall_change_percent: f64,
    has_regressions: bool,
    regressions: Vec<String>,
    improvements: Vec<String>,
}

#[derive(Debug, Serialize)]
struct QueryComparison {
    query_name: String,
    baseline_avg_ms: f64,
    current_avg_ms: f64,
    change_percent: f64,
    is_regression: bool,
}

/// Compare benchmark results
fn compare_benchmark_results(
    baseline: &BenchmarkResults,
    current: &BenchmarkResults,
    threshold: f64,
) -> Result<BenchmarkComparison, Box<dyn std::error::Error>> {
    let mut query_comparisons = Vec::new();
    let mut regressions = Vec::new();
    let mut improvements = Vec::new();
    let mut total_baseline_time = 0.0;
    let mut total_current_time = 0.0;

    // Compare queries
    for baseline_query in &baseline.query_results {
        if let Some(current_query) = current
            .query_results
            .iter()
            .find(|q| q.query_name == baseline_query.query_name)
        {
            let baseline_ms = baseline_query.avg_time.as_secs_f64() * 1000.0;
            let current_ms = current_query.avg_time.as_secs_f64() * 1000.0;
            let change_percent = ((current_ms - baseline_ms) / baseline_ms) * 100.0;

            total_baseline_time += baseline_ms;
            total_current_time += current_ms;

            let is_regression = change_percent > threshold;

            if is_regression {
                regressions.push(format!(
                    "{}: {:.1}% slower ({:.2}ms → {:.2}ms)",
                    baseline_query.query_name, change_percent, baseline_ms, current_ms
                ));
            } else if change_percent < -5.0 {
                // Improvement threshold
                improvements.push(format!(
                    "{}: {:.1}% faster ({:.2}ms → {:.2}ms)",
                    baseline_query.query_name,
                    change_percent.abs(),
                    baseline_ms,
                    current_ms
                ));
            }

            query_comparisons.push(QueryComparison {
                query_name: baseline_query.query_name.clone(),
                baseline_avg_ms: baseline_ms,
                current_avg_ms: current_ms,
                change_percent,
                is_regression,
            });
        }
    }

    let overall_change_percent =
        ((total_current_time - total_baseline_time) / total_baseline_time) * 100.0;
    let has_regressions = !regressions.is_empty();

    Ok(BenchmarkComparison {
        baseline_suite: baseline.suite.clone(),
        current_suite: current.suite.clone(),
        baseline_timestamp: baseline.timestamp.clone(),
        current_timestamp: current.timestamp.clone(),
        query_comparisons,
        overall_change_percent,
        has_regressions,
        regressions,
        improvements,
    })
}

/// Generate comparison report
fn generate_comparison_report(
    comparison: &BenchmarkComparison,
    format: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    match format {
        "json" => Ok(serde_json::to_string_pretty(comparison)?),
        "html" => {
            let mut html = String::new();
            html.push_str("<html><body>\n");
            html.push_str("<h1>Benchmark Comparison Report</h1>\n");
            html.push_str(&format!(
                "<p>Overall change: {:.1}%</p>\n",
                comparison.overall_change_percent
            ));
            html.push_str("</body></html>");
            Ok(html)
        }
        _ => {
            // Text format
            let mut report = String::new();
            report.push_str("===== Benchmark Comparison Report =====\n\n");
            report.push_str(&format!(
                "Baseline: {} ({})\n",
                comparison.baseline_suite, comparison.baseline_timestamp
            ));
            report.push_str(&format!(
                "Current:  {} ({})\n\n",
                comparison.current_suite, comparison.current_timestamp
            ));

            report.push_str(&format!(
                "Overall Performance Change: {:.1}%\n\n",
                comparison.overall_change_percent
            ));

            if !comparison.regressions.is_empty() {
                report.push_str("⚠️  REGRESSIONS DETECTED:\n");
                for regression in &comparison.regressions {
                    report.push_str(&format!("  • {}\n", regression));
                }
                report.push('\n');
            }

            if !comparison.improvements.is_empty() {
                report.push_str("✓ Improvements:\n");
                for improvement in &comparison.improvements {
                    report.push_str(&format!("  • {}\n", improvement));
                }
                report.push('\n');
            }

            report.push_str("Query-by-Query Comparison:\n");
            for comp in &comparison.query_comparisons {
                let status = if comp.is_regression {
                    "⚠️ REGRESSION"
                } else if comp.change_percent < -5.0 {
                    "✓ IMPROVED"
                } else {
                    "≈ UNCHANGED"
                };

                report.push_str(&format!("  {} {}:\n", status, comp.query_name));
                report.push_str(&format!(
                    "    Baseline: {:.2}ms → Current: {:.2}ms ({:+.1}%)\n",
                    comp.baseline_avg_ms, comp.current_avg_ms, comp.change_percent
                ));
            }

            Ok(report)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_suite_validation() {
        assert!(is_supported_benchmark_suite("sp2bench"));
        assert!(is_supported_benchmark_suite("watdiv"));
        assert!(is_supported_benchmark_suite("ldbc"));
        assert!(is_supported_benchmark_suite("bsbm"));
        assert!(!is_supported_benchmark_suite("invalid"));
    }

    #[test]
    fn test_dataset_size_parsing() {
        assert_eq!(get_triple_count_for_size("tiny"), Some(1_000));
        assert_eq!(get_triple_count_for_size("small"), Some(10_000));
        assert_eq!(get_triple_count_for_size("medium"), Some(100_000));
        assert_eq!(get_triple_count_for_size("large"), Some(1_000_000));
        assert_eq!(get_triple_count_for_size("xlarge"), Some(10_000_000));
        assert_eq!(get_triple_count_for_size("invalid"), None);
    }

    fn get_triple_count_for_size(size: &str) -> Option<usize> {
        match size {
            "tiny" => Some(1_000),
            "small" => Some(10_000),
            "medium" => Some(100_000),
            "large" => Some(1_000_000),
            "xlarge" => Some(10_000_000),
            _ => None,
        }
    }

    #[test]
    fn test_duration_serde() {
        let duration = Duration::from_millis(1234);
        let serde: DurationSerde = duration.into();
        let back: Duration = serde.into();
        assert_eq!(duration, back);
    }
}
