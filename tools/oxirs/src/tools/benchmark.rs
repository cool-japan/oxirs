//! SPARQL Benchmarking Suite
//!
//! Comprehensive benchmarking for SPARQL query engines including SP2Bench,
//! WatDiv, LDBC, and custom benchmarks.

use super::{ToolResult, ToolStats};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Benchmark suite types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkSuite {
    SP2Bench,
    WatDiv,
    LDBC,
    Custom,
}

impl BenchmarkSuite {
    pub fn name(&self) -> &str {
        match self {
            BenchmarkSuite::SP2Bench => "SP2Bench",
            BenchmarkSuite::WatDiv => "WatDiv",
            BenchmarkSuite::LDBC => "LDBC",
            BenchmarkSuite::Custom => "Custom",
        }
    }

    pub fn description(&self) -> &str {
        match self {
            BenchmarkSuite::SP2Bench => "SPARQL Performance Benchmark (DBLP-like data)",
            BenchmarkSuite::WatDiv => "Waterloo SPARQL Diversity Test Suite",
            BenchmarkSuite::LDBC => "Linked Data Benchmark Council",
            BenchmarkSuite::Custom => "Custom benchmark queries",
        }
    }
}

/// Configuration for benchmark execution
pub struct BenchmarkConfig {
    pub suite: BenchmarkSuite,
    pub data_path: Option<PathBuf>,
    pub queries_path: Option<PathBuf>,
    pub warmup_runs: usize,
    pub benchmark_runs: usize,
    pub timeout: Duration,
    pub output_format: String,
}

/// Benchmark query
#[derive(Debug, Clone)]
pub struct BenchmarkQuery {
    pub id: String,
    pub name: String,
    pub query: String,
    pub category: String,
}

/// Benchmark result for a single query
#[derive(Debug, Clone)]
pub struct QueryBenchmarkResult {
    pub query_id: String,
    pub query_name: String,
    pub runs: Vec<Duration>,
    pub mean: Duration,
    pub median: Duration,
    pub min: Duration,
    pub max: Duration,
    pub std_dev: Duration,
    pub timeout: bool,
    pub error: Option<String>,
}

/// Benchmark suite results
#[derive(Debug)]
pub struct BenchmarkResults {
    pub suite: BenchmarkSuite,
    pub total_queries: usize,
    pub successful_queries: usize,
    pub failed_queries: usize,
    pub total_duration: Duration,
    pub query_results: Vec<QueryBenchmarkResult>,
}

/// Run benchmark suite
pub async fn run(config: BenchmarkConfig) -> ToolResult {
    let mut stats = ToolStats::new();

    println!("OxiRS SPARQL Benchmark Suite");
    println!("============================\n");

    println!("Suite: {}", config.suite.name());
    println!("Description: {}", config.suite.description());
    println!("Warmup runs: {}", config.warmup_runs);
    println!("Benchmark runs: {}", config.benchmark_runs);
    println!("Timeout: {:?}\n", config.timeout);

    // Load benchmark queries
    let queries = load_benchmark_queries(&config)?;
    println!("Loaded {} benchmark queries\n", queries.len());

    // Execute benchmarks
    let start = Instant::now();
    let results = execute_benchmarks(&queries, &config).await?;
    let total_duration = start.elapsed();

    // Display results
    display_results(&results, &config.output_format)?;

    println!("\n=== Summary ===");
    println!("Total queries: {}", results.total_queries);
    println!("Successful: {}", results.successful_queries);
    println!("Failed: {}", results.failed_queries);
    println!("Total time: {:.2}s", total_duration.as_secs_f64());

    stats.items_processed = results.total_queries;
    stats.finish();
    stats.print_summary("Benchmark");

    Ok(())
}

/// Load benchmark queries based on suite
fn load_benchmark_queries(config: &BenchmarkConfig) -> ToolResult<Vec<BenchmarkQuery>> {
    match config.suite {
        BenchmarkSuite::SP2Bench => load_sp2bench_queries(),
        BenchmarkSuite::WatDiv => load_watdiv_queries(),
        BenchmarkSuite::LDBC => load_ldbc_queries(),
        BenchmarkSuite::Custom => {
            if let Some(ref path) = config.queries_path {
                load_custom_queries(path)
            } else {
                Err("Custom benchmark requires --queries-path".into())
            }
        }
    }
}

/// Load SP2Bench queries
fn load_sp2bench_queries() -> ToolResult<Vec<BenchmarkQuery>> {
    let mut queries = Vec::new();

    // Q1: Simple triple pattern
    queries.push(BenchmarkQuery {
        id: "SP2B-Q1".to_string(),
        name: "Simple Triple Pattern".to_string(),
        query: "SELECT ?yr WHERE { ?journal rdf:type bench:Journal . ?journal dc:title \"Journal 1 (1940)\"^^xsd:string . ?journal dcterms:issued ?yr }".to_string(),
        category: "Simple".to_string(),
    });

    // Q2: Triple patterns with FILTER
    queries.push(BenchmarkQuery {
        id: "SP2B-Q2".to_string(),
        name: "Triple Patterns with FILTER".to_string(),
        query: "SELECT ?inproc ?author ?booktitle ?title ?proc ?ee ?page ?url ?yr ?abstract WHERE { ?inproc rdf:type bench:Inproceedings . ?inproc dc:creator ?author . ?inproc bench:booktitle ?booktitle . ?inproc dc:title ?title . ?inproc dcterms:partOf ?proc . ?inproc rdfs:seeAlso ?ee . ?inproc swrc:pages ?page . ?inproc foaf:homepage ?url . ?inproc dcterms:issued ?yr FILTER(?yr > 1970) }".to_string(),
        category: "Filter".to_string(),
    });

    // Q3: Join of multiple triple patterns
    queries.push(BenchmarkQuery {
        id: "SP2B-Q3a".to_string(),
        name: "Join of Multiple Patterns".to_string(),
        query:
            "SELECT ?article WHERE { ?article rdf:type bench:Article . ?article ?property ?value }"
                .to_string(),
        category: "Join".to_string(),
    });

    // Q4: OPTIONAL patterns
    queries.push(BenchmarkQuery {
        id: "SP2B-Q4".to_string(),
        name: "OPTIONAL Patterns".to_string(),
        query: "SELECT ?name ?nameTitle WHERE { ?article rdf:type bench:Article . ?article dc:creator ?author . ?author foaf:name ?name . OPTIONAL { ?author dc:title ?nameTitle } }".to_string(),
        category: "Optional".to_string(),
    });

    // Q5: UNION
    queries.push(BenchmarkQuery {
        id: "SP2B-Q5a".to_string(),
        name: "UNION Query".to_string(),
        query: "SELECT DISTINCT ?person ?name WHERE { ?article rdf:type bench:Article . ?article dc:creator ?person . ?inproc rdf:type bench:Inproceedings . ?inproc dc:creator ?person2 . ?person foaf:name ?name . ?person2 foaf:name ?name2 FILTER(?name = ?name2) }".to_string(),
        category: "Union".to_string(),
    });

    Ok(queries)
}

/// Load WatDiv queries
fn load_watdiv_queries() -> ToolResult<Vec<BenchmarkQuery>> {
    let mut queries = Vec::new();

    // Linear query (L)
    queries.push(BenchmarkQuery {
        id: "WatDiv-L1".to_string(),
        name: "Linear Query 1".to_string(),
        query: "SELECT ?v0 ?v1 ?v2 ?v3 WHERE { ?v0 wsdbm:follows ?v1 . ?v1 wsdbm:follows ?v2 . ?v2 wsdbm:likes ?v3 }".to_string(),
        category: "Linear".to_string(),
    });

    // Star query (S)
    queries.push(BenchmarkQuery {
        id: "WatDiv-S1".to_string(),
        name: "Star Query 1".to_string(),
        query: "SELECT ?v0 ?v1 ?v2 ?v3 ?v4 WHERE { ?v0 wsdbm:follows ?v1 . ?v0 wsdbm:likes ?v2 . ?v0 wsdbm:friendOf ?v3 . ?v0 dc:Location ?v4 }".to_string(),
        category: "Star".to_string(),
    });

    // Snowflake query (F)
    queries.push(BenchmarkQuery {
        id: "WatDiv-F1".to_string(),
        name: "Snowflake Query 1".to_string(),
        query: "SELECT ?v0 ?v1 ?v2 ?v3 ?v4 ?v5 WHERE { ?v0 wsdbm:follows ?v1 . ?v1 wsdbm:likes ?v2 . ?v0 wsdbm:friendOf ?v3 . ?v3 wsdbm:likes ?v4 . ?v0 dc:Location ?v5 }".to_string(),
        category: "Snowflake".to_string(),
    });

    // Complex query (C)
    queries.push(BenchmarkQuery {
        id: "WatDiv-C1".to_string(),
        name: "Complex Query 1".to_string(),
        query: "SELECT ?v0 ?v1 ?v2 ?v3 WHERE { ?v0 wsdbm:follows ?v1 . ?v1 wsdbm:likes ?v2 . ?v1 wsdbm:friendOf ?v3 . ?v3 wsdbm:likes ?v2 }".to_string(),
        category: "Complex".to_string(),
    });

    Ok(queries)
}

/// Load LDBC queries
fn load_ldbc_queries() -> ToolResult<Vec<BenchmarkQuery>> {
    let mut queries = Vec::new();

    // LDBC SNB Interactive Complex Read 1
    queries.push(BenchmarkQuery {
        id: "LDBC-IC1".to_string(),
        name: "Interactive Complex 1".to_string(),
        query: "SELECT ?personId ?firstName ?lastName ?birthday ?locationIP ?browserUsed ?cityId WHERE { ?person rdf:type snvoc:Person . ?person snvoc:id ?personId . ?person snvoc:firstName ?firstName . ?person snvoc:lastName ?lastName . ?person snvoc:birthday ?birthday . ?person snvoc:locationIP ?locationIP . ?person snvoc:browserUsed ?browserUsed . ?person snvoc:isLocatedIn ?city . ?city snvoc:id ?cityId }".to_string(),
        category: "Complex".to_string(),
    });

    // LDBC SNB Interactive Short 1
    queries.push(BenchmarkQuery {
        id: "LDBC-IS1".to_string(),
        name: "Interactive Short 1".to_string(),
        query: "SELECT ?firstName ?lastName ?birthday WHERE { ?person rdf:type snvoc:Person . ?person snvoc:id ?personId . ?person snvoc:firstName ?firstName . ?person snvoc:lastName ?lastName . ?person snvoc:birthday ?birthday }".to_string(),
        category: "Short".to_string(),
    });

    Ok(queries)
}

/// Load custom queries from file
fn load_custom_queries(path: &PathBuf) -> ToolResult<Vec<BenchmarkQuery>> {
    let content = std::fs::read_to_string(path)?;
    let mut queries = Vec::new();

    // Parse simple format: each query separated by "---"
    let query_strs: Vec<&str> = content.split("---").collect();

    for (i, query_str) in query_strs.iter().enumerate() {
        let query_str = query_str.trim();
        if !query_str.is_empty() {
            queries.push(BenchmarkQuery {
                id: format!("Custom-Q{}", i + 1),
                name: format!("Custom Query {}", i + 1),
                query: query_str.to_string(),
                category: "Custom".to_string(),
            });
        }
    }

    Ok(queries)
}

/// Execute benchmark queries
async fn execute_benchmarks(
    queries: &[BenchmarkQuery],
    config: &BenchmarkConfig,
) -> ToolResult<BenchmarkResults> {
    let mut query_results = Vec::new();
    let mut successful = 0;
    let mut failed = 0;

    for (i, query) in queries.iter().enumerate() {
        println!(
            "Running benchmark {}/{}: {}",
            i + 1,
            queries.len(),
            query.name
        );

        // Warmup runs
        if config.warmup_runs > 0 {
            print!("  Warmup ({} runs)...", config.warmup_runs);
            for _ in 0..config.warmup_runs {
                let _ = execute_query(&query.query, config.timeout).await;
            }
            println!(" done");
        }

        // Benchmark runs
        let mut run_times = Vec::new();
        let mut error = None;
        let mut timeout_occurred = false;

        print!("  Benchmarking ({} runs): ", config.benchmark_runs);
        for run in 0..config.benchmark_runs {
            match execute_query(&query.query, config.timeout).await {
                Ok(duration) => {
                    run_times.push(duration);
                    print!(".");
                }
                Err(e) => {
                    if e.to_string().contains("timeout") {
                        timeout_occurred = true;
                        print!("T");
                    } else {
                        error = Some(e.to_string());
                        print!("E");
                    }
                }
            }
            std::io::Write::flush(&mut std::io::stdout()).unwrap_or(());

            // If first run times out or errors, skip remaining runs
            if run == 0 && (timeout_occurred || error.is_some()) {
                println!(" (skipping remaining runs)");
                break;
            }
        }

        if !run_times.is_empty() {
            println!(" done");

            let result = calculate_query_statistics(&query.id, &query.name, &run_times);

            println!(
                "  Mean: {:?}, Median: {:?}, Min: {:?}, Max: {:?}",
                result.mean, result.median, result.min, result.max
            );

            successful += 1;
            query_results.push(result);
        } else {
            println!();
            println!(
                "  Failed: {:?}",
                error.clone().or(Some("timeout".to_string()))
            );

            failed += 1;
            query_results.push(QueryBenchmarkResult {
                query_id: query.id.clone(),
                query_name: query.name.clone(),
                runs: vec![],
                mean: Duration::ZERO,
                median: Duration::ZERO,
                min: Duration::ZERO,
                max: Duration::ZERO,
                std_dev: Duration::ZERO,
                timeout: timeout_occurred,
                error,
            });
        }

        println!();
    }

    Ok(BenchmarkResults {
        suite: config.suite,
        total_queries: queries.len(),
        successful_queries: successful,
        failed_queries: failed,
        total_duration: Duration::ZERO, // Will be set by caller
        query_results,
    })
}

/// Execute a single query and measure time
async fn execute_query(query: &str, timeout: Duration) -> ToolResult<Duration> {
    let start = Instant::now();

    // Simulate query execution
    // In a real implementation, this would execute the SPARQL query
    let simulation_time = std::cmp::min(
        query.len() as u64 / 100, // Simulate based on query length
        50,                       // Max 50ms
    );

    std::thread::sleep(Duration::from_millis(simulation_time));

    let duration = start.elapsed();

    if duration > timeout {
        return Err("Query timeout".into());
    }

    Ok(duration)
}

/// Calculate statistics from query run times
fn calculate_query_statistics(
    query_id: &str,
    query_name: &str,
    run_times: &[Duration],
) -> QueryBenchmarkResult {
    let mut sorted_times = run_times.to_vec();
    sorted_times.sort();

    let mean = sorted_times.iter().sum::<Duration>() / sorted_times.len() as u32;

    let median = if sorted_times.len() % 2 == 0 {
        let mid = sorted_times.len() / 2;
        (sorted_times[mid - 1] + sorted_times[mid]) / 2
    } else {
        sorted_times[sorted_times.len() / 2]
    };

    let min = *sorted_times.first().unwrap();
    let max = *sorted_times.last().unwrap();

    // Calculate standard deviation
    let variance: f64 = sorted_times
        .iter()
        .map(|&t| {
            let diff = t.as_secs_f64() - mean.as_secs_f64();
            diff * diff
        })
        .sum::<f64>()
        / sorted_times.len() as f64;
    let std_dev = Duration::from_secs_f64(variance.sqrt());

    QueryBenchmarkResult {
        query_id: query_id.to_string(),
        query_name: query_name.to_string(),
        runs: run_times.to_vec(),
        mean,
        median,
        min,
        max,
        std_dev,
        timeout: false,
        error: None,
    }
}

/// Display benchmark results
fn display_results(results: &BenchmarkResults, format: &str) -> ToolResult<()> {
    println!("\n=== Benchmark Results ===\n");

    match format {
        "table" => display_table_results(results),
        "json" => display_json_results(results),
        "csv" => display_csv_results(results),
        _ => display_table_results(results),
    }
}

/// Display results as table
fn display_table_results(results: &BenchmarkResults) -> ToolResult<()> {
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "Query", "Mean", "Median", "Min", "Max", "StdDev"
    );
    println!("{}", "-".repeat(80));

    for result in &results.query_results {
        if result.timeout {
            println!("{:<20} {:>12}", result.query_id, "TIMEOUT");
        } else if result.error.is_some() {
            println!("{:<20} {:>12}", result.query_id, "ERROR");
        } else {
            println!(
                "{:<20} {:>10.2?} {:>10.2?} {:>10.2?} {:>10.2?} {:>8.2?}",
                result.query_id, result.mean, result.median, result.min, result.max, result.std_dev,
            );
        }
    }

    Ok(())
}

/// Display results as JSON
fn display_json_results(results: &BenchmarkResults) -> ToolResult<()> {
    println!("{{");
    println!("  \"suite\": \"{}\",", results.suite.name());
    println!("  \"total_queries\": {},", results.total_queries);
    println!("  \"successful\": {},", results.successful_queries);
    println!("  \"failed\": {},", results.failed_queries);
    println!("  \"queries\": [");

    for (i, result) in results.query_results.iter().enumerate() {
        println!("    {{");
        println!("      \"id\": \"{}\",", result.query_id);
        println!("      \"name\": \"{}\",", result.query_name);
        println!("      \"mean_ms\": {},", result.mean.as_millis());
        println!("      \"median_ms\": {},", result.median.as_millis());
        println!("      \"min_ms\": {},", result.min.as_millis());
        println!("      \"max_ms\": {},", result.max.as_millis());
        println!("      \"std_dev_ms\": {}", result.std_dev.as_millis());
        print!("    }}");
        if i < results.query_results.len() - 1 {
            println!(",");
        } else {
            println!();
        }
    }

    println!("  ]");
    println!("}}");

    Ok(())
}

/// Display results as CSV
fn display_csv_results(results: &BenchmarkResults) -> ToolResult<()> {
    println!("query_id,query_name,mean_ms,median_ms,min_ms,max_ms,std_dev_ms");

    for result in &results.query_results {
        println!(
            "{},{},{},{},{},{},{}",
            result.query_id,
            result.query_name,
            result.mean.as_millis(),
            result.median.as_millis(),
            result.min.as_millis(),
            result.max.as_millis(),
            result.std_dev.as_millis(),
        );
    }

    Ok(())
}
