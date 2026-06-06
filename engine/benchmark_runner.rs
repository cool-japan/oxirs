//! Benchmark Runner Executable
//!
//! Command-line interface for running comprehensive OxiRS vs Apache Jena benchmarks

use anyhow::{Context, Result};
use clap::{Arg, Command, ArgMatches};
use std::path::PathBuf;
use std::time::Instant;

mod comprehensive_performance_benchmark;
use comprehensive_performance_benchmark::{
    BenchmarkCategory, BenchmarkRunner, ComprehensiveBenchmarkConfig, ComprehensiveBenchmarkSuite,
};

fn main() -> Result<()> {
    let app = Command::new("OxiRS Performance Benchmark Runner")
        .version("1.0.0")
        .author("OxiRS Development Team")
        .about("Comprehensive performance benchmarking suite comparing OxiRS vs Apache Jena")
        .arg(
            Arg::new("mode")
                .short('m')
                .long("mode")
                .value_name("MODE")
                .help("Benchmark mode: full, quick, category")
                .default_value("quick"),
        )
        .arg(
            Arg::new("category")
                .short('c')
                .long("category")
                .value_name("CATEGORY")
                .help("Specific category to benchmark (when mode=category)")
                .value_parser(["sparql", "parsing", "reasoning", "shacl", "vector", "scalability"]),
        )
        .arg(
            Arg::new("jena-path")
                .short('j')
                .long("jena-path")
                .value_name("PATH")
                .help("Path to Apache Jena installation")
                .default_value("~/work/jena"),
        )
        .arg(
            Arg::new("datasets-path")
                .short('d')
                .long("datasets-path")
                .value_name("PATH")
                .help("Path to test datasets directory")
                .default_value("./data"),
        )
        .arg(
            Arg::new("output-dir")
                .short('o')
                .long("output-dir")
                .value_name("PATH")
                .help("Output directory for benchmark results")
                .default_value("./benchmark_results"),
        )
        .arg(
            Arg::new("runs")
                .short('r')
                .long("runs")
                .value_name("COUNT")
                .help("Number of benchmark runs per test")
                .default_value("10"),
        )
        .arg(
            Arg::new("warmup")
                .short('w')
                .long("warmup")
                .value_name("COUNT")
                .help("Number of warmup runs before measurement")
                .default_value("3"),
        )
        .arg(
            Arg::new("timeout")
                .short('t')
                .long("timeout")
                .value_name("SECONDS")
                .help("Maximum time per benchmark in seconds")
                .default_value("300"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("no-memory")
                .long("no-memory")
                .help("Disable memory profiling")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("no-cpu")
                .long("no-cpu")
                .help("Disable CPU profiling")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .value_name("NUMBER")
                .help("Random seed for reproducible results")
                .default_value("42"),
        );

    let matches = app.get_matches();

    println!("🚀 OxiRS vs Apache Jena Performance Benchmark Suite");
    println!("===================================================");

    let start_time = Instant::now();

    let result = match matches.get_one::<String>("mode").expect("mode has a default value").as_str() {
        "full" => run_full_benchmarks(&matches),
        "quick" => run_quick_benchmarks(&matches),
        "category" => run_category_benchmarks(&matches),
        mode => {
            eprintln!("❌ Unknown benchmark mode: {}", mode);
            std::process::exit(1);
        }
    };

    match result {
        Ok(comparison_count) => {
            let total_time = start_time.elapsed();
            println!("\n✅ Benchmark suite completed successfully!");
            println!("📊 Total comparisons: {}", comparison_count);
            println!("⏱️  Total time: {:.2} seconds", total_time.as_secs_f64());
            println!("📁 Results saved to: {}", matches.get_one::<String>("output-dir").expect("output-dir has a default value"));

            println!("\n📋 Next steps:");
            println!("   1. Review the generated HTML report: benchmark_report.html");
            println!("   2. Check the markdown summary: BENCHMARK_SUMMARY.md");
            println!("   3. Analyze the CSV data: benchmark_results.csv");
        }
        Err(e) => {
            eprintln!("❌ Benchmark suite failed: {}", e);
            eprintln!("🔍 Check the error details above and verify your configuration");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn run_full_benchmarks(matches: &ArgMatches) -> Result<usize> {
    println!("🏃 Running comprehensive benchmark suite...");
    println!("⚠️  This may take 15-30 minutes depending on your system");

    let config = build_config(matches)?;
    let mut suite = ComprehensiveBenchmarkSuite::new(config);

    println!("📥 Loading standard test suite...");
    suite.load_standard_tests()
        .context("Failed to load standard tests")?;

    println!("🎯 Running {} benchmark tests", suite.tests.len());
    let results = suite.run_all_benchmarks()
        .context("Failed to run benchmarks")?;

    Ok(results.len())
}

fn run_quick_benchmarks(matches: &ArgMatches) -> Result<usize> {
    println!("⚡ Running quick benchmark suite...");
    println!("🕐 This should complete in 2-5 minutes");

    let results = BenchmarkRunner::run_quick_benchmarks()
        .context("Failed to run quick benchmarks")?;

    Ok(results.len())
}

fn run_category_benchmarks(matches: &ArgMatches) -> Result<usize> {
    let category_str = matches.get_one::<String>("category")
        .ok_or_else(|| anyhow::anyhow!("Category must be specified when mode=category"))?;

    let category = parse_category(category_str)?;

    println!("🎯 Running {:?} category benchmarks...", category);

    let results = BenchmarkRunner::run_category_benchmarks(category)
        .context("Failed to run category benchmarks")?;

    Ok(results.len())
}

fn build_config(matches: &ArgMatches) -> Result<ComprehensiveBenchmarkConfig> {
    let runs: usize = matches.get_one::<String>("runs").expect("runs has a default value").parse()
        .context("Invalid runs count")?;
    let warmup: usize = matches.get_one::<String>("warmup").expect("warmup has a default value").parse()
        .context("Invalid warmup count")?;
    let timeout: u64 = matches.get_one::<String>("timeout").expect("timeout has a default value").parse()
        .context("Invalid timeout")?;
    let seed: u64 = matches.get_one::<String>("seed").expect("seed has a default value").parse()
        .context("Invalid seed")?;

    let jena_path = PathBuf::from(matches.get_one::<String>("jena-path").expect("jena-path has a default value"));
    let datasets_path = PathBuf::from(matches.get_one::<String>("datasets-path").expect("datasets-path has a default value"));
    let output_dir = PathBuf::from(matches.get_one::<String>("output-dir").expect("output-dir has a default value"));

    Ok(ComprehensiveBenchmarkConfig {
        warmup_runs: warmup,
        benchmark_runs: runs,
        max_duration_secs: timeout,
        profile_memory: !matches.get_flag("no-memory"),
        profile_cpu: !matches.get_flag("no-cpu"),
        profile_io: true,
        jena_path,
        datasets_path,
        output_dir,
        verbose: matches.get_flag("verbose"),
        random_seed: Some(seed),
        jena_version: None,
    })
}

fn parse_category(category_str: &str) -> Result<BenchmarkCategory> {
    match category_str.to_lowercase().as_str() {
        "sparql" => Ok(BenchmarkCategory::SparqlQuery),
        "parsing" => Ok(BenchmarkCategory::RdfParsing),
        "reasoning" => Ok(BenchmarkCategory::RuleBasedReasoning),
        "shacl" => Ok(BenchmarkCategory::ShaclValidation),
        "vector" => Ok(BenchmarkCategory::VectorSearch),
        "scalability" => Ok(BenchmarkCategory::Scalability),
        _ => Err(anyhow::anyhow!("Unknown category: {}", category_str)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_parsing() {
        assert!(matches!(parse_category("sparql").unwrap(), BenchmarkCategory::SparqlQuery));
        assert!(matches!(parse_category("SPARQL").unwrap(), BenchmarkCategory::SparqlQuery));
        assert!(matches!(parse_category("parsing").unwrap(), BenchmarkCategory::RdfParsing));
        assert!(matches!(parse_category("reasoning").unwrap(), BenchmarkCategory::RuleBasedReasoning));
        assert!(matches!(parse_category("shacl").unwrap(), BenchmarkCategory::ShaclValidation));
        assert!(matches!(parse_category("vector").unwrap(), BenchmarkCategory::VectorSearch));
        assert!(matches!(parse_category("scalability").unwrap(), BenchmarkCategory::Scalability));

        assert!(parse_category("invalid").is_err());
    }
}