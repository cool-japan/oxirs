//! Benchmark comparison and regression detection tool
//!
//! This tool compares benchmark results across versions to detect performance
//! regressions and improvements.
//!
//! # Usage
//!
//! ```bash
//! # Compare latest benchmarks with previous version
//! cargo run --bin compare_benchmarks -- v0.1.0 v0.1.1
//!
//! # Generate performance report
//! cargo run --bin compare_benchmarks -- --report
//!
//! # Check for regressions (exits with error if found)
//! cargo run --bin compare_benchmarks -- --check-regression --threshold 10.0
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    pub timestamp: u64,
    pub version: String,
    pub rust_version: String,
    pub platform: String,
    pub benchmark_name: String,
    pub mean_time_ns: f64,
    pub std_dev_ns: f64,
    pub throughput_ops_per_sec: Option<f64>,
    pub sample_count: usize,
}

#[derive(Debug)]
pub struct BenchmarkComparison {
    pub benchmark_name: String,
    pub old_mean_ns: f64,
    pub new_mean_ns: f64,
    pub change_percent: f64,
    pub is_regression: bool,
    pub is_improvement: bool,
}

impl BenchmarkComparison {
    pub fn new(name: String, old_mean: f64, new_mean: f64, threshold: f64) -> Self {
        let change_percent = ((new_mean - old_mean) / old_mean) * 100.0;
        let is_regression = change_percent > threshold;
        let is_improvement = change_percent < -threshold;

        Self {
            benchmark_name: name,
            old_mean_ns: old_mean,
            new_mean_ns: new_mean,
            change_percent,
            is_regression,
            is_improvement,
        }
    }
}

/// Load the latest benchmark results
pub fn load_latest_benchmarks() -> std::io::Result<Vec<BenchmarkRecord>> {
    let results_dir = Path::new("target/benchmark_history");
    let latest_path = results_dir.join("latest.json");

    if !latest_path.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No benchmark results found. Run benchmarks first with: cargo bench",
        ));
    }

    let content = fs::read_to_string(latest_path)?;
    let records: Vec<BenchmarkRecord> = serde_json::from_str(&content)?;
    Ok(records)
}

/// Load benchmark results for a specific version
pub fn load_version_benchmarks(version: &str) -> std::io::Result<Vec<BenchmarkRecord>> {
    let results_dir = Path::new("target/benchmark_history");

    let mut matching_files: Vec<_> = fs::read_dir(results_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with(&format!("benchmarks_{}_", version))
        })
        .collect();

    matching_files.sort_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()));

    if let Some(latest) = matching_files.last() {
        let content = fs::read_to_string(latest.path())?;
        let records: Vec<BenchmarkRecord> = serde_json::from_str(&content)?;
        Ok(records)
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("No benchmark results found for version {}", version),
        ))
    }
}

/// Compare two sets of benchmark results
pub fn compare_benchmarks(
    old_results: &[BenchmarkRecord],
    new_results: &[BenchmarkRecord],
    threshold: f64,
) -> Vec<BenchmarkComparison> {
    let mut old_map: HashMap<String, f64> = HashMap::new();
    for record in old_results {
        old_map.insert(record.benchmark_name.clone(), record.mean_time_ns);
    }

    let mut comparisons = Vec::new();

    for new_record in new_results {
        if let Some(&old_mean) = old_map.get(&new_record.benchmark_name) {
            let comparison = BenchmarkComparison::new(
                new_record.benchmark_name.clone(),
                old_mean,
                new_record.mean_time_ns,
                threshold,
            );
            comparisons.push(comparison);
        }
    }

    comparisons.sort_by(|a, b| {
        b.change_percent
            .abs()
            .partial_cmp(&a.change_percent.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    comparisons
}

/// Print comparison report
pub fn print_comparison_report(
    comparisons: &[BenchmarkComparison],
    old_version: &str,
    new_version: &str,
) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         Benchmark Comparison Report                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Comparing: {} → {}                        ║",
        old_version, new_version
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let regressions: Vec<_> = comparisons.iter().filter(|c| c.is_regression).collect();
    let improvements: Vec<_> = comparisons.iter().filter(|c| c.is_improvement).collect();
    let stable: Vec<_> = comparisons
        .iter()
        .filter(|c| !c.is_regression && !c.is_improvement)
        .collect();

    // Print summary
    println!("📊 Summary:");
    println!("  Total benchmarks: {}", comparisons.len());
    println!("  ⚠️  Regressions: {}", regressions.len());
    println!("  ✅ Improvements: {}", improvements.len());
    println!("  ➖ Stable: {}", stable.len());
    println!();

    // Print regressions
    if !regressions.is_empty() {
        println!("⚠️  Performance Regressions:");
        println!("  ┌─────────────────────────────────────────────────────────┐");
        for comp in regressions {
            println!(
                "  │ {} │ {:.2}% slower │ {:.2}μs → {:.2}μs",
                comp.benchmark_name,
                comp.change_percent,
                comp.old_mean_ns / 1000.0,
                comp.new_mean_ns / 1000.0
            );
        }
        println!("  └─────────────────────────────────────────────────────────┘\n");
    }

    // Print improvements
    if !improvements.is_empty() {
        println!("✅ Performance Improvements:");
        println!("  ┌─────────────────────────────────────────────────────────┐");
        for comp in improvements {
            println!(
                "  │ {} │ {:.2}% faster │ {:.2}μs → {:.2}μs",
                comp.benchmark_name,
                comp.change_percent.abs(),
                comp.old_mean_ns / 1000.0,
                comp.new_mean_ns / 1000.0
            );
        }
        println!("  └─────────────────────────────────────────────────────────┘\n");
    }

    // Print all results table
    println!("📋 Detailed Results:");
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │ Benchmark Name              │ Old (μs) │ New (μs) │ Change %");
    println!("  ├─────────────────────────────────────────────────────────┤");
    for comp in comparisons {
        let symbol = if comp.is_regression {
            "⚠️ "
        } else if comp.is_improvement {
            "✅"
        } else {
            "  "
        };
        println!(
            "  │ {} {:30} │ {:8.2} │ {:8.2} │ {:+.2}%",
            symbol,
            &comp.benchmark_name[..comp.benchmark_name.len().min(30)],
            comp.old_mean_ns / 1000.0,
            comp.new_mean_ns / 1000.0,
            comp.change_percent
        );
    }
    println!("  └─────────────────────────────────────────────────────────┘\n");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <old_version> <new_version> [--threshold <percent>]",
            args[0]
        );
        eprintln!("   or: {} --report", args[0]);
        eprintln!("   or: {} --check-regression --threshold 10.0", args[0]);
        std::process::exit(1);
    }

    // Parse arguments
    let mut threshold = 5.0; // Default 5% threshold
    let mut check_regression = false;
    let mut report_mode = false;

    for i in 1..args.len() {
        match args[i].as_str() {
            "--threshold" => {
                if i + 1 < args.len() {
                    threshold = args[i + 1].parse()?;
                }
            }
            "--check-regression" => check_regression = true,
            "--report" => report_mode = true,
            _ => {}
        }
    }

    if report_mode {
        // Generate report from latest benchmarks
        let latest = load_latest_benchmarks()?;
        println!("Latest benchmark results:");
        for record in &latest {
            println!(
                "  {} - {:.2}μs (version {})",
                record.benchmark_name,
                record.mean_time_ns / 1000.0,
                record.version
            );
        }
        return Ok(());
    }

    // Compare versions
    if args.len() < 3 || args[1].starts_with("--") {
        eprintln!("Error: Please specify old and new versions");
        std::process::exit(1);
    }

    let old_version = &args[1];
    let new_version = &args[2];

    println!("Loading benchmark results...");
    let old_results = load_version_benchmarks(old_version)?;
    let new_results = load_version_benchmarks(new_version)?;

    let comparisons = compare_benchmarks(&old_results, &new_results, threshold);
    print_comparison_report(&comparisons, old_version, new_version);

    // Check for regressions if requested
    if check_regression {
        let regression_count = comparisons.iter().filter(|c| c.is_regression).count();
        if regression_count > 0 {
            eprintln!("\n❌ Found {} performance regression(s)!", regression_count);
            std::process::exit(1);
        } else {
            println!("\n✅ No performance regressions detected!");
        }
    }

    Ok(())
}
