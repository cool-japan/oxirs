use std::fs;

use anyhow::Result;

use crate::cli_commands::{AnalysisResult, BenchmarkResults, ValidationResult};

pub fn print_validation_result(result: &ValidationResult) {
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
            println!("  ⚠ {warning}");
        }
    }

    if !result.errors.is_empty() {
        println!("\nErrors:");
        for error in &result.errors {
            println!("  ✗ {error}");
        }
    }
}

pub fn write_validation_report(result: &ValidationResult, path: &str) -> Result<()> {
    let report = serde_json::to_string_pretty(result)?;
    fs::write(path, report)?;
    println!("Validation report written to: {path}");
    Ok(())
}

pub fn print_analysis_result(analysis: &AnalysisResult) {
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
            println!("  {ns}");
        }
    }
}

pub fn format_analysis_report(analysis: &AnalysisResult) -> String {
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

pub fn print_benchmark_results(results: &BenchmarkResults) {
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

pub fn display_profiling_summary(report: &crate::profiling::ProfilingReport) {
    println!("\n=== Profiling Summary ===");
    println!("Total Duration: {:?}", report.total_duration);
    println!("Total Samples: {}", report.total_samples);

    println!("\n=== Operation Statistics ===");
    for (operation, stats) in &report.operation_stats {
        println!("{operation}:");
        println!("  Count: {}", stats.count);
        println!("  Average: {:?}", stats.average_duration);
        println!("  Min: {:?}", stats.min_duration);
        println!("  Max: {:?}", stats.max_duration);
        println!("  Ops/sec: {:.2}", stats.ops_per_second);
        if let Some(bytes_per_sec) = stats.bytes_per_second {
            println!("  MB/sec: {:.2}", bytes_per_sec / 1_000_000.0);
        }
    }

    if !report.bottlenecks.is_empty() {
        println!("\n=== Performance Bottlenecks ===");
        for bottleneck in &report.bottlenecks {
            println!(
                "{}: {:.1}% - {}",
                bottleneck.operation, bottleneck.time_percentage, bottleneck.description
            );
            for suggestion in &bottleneck.suggestions {
                println!("  → {suggestion}");
            }
        }
    }

    if let Some(memory) = &report.memory_patterns {
        println!("\n=== Memory Analysis ===");
        println!(
            "Peak Memory: {:.2} MB",
            memory.peak_memory as f64 / 1_000_000.0
        );
        println!(
            "Average Memory: {:.2} MB",
            memory.average_memory as f64 / 1_000_000.0
        );
        println!("Efficiency Ratio: {:.2}", memory.efficiency_ratio);
        if !memory.potential_leaks.is_empty() {
            println!("Potential Issues:");
            for leak in &memory.potential_leaks {
                println!("  ⚠ {leak}");
            }
        }
    }
}

pub fn generate_html_report(report: &crate::profiling::ProfilingReport) -> Result<String> {
    let mut html = String::new();
    html.push_str(
        "<!DOCTYPE html><html><head><title>RDF-star Profiling Report</title></head><body>",
    );
    html.push_str("<h1>RDF-star Profiling Report</h1>");
    html.push_str(&format!(
        "<p>Generated: {}</p>",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));
    html.push_str(&format!(
        "<p>Total Duration: {:?}</p>",
        report.total_duration
    ));
    html.push_str(&format!("<p>Total Samples: {}</p>", report.total_samples));

    html.push_str("<h2>Operation Statistics</h2><table border='1'>");
    html.push_str("<tr><th>Operation</th><th>Count</th><th>Average</th><th>Min</th><th>Max</th><th>Ops/sec</th></tr>");
    for (operation, stats) in &report.operation_stats {
        html.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{:?}</td><td>{:?}</td><td>{:?}</td><td>{:.2}</td></tr>",
            operation,
            stats.count,
            stats.average_duration,
            stats.min_duration,
            stats.max_duration,
            stats.ops_per_second
        ));
    }
    html.push_str("</table>");
    html.push_str("</body></html>");
    Ok(html)
}

pub fn generate_text_report(report: &crate::profiling::ProfilingReport) -> Result<String> {
    let mut text = String::new();
    text.push_str("RDF-star Profiling Report\n");
    text.push_str("========================\n\n");
    text.push_str(&format!(
        "Generated: {}\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));
    text.push_str(&format!("Total Duration: {:?}\n", report.total_duration));
    text.push_str(&format!("Total Samples: {}\n\n", report.total_samples));

    text.push_str("Operation Statistics\n");
    text.push_str("-------------------\n");
    for (operation, stats) in &report.operation_stats {
        text.push_str(&format!("{operation}:\n"));
        text.push_str(&format!("  Count: {}\n", stats.count));
        text.push_str(&format!("  Average: {:?}\n", stats.average_duration));
        text.push_str(&format!("  Min: {:?}\n", stats.min_duration));
        text.push_str(&format!("  Max: {:?}\n", stats.max_duration));
        text.push_str(&format!("  Ops/sec: {:.2}\n\n", stats.ops_per_second));
    }

    if !report.bottlenecks.is_empty() {
        text.push_str("Performance Bottlenecks\n");
        text.push_str("----------------------\n");
        for bottleneck in &report.bottlenecks {
            text.push_str(&format!(
                "{}: {:.1}% - {}\n",
                bottleneck.operation, bottleneck.time_percentage, bottleneck.description
            ));
            for suggestion in &bottleneck.suggestions {
                text.push_str(&format!("  → {suggestion}\n"));
            }
            text.push('\n');
        }
    }

    Ok(text)
}
