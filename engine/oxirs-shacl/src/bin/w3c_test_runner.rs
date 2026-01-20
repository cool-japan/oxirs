//! CLI tool for running W3C SHACL test suite compliance tests
//!
//! This binary provides a command-line interface for executing the W3C SHACL test suite
//! and generating comprehensive compliance reports.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Result};

use oxirs_shacl::w3c_test_suite::{
    ComplianceReport, ComplianceStats, TestCategory, TestFilter, W3cTestConfig, W3cTestSuiteRunner,
};

/// Command-line arguments for the W3C test runner
#[derive(Debug)]
struct Args {
    /// Test suite location (URL or directory path)
    test_suite_location: Option<String>,

    /// Output directory for reports
    output_dir: Option<PathBuf>,

    /// Test categories to run
    categories: Vec<TestCategory>,

    /// Test patterns to include
    include_patterns: Vec<String>,

    /// Test patterns to exclude
    exclude_patterns: Vec<String>,

    /// Maximum parallel tests
    max_parallel: Option<usize>,

    /// Test timeout in seconds
    timeout: Option<u64>,

    /// Enable verbose logging
    verbose: bool,

    /// Output format (json, text, html)
    output_format: OutputFormat,

    /// Generate detailed report
    detailed_report: bool,
}

#[derive(Debug, Clone)]
enum OutputFormat {
    Json,
    Text,
    Html,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            test_suite_location: None,
            output_dir: None,
            categories: vec![
                TestCategory::Core,
                TestCategory::PropertyPaths,
                TestCategory::NodeShapes,
                TestCategory::PropertyShapes,
            ],
            include_patterns: Vec::new(),
            exclude_patterns: Vec::new(),
            max_parallel: None,
            timeout: None,
            verbose: false,
            output_format: OutputFormat::Text,
            detailed_report: false,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Parse command line arguments
    let args = parse_args()?;

    println!("🧪 OxiRS SHACL - W3C Test Suite Runner");
    println!("=====================================");

    if args.verbose {
        println!("Configuration:");
        println!("  Test suite location: {:?}", args.test_suite_location);
        println!("  Categories: {:?}", args.categories);
        println!("  Max parallel: {:?}", args.max_parallel);
        println!("  Timeout: {:?}s", args.timeout);
        println!("  Output format: {:?}", args.output_format);
        println!();
    }

    // Create test configuration
    let config = create_test_config(&args)?;

    // Create and configure test runner
    let mut runner = W3cTestSuiteRunner::new(config)?;

    println!("📥 Loading W3C SHACL test manifests...");
    runner.load_manifests().await?;

    println!("✅ Loaded {} test manifests", runner.manifests.len());

    // Count total tests
    let total_tests: usize = runner.manifests.iter().map(|m| m.entries.len()).sum();

    println!("🎯 Found {total_tests} total tests");
    println!();

    // Execute tests
    println!("🚀 Executing W3C SHACL compliance tests...");
    let start_time = std::time::Instant::now();

    let stats = runner.execute_all_tests().await?;

    let execution_time = start_time.elapsed();
    println!(
        "⏱️  Execution completed in {:.2}s",
        execution_time.as_secs_f64()
    );
    println!();

    // Display results summary
    display_results_summary(&stats);

    // Generate detailed compliance report
    let report = runner.generate_compliance_report();

    // Save reports if output directory specified
    if let Some(output_dir) = &args.output_dir {
        save_reports(&report, &stats, output_dir, &args).await?;
    }

    // Display final compliance assessment
    display_compliance_assessment(&stats);

    // Exit with appropriate code
    let exit_code = if stats.compliance_percentage >= 95.0 {
        0 // Excellent compliance
    } else if stats.compliance_percentage >= 80.0 {
        1 // Good compliance with issues
    } else {
        2 // Poor compliance
    };

    std::process::exit(exit_code);
}

/// Parse command line arguments
fn parse_args() -> Result<Args> {
    let mut args = Args::default();

    let env_args: Vec<String> = std::env::args().collect();

    let mut i = 1;
    while i < env_args.len() {
        match env_args[i].as_str() {
            "--test-suite" | "-t" => {
                i += 1;
                if i >= env_args.len() {
                    return Err(anyhow!("Missing value for --test-suite"));
                }
                args.test_suite_location = Some(env_args[i].clone());
            }
            "--output" | "-o" => {
                i += 1;
                if i >= env_args.len() {
                    return Err(anyhow!("Missing value for --output"));
                }
                args.output_dir = Some(PathBuf::from(&env_args[i]));
            }
            "--categories" | "-c" => {
                i += 1;
                if i >= env_args.len() {
                    return Err(anyhow!("Missing value for --categories"));
                }
                args.categories = parse_categories(&env_args[i])?;
            }
            "--include" => {
                i += 1;
                if i >= env_args.len() {
                    return Err(anyhow!("Missing value for --include"));
                }
                args.include_patterns.push(env_args[i].clone());
            }
            "--exclude" => {
                i += 1;
                if i >= env_args.len() {
                    return Err(anyhow!("Missing value for --exclude"));
                }
                args.exclude_patterns.push(env_args[i].clone());
            }
            "--parallel" | "-p" => {
                i += 1;
                if i >= env_args.len() {
                    return Err(anyhow!("Missing value for --parallel"));
                }
                args.max_parallel = Some(env_args[i].parse()?);
            }
            "--timeout" => {
                i += 1;
                if i >= env_args.len() {
                    return Err(anyhow!("Missing value for --timeout"));
                }
                args.timeout = Some(env_args[i].parse()?);
            }
            "--verbose" | "-v" => {
                args.verbose = true;
            }
            "--format" | "-f" => {
                i += 1;
                if i >= env_args.len() {
                    return Err(anyhow!("Missing value for --format"));
                }
                args.output_format = parse_output_format(&env_args[i])?;
            }
            "--detailed" | "-d" => {
                args.detailed_report = true;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            arg => {
                return Err(anyhow!("Unknown argument: {}", arg));
            }
        }
        i += 1;
    }

    Ok(args)
}

/// Parse test categories from comma-separated string
fn parse_categories(categories_str: &str) -> Result<Vec<TestCategory>> {
    let mut categories = Vec::new();

    for category in categories_str.split(',') {
        let category = category.trim().to_lowercase();
        match category.as_str() {
            "core" => categories.push(TestCategory::Core),
            "property-paths" | "paths" => categories.push(TestCategory::PropertyPaths),
            "node-shapes" | "nodes" => categories.push(TestCategory::NodeShapes),
            "property-shapes" | "properties" => categories.push(TestCategory::PropertyShapes),
            "logical" => categories.push(TestCategory::LogicalConstraints),
            "sparql" => categories.push(TestCategory::SparqlConstraints),
            "advanced" => categories.push(TestCategory::Advanced),
            "performance" => categories.push(TestCategory::Performance),
            "all" => {
                categories = vec![
                    TestCategory::Core,
                    TestCategory::PropertyPaths,
                    TestCategory::NodeShapes,
                    TestCategory::PropertyShapes,
                    TestCategory::LogicalConstraints,
                    TestCategory::SparqlConstraints,
                    TestCategory::Advanced,
                    TestCategory::Performance,
                ];
                break;
            }
            _ => return Err(anyhow!("Unknown test category: {}", category)),
        }
    }

    Ok(categories)
}

/// Parse output format from string
fn parse_output_format(format_str: &str) -> Result<OutputFormat> {
    match format_str.to_lowercase().as_str() {
        "json" => Ok(OutputFormat::Json),
        "text" | "txt" => Ok(OutputFormat::Text),
        "html" => Ok(OutputFormat::Html),
        _ => Err(anyhow!("Unknown output format: {}", format_str)),
    }
}

/// Create test configuration from command line arguments
fn create_test_config(args: &Args) -> Result<W3cTestConfig> {
    let mut enabled_categories = HashSet::new();
    for category in &args.categories {
        enabled_categories.insert(category.clone());
    }

    let mut test_filters = Vec::new();

    // Add include filters
    for pattern in &args.include_patterns {
        test_filters.push(TestFilter {
            name: format!("include-{pattern}"),
            test_pattern: pattern.clone(),
            include: true,
        });
    }

    // Add exclude filters
    for pattern in &args.exclude_patterns {
        test_filters.push(TestFilter {
            name: format!("exclude-{pattern}"),
            test_pattern: pattern.clone(),
            include: false,
        });
    }

    Ok(W3cTestConfig {
        test_suite_location: args.test_suite_location.clone().unwrap_or_else(|| {
            "https://w3c.github.io/data-shapes/data-shapes-test-suite/".to_string()
        }),
        enabled_categories,
        test_timeout_seconds: args.timeout.unwrap_or(30),
        max_parallel_tests: args.max_parallel.unwrap_or(4),
        verbose_logging: args.verbose,
        output_directory: args.output_dir.clone(),
        test_filters,
    })
}

/// Display results summary
fn display_results_summary(stats: &ComplianceStats) {
    println!("📊 Test Results Summary");
    println!("======================");
    println!("Total tests:     {}", stats.total_tests);
    println!(
        "✅ Passed:       {} ({:.1}%)",
        stats.tests_passed,
        if stats.total_tests > 0 {
            (stats.tests_passed as f64 / stats.total_tests as f64) * 100.0
        } else {
            0.0
        }
    );
    println!(
        "❌ Failed:       {} ({:.1}%)",
        stats.tests_failed,
        if stats.total_tests > 0 {
            (stats.tests_failed as f64 / stats.total_tests as f64) * 100.0
        } else {
            0.0
        }
    );
    println!(
        "⏭️  Skipped:      {} ({:.1}%)",
        stats.tests_skipped,
        if stats.total_tests > 0 {
            (stats.tests_skipped as f64 / stats.total_tests as f64) * 100.0
        } else {
            0.0
        }
    );
    println!(
        "💥 Errors:       {} ({:.1}%)",
        stats.tests_error,
        if stats.total_tests > 0 {
            (stats.tests_error as f64 / stats.total_tests as f64) * 100.0
        } else {
            0.0
        }
    );
    println!("⏱️  Execution time: {}ms", stats.total_execution_time_ms);
    println!();

    // Display common issues if any
    if !stats.common_issues.is_empty() {
        println!("🚨 Common Issues:");
        for (issue_type, count) in &stats.common_issues[..3.min(stats.common_issues.len())] {
            println!("  {issue_type:?}: {count} occurrences");
        }
        println!();
    }
}

/// Display compliance assessment
fn display_compliance_assessment(stats: &ComplianceStats) {
    println!("🏆 Compliance Assessment");
    println!("========================");

    let compliance = stats.compliance_percentage;
    let status = if compliance >= 95.0 {
        ("🌟 EXCELLENT", "Full W3C SHACL compliance")
    } else if compliance >= 80.0 {
        ("✅ GOOD", "Good compliance with minor issues")
    } else if compliance >= 60.0 {
        ("⚠️  FAIR", "Partial compliance - improvements needed")
    } else {
        ("❌ POOR", "Significant compliance issues")
    };

    println!(
        "Overall compliance: {:.1}% - {} ({})",
        compliance, status.0, status.1
    );
    println!();

    if compliance < 100.0 {
        println!("💡 Recommendations:");
        if stats.tests_failed > 0 {
            println!(
                "  • Review and fix {} failing test case(s)",
                stats.tests_failed
            );
        }
        if stats.tests_error > 0 {
            println!(
                "  • Investigate {} test execution error(s)",
                stats.tests_error
            );
        }
        if !stats.common_issues.is_empty() {
            println!("  • Address common compliance issues listed above");
        }
        println!("  • Run tests with --verbose for detailed analysis");
        println!("  • Generate detailed report with --detailed flag");
    }
}

/// Save reports to output directory
async fn save_reports(
    report: &ComplianceReport,
    stats: &ComplianceStats,
    output_dir: &PathBuf,
    args: &Args,
) -> Result<()> {
    // Create output directory if it doesn't exist
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    println!("💾 Saving reports to: {}", output_dir.display());

    // Save summary statistics
    let stats_file = output_dir.join("compliance_stats.json");
    let stats_json = serde_json::to_string_pretty(stats)?;
    fs::write(&stats_file, stats_json)?;
    println!("  📄 Compliance statistics: {}", stats_file.display());

    // Save detailed report based on format
    match args.output_format {
        OutputFormat::Json => {
            let report_file = output_dir.join("compliance_report.json");
            let report_json = serde_json::to_string_pretty(report)?;
            fs::write(&report_file, report_json)?;
            println!("  📋 Detailed report: {}", report_file.display());
        }
        OutputFormat::Text => {
            let report_file = output_dir.join("compliance_report.txt");
            let report_text = generate_text_report(report, stats);
            fs::write(&report_file, report_text)?;
            println!("  📋 Detailed report: {}", report_file.display());
        }
        OutputFormat::Html => {
            let report_file = output_dir.join("compliance_report.html");
            let report_html = generate_html_report(report, stats);
            fs::write(&report_file, report_html)?;
            println!("  📋 Detailed report: {}", report_file.display());
        }
    }

    Ok(())
}

/// Generate text format report
fn generate_text_report(report: &ComplianceReport, stats: &ComplianceStats) -> String {
    let mut text = String::new();

    text.push_str("W3C SHACL Compliance Report\n");
    text.push_str("===========================\n\n");

    text.push_str(&format!(
        "Generated: {}\n",
        report.generated_at.format("%Y-%m-%d %H:%M:%S UTC")
    ));
    text.push_str(&format!(
        "Implementation: OxiRS SHACL v{}\n",
        report.implementation_details.version
    ));
    text.push_str(&format!("Test Suite: {}\n\n", report.test_suite_version));

    text.push_str("Summary Statistics:\n");
    text.push_str(&format!("  Total tests: {}\n", stats.total_tests));
    text.push_str(&format!("  Passed: {}\n", stats.tests_passed));
    text.push_str(&format!("  Failed: {}\n", stats.tests_failed));
    text.push_str(&format!("  Skipped: {}\n", stats.tests_skipped));
    text.push_str(&format!("  Errors: {}\n", stats.tests_error));
    text.push_str(&format!(
        "  Compliance: {:.1}%\n\n",
        stats.compliance_percentage
    ));

    if !report.implementation_details.limitations.is_empty() {
        text.push_str("Known Limitations:\n");
        for limitation in &report.implementation_details.limitations {
            text.push_str(&format!("  • {limitation}\n"));
        }
        text.push('\n');
    }

    text
}

/// Generate HTML format report
fn generate_html_report(report: &ComplianceReport, stats: &ComplianceStats) -> String {
    format!(
        r#"
<!DOCTYPE html>
<html>
<head>
    <title>W3C SHACL Compliance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
        .stat-card {{ background: #f5f5f5; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #2c5aa0; }}
        .compliance {{ font-size: 1.5em; margin: 20px 0; }}
        .excellent {{ color: #28a745; }}
        .good {{ color: #17a2b8; }}
        .fair {{ color: #ffc107; }}
        .poor {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 W3C SHACL Compliance Report</h1>
        <p>Generated: {}</p>
        <p>Implementation: OxiRS SHACL v{}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <h3>Total Tests</h3>
            <div class="stat-number">{}</div>
        </div>
        <div class="stat-card">
            <h3>Passed</h3>
            <div class="stat-number">{}</div>
        </div>
        <div class="stat-card">
            <h3>Failed</h3>
            <div class="stat-number">{}</div>
        </div>
        <div class="stat-card">
            <h3>Compliance</h3>
            <div class="stat-number">{:.1}%</div>
        </div>
    </div>

    <div class="compliance {}">
        Overall Compliance: {:.1}%
    </div>

    <h2>Implementation Details</h2>
    <ul>
        {}
    </ul>
</body>
</html>
"#,
        report.generated_at.format("%Y-%m-%d %H:%M:%S UTC"),
        report.implementation_details.version,
        stats.total_tests,
        stats.tests_passed,
        stats.tests_failed,
        stats.compliance_percentage,
        if stats.compliance_percentage >= 95.0 {
            "excellent"
        } else if stats.compliance_percentage >= 80.0 {
            "good"
        } else if stats.compliance_percentage >= 60.0 {
            "fair"
        } else {
            "poor"
        },
        stats.compliance_percentage,
        report
            .implementation_details
            .features
            .iter()
            .map(|f| format!("<li>{f}</li>"))
            .collect::<Vec<_>>()
            .join("\n        ")
    )
}

/// Print help message
fn print_help() {
    println!("OxiRS SHACL - W3C Test Suite Runner");
    println!();
    println!("USAGE:");
    println!("    w3c_test_runner [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    -t, --test-suite <PATH>    Test suite location (URL or directory)");
    println!("    -o, --output <DIR>         Output directory for reports");
    println!("    -c, --categories <LIST>    Test categories (comma-separated)");
    println!("                               Options: core,paths,nodes,properties,logical,sparql,advanced,performance,all");
    println!("    --include <PATTERN>        Include tests matching pattern");
    println!("    --exclude <PATTERN>        Exclude tests matching pattern");
    println!("    -p, --parallel <N>         Maximum parallel tests");
    println!("    --timeout <SECONDS>        Test timeout in seconds");
    println!("    -f, --format <FORMAT>      Output format (json,text,html)");
    println!("    -v, --verbose              Enable verbose logging");
    println!("    -d, --detailed             Generate detailed report");
    println!("    -h, --help                 Print this help message");
    println!();
    println!("EXAMPLES:");
    println!("    w3c_test_runner --categories core,paths --output ./reports");
    println!("    w3c_test_runner --verbose --detailed --format html");
    println!("    w3c_test_runner --include core- --exclude sparql-");
}
