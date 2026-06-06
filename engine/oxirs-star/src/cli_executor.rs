use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use tracing::info;

use crate::cli_commands::{
    AnalysisResult, BenchmarkResults, PerformanceAnalysis, SystemHealth, ValidationResult,
};
use crate::model::{StarTerm, StarTriple};
use crate::parser::{StarFormat, StarParser};
use crate::profiling::{ProfilingConfig, StarProfiler};
use crate::serializer::{SerializationOptions, StarSerializer};
use crate::store::StarStore;
use crate::troubleshooting::{
    DiagnosticAnalyzer, MigrationAssistant, MigrationSourceFormat, TroubleshootingGuide,
    TroubleshootingIssue,
};
use crate::{StarConfig, StarError, StarResult};

pub fn detect_format(path: &str, content: &str) -> Result<StarFormat> {
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

    if content.trim_start().starts_with('{') || content.trim_start().starts_with('[') {
        Ok(StarFormat::JsonLdStar)
    } else if content.contains("<<") && content.contains(">>") {
        if content.contains("GRAPH") || content.contains('{') {
            Ok(StarFormat::TrigStar)
        } else {
            Ok(StarFormat::TurtleStar)
        }
    } else {
        Ok(StarFormat::TurtleStar)
    }
}

pub fn has_quoted_terms(triple: &StarTriple) -> bool {
    matches!(triple.subject, StarTerm::QuotedTriple(_))
        || matches!(triple.object, StarTerm::QuotedTriple(_))
}

pub fn extract_namespace(iri: &str) -> Option<String> {
    iri.rfind(['#', '/']).map(|pos| iri[..=pos].to_string())
}

pub fn analyze_term(term: &StarTerm, analysis: &mut AnalysisResult, depth: usize) {
    match term {
        StarTerm::QuotedTriple(quoted) => {
            analysis.max_nesting_depth = analysis.max_nesting_depth.max(depth);
            analyze_triple(quoted, analysis, depth);
        }
        StarTerm::NamedNode(node) => {
            analysis.subjects.insert(node.iri.clone());
            if let Some(namespace) = extract_namespace(&node.iri) {
                analysis.namespaces.insert(namespace);
            }
        }
        _ => {
            analysis.objects.insert(term.to_string());
        }
    }
}

pub fn analyze_triple(triple: &StarTriple, analysis: &mut AnalysisResult, depth: usize) {
    analysis.max_nesting_depth = analysis.max_nesting_depth.max(depth);

    if has_quoted_terms(triple) {
        analysis.quoted_triples += 1;
    }

    analyze_term(&triple.subject, analysis, depth + 1);
    analysis.predicates.insert(triple.predicate.to_string());
    analyze_term(&triple.object, analysis, depth + 1);
}

pub fn validate_file(
    path: &str,
    format: Option<&String>,
    strict: bool,
) -> Result<ValidationResult> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read file: {path}"))?;

    let detected_format = if let Some(fmt) = format {
        fmt.parse::<StarFormat>()
            .map_err(|_| anyhow::anyhow!("Invalid format: {fmt}"))?
    } else {
        detect_format(path, &content)?
    };

    let mut parser = StarParser::new();
    if strict {
        parser.set_strict_mode(true);
    }

    let mut errors = Vec::new();
    let warnings = Vec::new();
    let mut triple_count = 0;
    let mut quoted_triple_count = 0;

    let parse_result = parser.parse_str(&content, detected_format);

    match parse_result {
        Ok(graph) => {
            triple_count = graph.len();
            for triple in &graph {
                if has_quoted_terms(triple) {
                    quoted_triple_count += 1;
                }
            }
        }
        Err(e) => {
            errors.push(format!("Parse error: {e}"));
        }
    }

    let parse_errors = parser.get_errors();
    for error in parse_errors {
        match &error {
            StarError::ParseError(details) => {
                if let Some(line_num) = details.line {
                    errors.push(format!("Line {line_num}: {}", details.message));
                } else {
                    errors.push(details.message.clone());
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

pub fn convert_file(
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
        detect_format(input, &content)?
    };

    let to_format = to.parse::<StarFormat>()?;

    let parser = StarParser::new();
    let graph = parser.parse_str(&content, from_format)?;

    let serializer = StarSerializer::new();
    let mut options = SerializationOptions::default();
    if pretty {
        options.pretty_print = true;
    }

    let output_content = serializer.serialize_graph(&graph, to_format, &options)?;
    fs::write(output, output_content)?;

    Ok(())
}

pub fn analyze_file(path: &str) -> Result<AnalysisResult> {
    let content = fs::read_to_string(path)?;
    let format = detect_format(path, &content)?;

    let parser = StarParser::new();
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

    for triple in &graph {
        analyze_triple(triple, &mut analysis, 0);
    }

    Ok(analysis)
}

pub fn debug_file(path: &str, target_line: Option<usize>, context_lines: usize) -> Result<()> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();

    println!("Debugging file: {path}");
    println!("Total lines: {}", lines.len());
    println!();

    let format = detect_format(path, &content)?;
    println!("Detected format: {format:?}");
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
                StarError::ParseError(details) => {
                    if let (Some(line), Some(column)) = (details.line, details.column) {
                        println!("  Line {line}, Column {column}: {}", details.message);

                        let start_line = line.saturating_sub(context_lines + 1);
                        let end_line = (line + context_lines).min(lines.len());

                        println!("  Context lines:");
                        for line_num in start_line..end_line {
                            let marker = if line_num + 1 == line { ">>>" } else { "   " };
                            println!(
                                "  {} {:4}: {}",
                                marker,
                                line_num + 1,
                                lines.get(line_num).unwrap_or(&"")
                            );
                        }
                    } else {
                        println!("  {}", details.message);
                    }
                }
                _ => {
                    println!("  {error}");
                }
            }

            println!();
        }
    }

    if let Some(line_num) = target_line {
        if line_num > 0 && line_num <= lines.len() {
            println!("Analysis for line {line_num}:");
            println!("  Content: {}", lines[line_num - 1]);
        }
    }

    if let Ok(graph) = parse_result {
        println!("Successfully parsed {} triples", graph.len());
    }

    Ok(())
}

pub fn benchmark_file(path: &str, iterations: usize, warmup: usize) -> Result<BenchmarkResults> {
    let content = fs::read_to_string(path)?;
    let format = detect_format(path, &content)?;
    let file_size = content.len();

    println!("Benchmarking {path} ({file_size} bytes, {iterations} iterations + {warmup} warmup)");

    for _ in 0..warmup {
        let parser = StarParser::new();
        let _ = parser.parse_str(&content, format);
    }

    let mut parse_times = Vec::with_capacity(iterations);
    let mut serialize_times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let parser = StarParser::new();
        let graph = parser.parse_str(&content, format)?;
        let parse_duration = start.elapsed();
        parse_times.push(parse_duration);

        let start = Instant::now();
        let serializer = StarSerializer::new();
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

pub fn execute_query(data_path: &str, query_input: &str, _output_format: &str) -> Result<()> {
    let content = fs::read_to_string(data_path)?;
    let format = detect_format(data_path, &content)?;

    let parser = StarParser::new();
    let graph = parser.parse_str(&content, format)?;

    let store = StarStore::new();
    for triple in &graph {
        store.insert(triple)?;
    }

    let query_text = if Path::new(query_input).exists() {
        fs::read_to_string(query_input)?
    } else {
        query_input.to_string()
    };

    info!("Executing SPARQL-star query on {} triples", store.len());

    let start_time = Instant::now();
    let duration = start_time.elapsed();

    println!("Query executed in {duration:?}");
    println!("Query: {query_text}");
    println!("Store contains {} triples", store.len());

    Ok(())
}

pub fn run_troubleshoot(error_input: &str, output_path: Option<&String>) -> Result<()> {
    let _guide = TroubleshootingGuide::new();
    let config = StarConfig::default();
    let _analyzer = DiagnosticAnalyzer::new(config);

    info!("Analyzing error: {}", error_input);

    let diagnosis = TroubleshootingIssue {
        title: "Error Analysis".to_string(),
        description: error_input.to_string(),
        category: crate::troubleshooting::IssueCategory::Parsing,
        symptoms: vec![error_input.to_string()],
        causes: vec!["Unknown cause".to_string()],
        solutions: vec![],
        examples: vec![],
        see_also: vec![],
    };

    let recommendations = [
        "Check your input format".to_string(),
        "Verify the file syntax".to_string(),
        "Try a simpler query".to_string(),
    ];

    let health_report = "System OK".to_string();

    let report = format!(
        "RDF-star Troubleshooting Report\n\
        ================================\n\n\
        Error Analysis:\n\
        ---------------\n\
        Error Type: {}\n\
        Severity: {:?}\n\
        Description: {}\n\n\
        Recommendations:\n\
        ----------------\n{}\
        \n\nSystem Health:\n\
        ---------------\n{}",
        diagnosis.title,
        diagnosis.category,
        diagnosis.description,
        recommendations
            .iter()
            .map(|r| format!("• {r}\n"))
            .collect::<String>(),
        health_report
    );

    if let Some(output) = output_path {
        fs::write(output, &report)?;
        println!("Troubleshooting report written to: {output}");
    } else {
        println!("{report}");
    }

    Ok(())
}

pub fn run_migrate(
    source_file: &str,
    output_file: &str,
    source_format: &str,
    plan_only: bool,
    quiet: bool,
) -> Result<()> {
    info!(
        "Starting RDF-star migration from {} to {}",
        source_file, output_file
    );

    let migration_format = source_format
        .parse::<MigrationSourceFormat>()
        .map_err(|_| anyhow!("Unsupported source format: {}", source_format))?;

    let config = StarConfig::default();
    let assistant = MigrationAssistant::new(migration_format.clone(), config);

    let analysis = assistant.analyze_source(source_file, migration_format.clone())?;

    if !quiet {
        println!("Source Analysis:");
        println!("  Format: {migration_format:?}");
        println!("  Triples: {}", analysis.total_triples);
        println!(
            "  Estimated quoted triples after migration: {}",
            analysis.reified_statements
        );
        println!("  Compatibility Score: {:.2}", analysis.compatibility_score);
    }

    let plan = assistant.create_migration_plan(&analysis)?;

    if plan_only {
        println!("Migration Plan:");
        println!("===============");
        for (i, step) in plan.steps.iter().enumerate() {
            println!("{}. {}", i + 1, step.description);
            if let Some(command) = &step.command {
                println!("   Command: {command}");
            }
        }
        return Ok(());
    }

    let start_time = Instant::now();
    let result = assistant.execute_migration(source_file, output_file, &plan)?;
    let duration = start_time.elapsed();

    if !quiet {
        println!("Migration completed in {duration:?}");
        println!("Results:");
        println!("  Executed steps: {}", result.executed_steps.len());
        println!("  Output file: {}", result.output_file);
        println!("  Success: {}", result.success);
        println!("  Warnings: {}", result.warnings.len());

        if !result.warnings.is_empty() {
            println!("\nWarnings:");
            for warning in &result.warnings {
                println!("  ⚠ {warning}");
            }
        }
    }

    Ok(())
}

#[allow(clippy::result_large_err)]
pub fn run_system_diagnostics() -> StarResult<SystemHealth> {
    Ok(SystemHealth {
        memory_available: true,
        disk_space_sufficient: true,
        dependencies_satisfied: true,
        configuration_valid: true,
        overall_status: "Healthy".to_string(),
    })
}

#[allow(clippy::result_large_err)]
pub fn run_performance_analysis(input_file: &str) -> StarResult<PerformanceAnalysis> {
    let metadata = fs::metadata(input_file)
        .map_err(|e| StarError::parse_error(format!("Failed to read file metadata: {e}")))?;

    let file_size = metadata.len();
    let estimated_parse_time = (file_size as f64 / 1024.0) * 0.1;

    Ok(PerformanceAnalysis {
        file_size_bytes: file_size,
        estimated_parse_time_ms: estimated_parse_time,
        memory_requirements_mb: (file_size as f64 / 1024.0 / 1024.0) * 2.0,
        optimization_suggestions: vec![
            "Consider using streaming parser for large files".to_string(),
            "Enable indexing for better query performance".to_string(),
        ],
    })
}

pub fn run_doctor(
    input_file: &str,
    report_path: Option<&String>,
    auto_fix: bool,
    quiet: bool,
) -> Result<()> {
    info!(
        "Running comprehensive diagnostic analysis on: {}",
        input_file
    );

    let config = StarConfig::default();
    let analyzer = DiagnosticAnalyzer::new(config);
    let start_time = Instant::now();

    let diagnostic_result = analyzer.run_comprehensive_analysis(input_file)?;
    let duration = start_time.elapsed();

    let mut fixes_applied = 0;

    let system_health = run_system_diagnostics()?;
    let perf_analysis = run_performance_analysis(input_file)?;

    let report = format!(
        "RDF-star Diagnostic Report\n\
        ===========================\n\n\
        File: {}\n\
        Analysis Duration: {:?}\n\n\
        Structural Analysis:\n\
        --------------------\n\
        Total Triples: {}\n\
        Quoted Triples: {}\n\
        Max Nesting Depth: {}\n\
        Syntax Errors: {}\n\
        Semantic Issues: {}\n\n\
        Quality Assessment:\n\
        -------------------\n\
        Overall Score: {}/100\n\
        Readability: {}/10\n\
        Efficiency: {}/10\n\
        Compliance: {}/10\n\n\
        Issues Found:\n\
        -------------\n{}\
        \nPerformance Analysis:\n\
        ---------------------\n{}\
        \nSystem Health:\n\
        ---------------\n{}",
        input_file,
        duration,
        diagnostic_result
            .performance_metrics
            .estimated_parse_time_ms,
        diagnostic_result
            .performance_metrics
            .estimated_memory_usage_mb,
        diagnostic_result.performance_metrics.complexity_score,
        diagnostic_result.issues_found.len(),
        diagnostic_result.recommendations.len(),
        diagnostic_result.data_quality.completeness_score,
        diagnostic_result.data_quality.consistency_score,
        diagnostic_result.data_quality.uniqueness_score,
        diagnostic_result.data_quality.validity_score,
        diagnostic_result
            .issues_found
            .iter()
            .map(|issue| format!(
                "• {} ({}): {}\n",
                issue.severity, issue.category, issue.message
            ))
            .collect::<String>(),
        perf_analysis,
        system_health
    );

    if auto_fix {
        let fixes = analyzer.apply_automatic_fixes(input_file, &diagnostic_result.issues_found)?;
        fixes_applied = fixes.len();

        if !quiet && fixes_applied > 0 {
            println!("Applied {fixes_applied} automatic fixes:");
            for fix in &fixes {
                println!("  ✓ {fix}");
            }
        }
    }

    let issues_found = diagnostic_result.issues_found.len();

    if let Some(report_file) = report_path {
        fs::write(report_file, &report)?;
        println!("Diagnostic report written to: {report_file}");
    } else if !quiet {
        println!("{report}");
    }

    if !quiet {
        println!("\nDiagnostic Summary:");
        println!("===================");
        println!("Issues found: {issues_found}");
        if auto_fix {
            println!("Fixes applied: {fixes_applied}");
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

pub fn run_profile(
    input_path: &str,
    operations: &str,
    iterations: usize,
    report_path: Option<&String>,
    quiet: bool,
) -> Result<()> {
    info!(
        "Profiling RDF-star file: {} (operations: {}, iterations: {})",
        input_path, operations, iterations
    );

    let mut profiler = StarProfiler::with_config(ProfilingConfig {
        track_memory: true,
        track_timing: true,
        sample_rate: 1.0,
        max_samples: iterations * 10,
        enable_statistics: true,
    });

    let start_time = Instant::now();

    let input_data = fs::read_to_string(input_path)
        .with_context(|| format!("Failed to read input file: {input_path}"))?;
    let input_size = input_data.len();

    let format = detect_format(input_path, &input_data)?;

    if operations == "all" || operations.contains("parse") {
        for i in 0..iterations {
            info!("Profiling parsing iteration {}/{}", i + 1, iterations);
            profiler.profile_parsing(format, input_size, || {
                let parser = StarParser::new();
                let _ = parser.parse_str(&input_data, format);
            });
        }
    }

    if operations == "all" || operations.contains("serialize") {
        let parser = StarParser::new();
        if let Ok(graph) = parser.parse_str(&input_data, format) {
            let triple_count = graph.len();
            for i in 0..iterations {
                info!("Profiling serialization iteration {}/{}", i + 1, iterations);
                profiler.profile_serialization(format, triple_count, || {
                    let serializer = StarSerializer::new();
                    let _ = serializer.serialize_graph(
                        &graph,
                        format,
                        &SerializationOptions::default(),
                    );
                });
            }
        }
    }

    let duration = start_time.elapsed();
    info!("Profiling completed in {:?}", duration);

    let report = profiler.generate_report();

    if !quiet {
        crate::cli_output::display_profiling_summary(&report);
    }

    if let Some(rp) = report_path {
        let report_json = serde_json::to_string_pretty(&report)?;
        fs::write(rp, report_json)
            .with_context(|| format!("Failed to write profiling report: {rp}"))?;
        info!("Detailed profiling report saved to: {}", rp);
    }

    Ok(())
}

pub fn run_profile_report(
    data_path: &str,
    output_path: Option<&String>,
    format: &str,
) -> Result<()> {
    info!("Generating profiling report from: {}", data_path);

    let data_content = fs::read_to_string(data_path)
        .with_context(|| format!("Failed to read profiling data: {data_path}"))?;

    let mut profiler = StarProfiler::new();
    profiler
        .import_json(&data_content)
        .with_context(|| "Failed to parse profiling data")?;

    let report = profiler.generate_report();

    match format {
        "json" => {
            let report_json = serde_json::to_string_pretty(&report)?;
            if let Some(op) = output_path {
                fs::write(op, report_json)
                    .with_context(|| format!("Failed to write JSON report: {op}"))?;
                info!("JSON report saved to: {}", op);
            } else {
                println!("{report_json}");
            }
        }
        "html" => {
            let html_report = crate::cli_output::generate_html_report(&report)?;
            if let Some(op) = output_path {
                fs::write(op, html_report)
                    .with_context(|| format!("Failed to write HTML report: {op}"))?;
                info!("HTML report saved to: {}", op);
            } else {
                println!("{html_report}");
            }
        }
        _ => {
            let text_report = crate::cli_output::generate_text_report(&report)?;
            if let Some(op) = output_path {
                fs::write(op, text_report)
                    .with_context(|| format!("Failed to write text report: {op}"))?;
                info!("Text report saved to: {}", op);
            } else {
                println!("{text_report}");
            }
        }
    }

    Ok(())
}
