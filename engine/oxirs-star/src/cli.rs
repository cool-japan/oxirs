//! Command-line interface tools for RDF-star validation and debugging.

use anyhow::{anyhow, Result};
use clap::ArgMatches;
use serde_json;
use std::fs;
use tracing::info;

use crate::cli_commands::build_cli;
use crate::cli_executor::{
    analyze_file, benchmark_file, convert_file, debug_file, execute_query, run_doctor, run_migrate,
    run_profile, run_profile_report, run_troubleshoot, validate_file,
};
use crate::cli_output::{
    format_analysis_report, print_analysis_result, print_benchmark_results,
    print_validation_result, write_validation_report,
};

pub use crate::cli_commands::{
    AnalysisResult, BenchmarkResults, PerformanceAnalysis, SystemHealth, ValidationResult,
};

/// CLI application for RDF-star tools
pub struct StarCli {
    pub verbose: bool,
    pub quiet: bool,
}

impl StarCli {
    pub fn new() -> Self {
        Self {
            verbose: false,
            quiet: false,
        }
    }

    pub fn run(&mut self, args: Vec<String>) -> Result<()> {
        let app = build_cli();
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
            Some(("profile", sub_matches)) => self.profile_command(sub_matches),
            Some(("profile-report", sub_matches)) => self.profile_report_command(sub_matches),
            _ => {
                eprintln!("No command specified. Use --help for usage information.");
                std::process::exit(1);
            }
        }
    }

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

    fn validate_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches
            .get_one::<String>("input")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let format = matches.get_one::<String>("format");
        let strict = matches.get_flag("strict");
        let report_path = matches.get_one::<String>("report");

        info!("Validating RDF-star file: {}", input_path);

        let start_time = std::time::Instant::now();
        let validation_result = validate_file(input_path, format, strict)?;
        let duration = start_time.elapsed();

        if !self.quiet {
            println!("Validation completed in {duration:?}");
            print_validation_result(&validation_result);
        }

        if let Some(rp) = report_path {
            write_validation_report(&validation_result, rp)?;
        }

        if validation_result.is_valid {
            Ok(())
        } else {
            std::process::exit(1);
        }
    }

    fn convert_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches
            .get_one::<String>("input")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let output_path = matches
            .get_one::<String>("output")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let from_format = matches.get_one::<String>("from");
        let to_format = matches
            .get_one::<String>("to")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let pretty = matches.get_flag("pretty");

        info!("Converting {} to {}", input_path, output_path);

        let start_time = std::time::Instant::now();
        convert_file(input_path, output_path, from_format, to_format, pretty)?;
        let duration = start_time.elapsed();

        if !self.quiet {
            println!("Conversion completed in {duration:?}");
        }

        Ok(())
    }

    fn analyze_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches
            .get_one::<String>("input")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let output_path = matches.get_one::<String>("output");
        let json_output = matches.get_flag("json");

        info!("Analyzing RDF-star file: {}", input_path);

        let start_time = std::time::Instant::now();
        let analysis = analyze_file(input_path)?;
        let duration = start_time.elapsed();

        if !self.quiet {
            println!("Analysis completed in {duration:?}");
        }

        if json_output {
            let json_str = serde_json::to_string_pretty(&analysis)?;
            if let Some(op) = output_path {
                fs::write(op, json_str)?;
            } else {
                println!("{json_str}");
            }
        } else if let Some(op) = output_path {
            let report = format_analysis_report(&analysis);
            fs::write(op, report)?;
        } else {
            print_analysis_result(&analysis);
        }

        Ok(())
    }

    fn debug_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches
            .get_one::<String>("input")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let target_line = matches
            .get_one::<String>("line")
            .map(|s| s.parse::<usize>().unwrap_or(0));
        let context_lines: usize = matches
            .get_one::<String>("context")
            .ok_or_else(|| anyhow!("missing required argument"))?
            .parse()
            .unwrap_or(3);

        info!("Debugging RDF-star file: {}", input_path);
        debug_file(input_path, target_line, context_lines)
    }

    fn benchmark_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches
            .get_one::<String>("input")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let iterations: usize = matches
            .get_one::<String>("iterations")
            .ok_or_else(|| anyhow!("missing required argument"))?
            .parse()
            .unwrap_or(10);
        let warmup: usize = matches
            .get_one::<String>("warmup")
            .ok_or_else(|| anyhow!("missing required argument"))?
            .parse()
            .unwrap_or(3);

        info!("Benchmarking file: {}", input_path);

        let results = benchmark_file(input_path, iterations, warmup)?;
        print_benchmark_results(&results);

        Ok(())
    }

    fn query_command(&self, matches: &ArgMatches) -> Result<()> {
        let data_path = matches
            .get_one::<String>("data")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let query_input = matches
            .get_one::<String>("query")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let output_format = matches
            .get_one::<String>("format")
            .ok_or_else(|| anyhow!("missing required argument"))?;

        info!("Executing SPARQL-star query on: {}", data_path);
        execute_query(data_path, query_input, output_format)
    }

    fn troubleshoot_command(&self, matches: &ArgMatches) -> Result<()> {
        let error_input = matches
            .get_one::<String>("error")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let output_path = matches.get_one::<String>("output");
        run_troubleshoot(error_input, output_path)
    }

    fn migrate_command(&self, matches: &ArgMatches) -> Result<()> {
        let source_file = matches
            .get_one::<String>("source")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let output_file = matches
            .get_one::<String>("output")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let source_format = matches
            .get_one::<String>("source-format")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let plan_only = matches.get_flag("plan");
        run_migrate(
            source_file,
            output_file,
            source_format,
            plan_only,
            self.quiet,
        )
    }

    fn doctor_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_file = matches
            .get_one::<String>("input")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let report_path = matches.get_one::<String>("report");
        let auto_fix = matches.get_flag("fix");
        run_doctor(input_file, report_path, auto_fix, self.quiet)
    }

    fn profile_command(&self, matches: &ArgMatches) -> Result<()> {
        let input_path = matches
            .get_one::<String>("input")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let operations = matches
            .get_one::<String>("operations")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let iterations: usize = matches
            .get_one::<String>("iterations")
            .ok_or_else(|| anyhow!("missing required argument"))?
            .parse()?;
        let report_path = matches.get_one::<String>("output");
        run_profile(input_path, operations, iterations, report_path, self.quiet)
    }

    fn profile_report_command(&self, matches: &ArgMatches) -> Result<()> {
        let data_path = matches
            .get_one::<String>("data")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        let output_path = matches.get_one::<String>("output");
        let format = matches
            .get_one::<String>("format")
            .ok_or_else(|| anyhow!("missing required argument"))?;
        run_profile_report(data_path, output_path, format)
    }
}

impl Default for StarCli {
    fn default() -> Self {
        Self::new()
    }
}
