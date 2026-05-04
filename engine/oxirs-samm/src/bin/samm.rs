//! `samm` — OxiRS SAMM command-line tool.
//!
//! # Usage
//!
//! ```text
//! samm generate --target <TARGET> --input <FILE> --output <FILE>
//! ```
//!
//! ## Sub-commands
//!
//! | Sub-command | Description |
//! |-------------|-------------|
//! | `generate`  | Generate code from a SAMM Turtle model |
//!
//! ## `generate` flags
//!
//! | Flag | Required | Description |
//! |------|----------|-------------|
//! | `--target` | yes | One of: `java`, `typescript`, `python`, `openapi`, `json-schema` |
//! | `--input`  | yes | Path to the SAMM Turtle (`.ttl`) model file |
//! | `--output` | yes | Destination file for the generated output |
//!
//! ## Examples
//!
//! ```text
//! samm generate --target java --input Movement.ttl --output Movement.java
//! samm generate --target typescript --input Movement.ttl --output movement.ts
//! samm generate --target python --input Movement.ttl --output movement.py
//! samm generate --target openapi --input Movement.ttl --output openapi.json
//! samm generate --target json-schema --input Movement.ttl --output schema.json
//! ```

use oxirs_samm::cli::generate::{run_generate, GenerateArgs, GenerateTarget};
use std::path::PathBuf;
use std::str::FromStr;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_help();
        return Ok(());
    }

    match args[1].as_str() {
        "generate" => {
            let gen_args = parse_generate_args(&args[2..])?;
            run_generate(&gen_args)?;
            println!(
                "Generated {} output → {}",
                gen_args.target,
                gen_args.output.display()
            );
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        "version" | "--version" | "-V" => {
            println!("samm {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        unknown => {
            Err(format!("unknown sub-command: '{unknown}'; run 'samm help' for usage").into())
        }
    }
}

/// Parse the positional arguments after `samm generate`.
fn parse_generate_args(args: &[String]) -> Result<GenerateArgs, Box<dyn std::error::Error>> {
    let mut target: Option<GenerateTarget> = None;
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--target" | "-t" => {
                i += 1;
                let val = args.get(i).ok_or("--target requires a value")?;
                target = Some(GenerateTarget::from_str(val)?);
            }
            "--input" | "-i" => {
                i += 1;
                let val = args.get(i).ok_or("--input requires a value")?;
                input = Some(PathBuf::from(val));
            }
            "--output" | "-o" => {
                i += 1;
                let val = args.get(i).ok_or("--output requires a value")?;
                output = Some(PathBuf::from(val));
            }
            "--help" | "-h" => {
                print_generate_help();
                std::process::exit(0);
            }
            flag => {
                return Err(format!(
                    "unknown flag: '{flag}'; run 'samm generate --help' for usage"
                )
                .into());
            }
        }
        i += 1;
    }

    let target = target.ok_or("--target is required")?;
    let input = input.ok_or("--input is required")?;
    let output = output.ok_or("--output is required")?;

    Ok(GenerateArgs {
        target,
        input,
        output,
    })
}

fn print_help() {
    println!(
        "samm {ver} — OxiRS SAMM command-line tool

USAGE:
    samm <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    generate    Generate code from a SAMM Turtle model
    help        Show this help message
    version     Print version information

Run 'samm <SUBCOMMAND> --help' for per-command options.",
        ver = env!("CARGO_PKG_VERSION"),
    );
}

fn print_generate_help() {
    println!(
        "samm generate — Generate code from a SAMM Turtle model

USAGE:
    samm generate --target <TARGET> --input <FILE> --output <FILE>

OPTIONS:
    -t, --target <TARGET>   Code-generation target (required)
                            java | typescript | python | openapi | json-schema
    -i, --input  <FILE>     Path to the SAMM Turtle model file (required)
    -o, --output <FILE>     Destination file path for the generated output (required)
    -h, --help              Show this help message

EXAMPLES:
    samm generate --target java --input Movement.ttl --output Movement.java
    samm generate --target typescript --input Movement.ttl --output movement.ts
    samm generate --target python --input Movement.ttl --output movement.py
    samm generate --target openapi --input Movement.ttl --output openapi.json
    samm generate --target json-schema --input Movement.ttl --output schema.json"
    );
}
