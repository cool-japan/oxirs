//! CLI Documentation Generator
//!
//! Automatically generates comprehensive documentation for all OxiRS CLI commands.
//! Supports multiple output formats: Markdown, HTML, Man pages, and Plain Text.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Documentation format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocFormat {
    /// Markdown format (for GitHub, docs sites)
    Markdown,
    /// HTML format (for web documentation)
    Html,
    /// Man page format (for Unix man pages)
    Man,
    /// Plain text format
    Text,
}

impl std::str::FromStr for DocFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "markdown" | "md" => Ok(DocFormat::Markdown),
            "html" => Ok(DocFormat::Html),
            "man" => Ok(DocFormat::Man),
            "text" | "txt" => Ok(DocFormat::Text),
            _ => Err(format!("Unknown documentation format: {}", s)),
        }
    }
}

/// Command documentation structure
#[derive(Debug, Clone)]
pub struct CommandDoc {
    /// Command name
    pub name: String,
    /// Brief description
    pub description: String,
    /// Detailed usage information
    pub usage: String,
    /// Command arguments
    pub arguments: Vec<ArgumentDoc>,
    /// Command options/flags
    pub options: Vec<OptionDoc>,
    /// Usage examples
    pub examples: Vec<ExampleDoc>,
    /// Related commands
    pub see_also: Vec<String>,
}

/// Argument documentation
#[derive(Debug, Clone)]
pub struct ArgumentDoc {
    /// Argument name
    pub name: String,
    /// Argument description
    pub description: String,
    /// Whether the argument is required
    pub required: bool,
    /// Default value (if any)
    pub default: Option<String>,
}

/// Option/flag documentation
#[derive(Debug, Clone)]
pub struct OptionDoc {
    /// Short flag (e.g., -v)
    pub short: Option<String>,
    /// Long flag (e.g., --verbose)
    pub long: String,
    /// Option description
    pub description: String,
    /// Value name (if option takes a value)
    pub value_name: Option<String>,
    /// Default value (if any)
    pub default: Option<String>,
}

/// Example documentation
#[derive(Debug, Clone)]
pub struct ExampleDoc {
    /// Example description
    pub description: String,
    /// Example command
    pub command: String,
}

/// Documentation generator
pub struct DocGenerator {
    commands: Vec<CommandDoc>,
}

impl DocGenerator {
    /// Create a new documentation generator
    pub fn new() -> Self {
        Self {
            commands: Self::get_all_commands(),
        }
    }

    /// Get documentation for all commands
    fn get_all_commands() -> Vec<CommandDoc> {
        vec![
            Self::query_command(),
            Self::update_command(),
            Self::import_command(),
            Self::export_command(),
            Self::migrate_command(),
            Self::serve_command(),
            Self::batch_command(),
            Self::interactive_command(),
            Self::config_command(),
            Self::benchmark_command(),
            Self::generate_command(),
            Self::tdbstats_command(),
            Self::tdbbackup_command(),
            Self::tdbcompact_command(),
        ]
    }

    /// Generate documentation in the specified format
    pub fn generate(&self, format: DocFormat) -> Result<String, Box<dyn std::error::Error>> {
        match format {
            DocFormat::Markdown => Ok(self.generate_markdown()),
            DocFormat::Html => Ok(self.generate_html()),
            DocFormat::Man => Ok(self.generate_man()),
            DocFormat::Text => Ok(self.generate_text()),
        }
    }

    /// Write documentation to a file
    pub fn write_to_file(
        &self,
        path: PathBuf,
        format: DocFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let content = self.generate(format)?;
        let mut file = fs::File::create(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }

    /// Generate Markdown documentation
    fn generate_markdown(&self) -> String {
        let mut output = String::new();

        output.push_str("# OxiRS CLI Reference\n\n");
        output.push_str(
            "Comprehensive command-line interface for OxiRS semantic web operations.\n\n",
        );
        output.push_str("## Table of Contents\n\n");

        // Table of contents
        for cmd in &self.commands {
            output.push_str(&format!("- [{}](#{})\n", cmd.name, cmd.name.to_lowercase()));
        }
        output.push_str("\n---\n\n");

        // Command details
        for cmd in &self.commands {
            output.push_str(&format!("## {}\n\n", cmd.name));
            output.push_str(&format!("{}\n\n", cmd.description));

            output.push_str("### Usage\n\n");
            output.push_str(&format!("```bash\n{}\n```\n\n", cmd.usage));

            if !cmd.arguments.is_empty() {
                output.push_str("### Arguments\n\n");
                for arg in &cmd.arguments {
                    let required = if arg.required {
                        " (required)"
                    } else {
                        " (optional)"
                    };
                    output.push_str(&format!(
                        "- **{}**{}: {}",
                        arg.name, required, arg.description
                    ));
                    if let Some(default) = &arg.default {
                        output.push_str(&format!(" (default: `{}`)", default));
                    }
                    output.push('\n');
                }
                output.push('\n');
            }

            if !cmd.options.is_empty() {
                output.push_str("### Options\n\n");
                for opt in &cmd.options {
                    let short = opt
                        .short
                        .as_ref()
                        .map(|s| format!("{}, ", s))
                        .unwrap_or_default();
                    let value = opt
                        .value_name
                        .as_ref()
                        .map(|v| format!(" <{}>", v))
                        .unwrap_or_default();
                    output.push_str(&format!(
                        "- **{}{}{}**: {}",
                        short, opt.long, value, opt.description
                    ));
                    if let Some(default) = &opt.default {
                        output.push_str(&format!(" (default: `{}`)", default));
                    }
                    output.push('\n');
                }
                output.push('\n');
            }

            if !cmd.examples.is_empty() {
                output.push_str("### Examples\n\n");
                for example in &cmd.examples {
                    output.push_str(&format!("**{}**\n\n", example.description));
                    output.push_str(&format!("```bash\n{}\n```\n\n", example.command));
                }
            }

            if !cmd.see_also.is_empty() {
                output.push_str("### See Also\n\n");
                for related in &cmd.see_also {
                    output.push_str(&format!("- [{}](#{})\n", related, related.to_lowercase()));
                }
                output.push('\n');
            }

            output.push_str("---\n\n");
        }

        output
    }

    /// Generate HTML documentation
    fn generate_html(&self) -> String {
        let mut output = String::new();

        output.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        output.push_str("    <meta charset=\"UTF-8\">\n");
        output.push_str("    <title>OxiRS CLI Reference</title>\n");
        output.push_str("    <style>\n");
        output.push_str("        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }\n");
        output.push_str("        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }\n");
        output.push_str(
            "        h2 { color: #34495e; margin-top: 40px; border-bottom: 2px solid #95a5a6; }\n",
        );
        output.push_str("        h3 { color: #7f8c8d; }\n");
        output.push_str(
            "        code { background: #ecf0f1; padding: 2px 6px; border-radius: 3px; }\n",
        );
        output.push_str("        pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; }\n");
        output.push_str("        .command { background: #fff; border: 1px solid #bdc3c7; padding: 20px; margin: 20px 0; border-radius: 5px; }\n");
        output
            .push_str("        .toc { background: #ecf0f1; padding: 20px; border-radius: 5px; }\n");
        output.push_str("        ul { line-height: 1.8; }\n");
        output.push_str("        .required { color: #e74c3c; font-weight: bold; }\n");
        output.push_str("        .optional { color: #95a5a6; }\n");
        output.push_str("    </style>\n");
        output.push_str("</head>\n<body>\n");

        output.push_str("    <h1>OxiRS CLI Reference</h1>\n");
        output.push_str("    <p>Comprehensive command-line interface for OxiRS semantic web operations.</p>\n\n");

        output.push_str("    <div class=\"toc\">\n");
        output.push_str("        <h2>Table of Contents</h2>\n");
        output.push_str("        <ul>\n");
        for cmd in &self.commands {
            output.push_str(&format!(
                "            <li><a href=\"#{}\">{}</a></li>\n",
                cmd.name.to_lowercase(),
                cmd.name
            ));
        }
        output.push_str("        </ul>\n");
        output.push_str("    </div>\n\n");

        for cmd in &self.commands {
            output.push_str(&format!(
                "    <div class=\"command\" id=\"{}\">\n",
                cmd.name.to_lowercase()
            ));
            output.push_str(&format!("        <h2>{}</h2>\n", cmd.name));
            output.push_str(&format!("        <p>{}</p>\n\n", cmd.description));

            output.push_str("        <h3>Usage</h3>\n");
            output.push_str(&format!("        <pre>{}</pre>\n\n", cmd.usage));

            if !cmd.arguments.is_empty() {
                output.push_str("        <h3>Arguments</h3>\n");
                output.push_str("        <ul>\n");
                for arg in &cmd.arguments {
                    let required_class = if arg.required { "required" } else { "optional" };
                    let required_text = if arg.required {
                        "(required)"
                    } else {
                        "(optional)"
                    };
                    output.push_str(&format!(
                        "            <li><strong>{}</strong> <span class=\"{}\">{}</span>: {}",
                        arg.name, required_class, required_text, arg.description
                    ));
                    if let Some(default) = &arg.default {
                        output.push_str(&format!(" <code>default: {}</code>", default));
                    }
                    output.push_str("</li>\n");
                }
                output.push_str("        </ul>\n\n");
            }

            if !cmd.options.is_empty() {
                output.push_str("        <h3>Options</h3>\n");
                output.push_str("        <ul>\n");
                for opt in &cmd.options {
                    let short = opt
                        .short
                        .as_ref()
                        .map(|s| format!("{}, ", s))
                        .unwrap_or_default();
                    let value = opt
                        .value_name
                        .as_ref()
                        .map(|v| format!(" &lt;{}&gt;", v))
                        .unwrap_or_default();
                    output.push_str(&format!(
                        "            <li><code>{}{}{}</code>: {}",
                        short, opt.long, value, opt.description
                    ));
                    if let Some(default) = &opt.default {
                        output.push_str(&format!(" <code>default: {}</code>", default));
                    }
                    output.push_str("</li>\n");
                }
                output.push_str("        </ul>\n\n");
            }

            if !cmd.examples.is_empty() {
                output.push_str("        <h3>Examples</h3>\n");
                for example in &cmd.examples {
                    output.push_str(&format!(
                        "        <p><strong>{}</strong></p>\n",
                        example.description
                    ));
                    output.push_str(&format!("        <pre>{}</pre>\n", example.command));
                }
            }

            output.push_str("    </div>\n\n");
        }

        output.push_str("</body>\n</html>\n");
        output
    }

    /// Generate man page documentation
    fn generate_man(&self) -> String {
        let mut output = String::new();

        output.push_str(".TH OXIRS 1 \"2025\" \"OxiRS v0.1.0-rc.1\" \"OxiRS Manual\"\n");
        output.push_str(".SH NAME\n");
        output.push_str("oxirs \\- Semantic web command-line interface\n");
        output.push_str(".SH SYNOPSIS\n");
        output.push_str(".B oxirs\n");
        output.push_str("[\\fIOPTIONS\\fR] \\fICOMMAND\\fR [\\fIARGS\\fR]\n");
        output.push_str(".SH DESCRIPTION\n");
        output.push_str("OxiRS is a comprehensive command-line interface for semantic web operations including SPARQL queries, RDF data management, and triplestore administration.\n");

        output.push_str(".SH COMMANDS\n");
        for cmd in &self.commands {
            output.push_str(&format!(".TP\n.B {}\n", cmd.name));
            output.push_str(&format!("{}\n", cmd.description));
        }

        output.push_str(".SH AUTHOR\n");
        output.push_str("Written by the OxiRS development team.\n");
        output.push_str(".SH REPORTING BUGS\n");
        output.push_str("Report bugs at: https://github.com/cool-japan/oxirs/issues\n");

        output
    }

    /// Generate plain text documentation
    fn generate_text(&self) -> String {
        let mut output = String::new();

        output.push_str(
            "================================================================================\n",
        );
        output.push_str("                          OxiRS CLI REFERENCE\n");
        output.push_str(
            "================================================================================\n\n",
        );

        for cmd in &self.commands {
            output.push_str(&format!("COMMAND: {}\n", cmd.name.to_uppercase()));
            output.push_str(&format!("{}\n\n", "=".repeat(80)));
            output.push_str(&format!("{}\n\n", cmd.description));

            output.push_str(&format!("Usage:\n  {}\n\n", cmd.usage));

            if !cmd.arguments.is_empty() {
                output.push_str("Arguments:\n");
                for arg in &cmd.arguments {
                    let required = if arg.required { "REQUIRED" } else { "OPTIONAL" };
                    output.push_str(&format!("  {} [{}]\n", arg.name, required));
                    output.push_str(&format!("    {}\n", arg.description));
                    if let Some(default) = &arg.default {
                        output.push_str(&format!("    Default: {}\n", default));
                    }
                }
                output.push('\n');
            }

            if !cmd.options.is_empty() {
                output.push_str("Options:\n");
                for opt in &cmd.options {
                    let short = opt
                        .short
                        .as_ref()
                        .map(|s| format!("{}, ", s))
                        .unwrap_or_default();
                    let value = opt
                        .value_name
                        .as_ref()
                        .map(|v| format!(" <{}>", v))
                        .unwrap_or_default();
                    output.push_str(&format!("  {}{}{}\n", short, opt.long, value));
                    output.push_str(&format!("    {}\n", opt.description));
                    if let Some(default) = &opt.default {
                        output.push_str(&format!("    Default: {}\n", default));
                    }
                }
                output.push('\n');
            }

            if !cmd.examples.is_empty() {
                output.push_str("Examples:\n");
                for example in &cmd.examples {
                    output.push_str(&format!("  {}\n", example.description));
                    output.push_str(&format!("    $ {}\n\n", example.command));
                }
            }

            output.push_str(&format!("{}\n\n", "=".repeat(80)));
        }

        output
    }

    // Command definitions (abbreviated for brevity - in real implementation, these would be complete)

    fn query_command() -> CommandDoc {
        CommandDoc {
            name: "query".to_string(),
            description: "Execute SPARQL queries against RDF datasets".to_string(),
            usage: "oxirs query [OPTIONS] <DATASET> <QUERY>".to_string(),
            arguments: vec![
                ArgumentDoc {
                    name: "DATASET".to_string(),
                    description: "Path to the RDF dataset or configuration file".to_string(),
                    required: true,
                    default: None,
                },
                ArgumentDoc {
                    name: "QUERY".to_string(),
                    description: "SPARQL query string or file path (with --file)".to_string(),
                    required: true,
                    default: None,
                },
            ],
            options: vec![
                OptionDoc {
                    short: Some("-f".to_string()),
                    long: "--file".to_string(),
                    description: "Read query from file instead of command line".to_string(),
                    value_name: None,
                    default: None,
                },
                OptionDoc {
                    short: Some("-o".to_string()),
                    long: "--output".to_string(),
                    description: "Output format (json, csv, tsv, table, xml, html, markdown, pdf, template-*)".to_string(),
                    value_name: Some("FORMAT".to_string()),
                    default: Some("table".to_string()),
                },
                OptionDoc {
                    short: Some("-v".to_string()),
                    long: "--verbose".to_string(),
                    description: "Enable verbose output with query complexity analysis".to_string(),
                    value_name: None,
                    default: None,
                },
            ],
            examples: vec![
                ExampleDoc {
                    description: "Simple SELECT query".to_string(),
                    command: "oxirs query my_dataset.tdb \"SELECT * WHERE { ?s ?p ?o } LIMIT 10\"".to_string(),
                },
                ExampleDoc {
                    description: "Query from file with JSON output".to_string(),
                    command: "oxirs query --file query.sparql --output json my_dataset.tdb".to_string(),
                },
                ExampleDoc {
                    description: "Query with custom template".to_string(),
                    command: "oxirs query --output template-markdown my_dataset.tdb query.sparql".to_string(),
                },
            ],
            see_also: vec!["update".to_string(), "interactive".to_string()],
        }
    }

    fn update_command() -> CommandDoc {
        CommandDoc {
            name: "update".to_string(),
            description: "Execute SPARQL UPDATE operations (INSERT, DELETE, MODIFY)".to_string(),
            usage: "oxirs update [OPTIONS] <DATASET> <UPDATE>".to_string(),
            arguments: vec![
                ArgumentDoc {
                    name: "DATASET".to_string(),
                    description: "Path to the RDF dataset".to_string(),
                    required: true,
                    default: None,
                },
                ArgumentDoc {
                    name: "UPDATE".to_string(),
                    description: "SPARQL UPDATE statement".to_string(),
                    required: true,
                    default: None,
                },
            ],
            options: vec![
                OptionDoc {
                    short: Some("-f".to_string()),
                    long: "--file".to_string(),
                    description: "Read update from file".to_string(),
                    value_name: None,
                    default: None,
                },
            ],
            examples: vec![
                ExampleDoc {
                    description: "Insert triples".to_string(),
                    command: "oxirs update dataset.tdb \"INSERT DATA { <http://ex.org/s> <http://ex.org/p> 'object' }\"".to_string(),
                },
            ],
            see_also: vec!["query".to_string()],
        }
    }

    fn import_command() -> CommandDoc {
        CommandDoc {
            name: "import".to_string(),
            description: "Import RDF data from files into a dataset".to_string(),
            usage: "oxirs import [OPTIONS] <DATASET> <FILE>".to_string(),
            arguments: vec![
                ArgumentDoc {
                    name: "DATASET".to_string(),
                    description: "Target dataset path".to_string(),
                    required: true,
                    default: None,
                },
                ArgumentDoc {
                    name: "FILE".to_string(),
                    description: "RDF file to import".to_string(),
                    required: true,
                    default: None,
                },
            ],
            options: vec![
                OptionDoc {
                    short: Some("-f".to_string()),
                    long: "--format".to_string(),
                    description: "RDF format (turtle, ntriples, rdfxml, jsonld, trig, nquads, n3)"
                        .to_string(),
                    value_name: Some("FORMAT".to_string()),
                    default: Some("auto-detect".to_string()),
                },
                OptionDoc {
                    short: Some("-g".to_string()),
                    long: "--graph".to_string(),
                    description: "Target named graph URI".to_string(),
                    value_name: Some("URI".to_string()),
                    default: None,
                },
            ],
            examples: vec![ExampleDoc {
                description: "Import Turtle file".to_string(),
                command: "oxirs import --format turtle dataset.tdb data.ttl".to_string(),
            }],
            see_also: vec!["export".to_string(), "batch".to_string()],
        }
    }

    fn export_command() -> CommandDoc {
        CommandDoc {
            name: "export".to_string(),
            description: "Export RDF data from a dataset to a file".to_string(),
            usage: "oxirs export [OPTIONS] <DATASET> <OUTPUT>".to_string(),
            arguments: vec![
                ArgumentDoc {
                    name: "DATASET".to_string(),
                    description: "Source dataset path".to_string(),
                    required: true,
                    default: None,
                },
                ArgumentDoc {
                    name: "OUTPUT".to_string(),
                    description: "Output file path".to_string(),
                    required: true,
                    default: None,
                },
            ],
            options: vec![OptionDoc {
                short: Some("-f".to_string()),
                long: "--format".to_string(),
                description: "Output RDF format".to_string(),
                value_name: Some("FORMAT".to_string()),
                default: Some("turtle".to_string()),
            }],
            examples: vec![ExampleDoc {
                description: "Export to N-Triples".to_string(),
                command: "oxirs export --format ntriples dataset.tdb output.nt".to_string(),
            }],
            see_also: vec!["import".to_string()],
        }
    }

    fn migrate_command() -> CommandDoc {
        CommandDoc {
            name: "migrate".to_string(),
            description: "Migrate data between triplestores and formats".to_string(),
            usage: "oxirs migrate <SUBCOMMAND>".to_string(),
            arguments: vec![],
            options: vec![],
            examples: vec![ExampleDoc {
                description: "Migrate from Virtuoso".to_string(),
                command: "oxirs migrate from-virtuoso http://localhost:8890/sparql output.tdb"
                    .to_string(),
            }],
            see_also: vec!["import".to_string(), "export".to_string()],
        }
    }

    fn serve_command() -> CommandDoc {
        CommandDoc {
            name: "serve".to_string(),
            description: "Start SPARQL HTTP server (Fuseki-compatible)".to_string(),
            usage: "oxirs serve [OPTIONS] <DATASET>".to_string(),
            arguments: vec![ArgumentDoc {
                name: "DATASET".to_string(),
                description: "Dataset to serve".to_string(),
                required: true,
                default: None,
            }],
            options: vec![OptionDoc {
                short: Some("-p".to_string()),
                long: "--port".to_string(),
                description: "HTTP port".to_string(),
                value_name: Some("PORT".to_string()),
                default: Some("3030".to_string()),
            }],
            examples: vec![ExampleDoc {
                description: "Serve dataset on port 3030".to_string(),
                command: "oxirs serve --port 3030 dataset.tdb".to_string(),
            }],
            see_also: vec!["query".to_string()],
        }
    }

    fn batch_command() -> CommandDoc {
        CommandDoc {
            name: "batch".to_string(),
            description: "Batch import multiple RDF files in parallel".to_string(),
            usage: "oxirs batch [OPTIONS] <DATASET> <FILES>...".to_string(),
            arguments: vec![
                ArgumentDoc {
                    name: "DATASET".to_string(),
                    description: "Target dataset".to_string(),
                    required: true,
                    default: None,
                },
                ArgumentDoc {
                    name: "FILES".to_string(),
                    description: "RDF files to import".to_string(),
                    required: true,
                    default: None,
                },
            ],
            options: vec![OptionDoc {
                short: Some("-j".to_string()),
                long: "--parallel".to_string(),
                description: "Number of parallel workers".to_string(),
                value_name: Some("N".to_string()),
                default: Some("4".to_string()),
            }],
            examples: vec![ExampleDoc {
                description: "Import 10 files in parallel".to_string(),
                command: "oxirs batch --parallel 8 dataset.tdb *.ttl".to_string(),
            }],
            see_also: vec!["import".to_string()],
        }
    }

    fn interactive_command() -> CommandDoc {
        CommandDoc {
            name: "interactive".to_string(),
            description: "Start interactive SPARQL REPL with history and autocomplete".to_string(),
            usage: "oxirs interactive <DATASET>".to_string(),
            arguments: vec![ArgumentDoc {
                name: "DATASET".to_string(),
                description: "Dataset for queries".to_string(),
                required: true,
                default: None,
            }],
            options: vec![],
            examples: vec![ExampleDoc {
                description: "Start interactive mode".to_string(),
                command: "oxirs interactive dataset.tdb".to_string(),
            }],
            see_also: vec!["query".to_string()],
        }
    }

    fn config_command() -> CommandDoc {
        CommandDoc {
            name: "config".to_string(),
            description: "Manage OxiRS configuration files".to_string(),
            usage: "oxirs config <SUBCOMMAND>".to_string(),
            arguments: vec![],
            options: vec![],
            examples: vec![ExampleDoc {
                description: "Initialize configuration".to_string(),
                command: "oxirs config init".to_string(),
            }],
            see_also: vec![],
        }
    }

    fn benchmark_command() -> CommandDoc {
        CommandDoc {
            name: "benchmark".to_string(),
            description: "Run SPARQL benchmarks (SP2Bench, WatDiv, LDBC, BSBM)".to_string(),
            usage: "oxirs benchmark <SUBCOMMAND>".to_string(),
            arguments: vec![],
            options: vec![],
            examples: vec![ExampleDoc {
                description: "Run SP2Bench".to_string(),
                command: "oxirs benchmark run --suite sp2bench dataset.tdb".to_string(),
            }],
            see_also: vec!["query".to_string()],
        }
    }

    fn generate_command() -> CommandDoc {
        CommandDoc {
            name: "generate".to_string(),
            description: "Generate synthetic RDF datasets for testing".to_string(),
            usage: "oxirs generate [OPTIONS] <OUTPUT>".to_string(),
            arguments: vec![ArgumentDoc {
                name: "OUTPUT".to_string(),
                description: "Output file path".to_string(),
                required: true,
                default: None,
            }],
            options: vec![OptionDoc {
                short: Some("-s".to_string()),
                long: "--size".to_string(),
                description: "Dataset size (tiny, small, medium, large, xlarge)".to_string(),
                value_name: Some("SIZE".to_string()),
                default: Some("small".to_string()),
            }],
            examples: vec![ExampleDoc {
                description: "Generate medium dataset".to_string(),
                command: "oxirs generate --size medium test_data.ttl".to_string(),
            }],
            see_also: vec!["import".to_string()],
        }
    }

    fn tdbstats_command() -> CommandDoc {
        CommandDoc {
            name: "tdbstats".to_string(),
            description: "Display detailed statistics for TDB2 datasets".to_string(),
            usage: "oxirs tdbstats [OPTIONS] <DATASET>".to_string(),
            arguments: vec![ArgumentDoc {
                name: "DATASET".to_string(),
                description: "Dataset path".to_string(),
                required: true,
                default: None,
            }],
            options: vec![OptionDoc {
                short: Some("-f".to_string()),
                long: "--format".to_string(),
                description: "Output format (text, json, csv)".to_string(),
                value_name: Some("FORMAT".to_string()),
                default: Some("text".to_string()),
            }],
            examples: vec![ExampleDoc {
                description: "Show dataset statistics".to_string(),
                command: "oxirs tdbstats --format json dataset.tdb".to_string(),
            }],
            see_also: vec!["tdbcompact".to_string()],
        }
    }

    fn tdbbackup_command() -> CommandDoc {
        CommandDoc {
            name: "tdbbackup".to_string(),
            description: "Create compressed backups of TDB2 datasets".to_string(),
            usage: "oxirs tdbbackup [OPTIONS] <DATASET> <BACKUP>".to_string(),
            arguments: vec![
                ArgumentDoc {
                    name: "DATASET".to_string(),
                    description: "Source dataset".to_string(),
                    required: true,
                    default: None,
                },
                ArgumentDoc {
                    name: "BACKUP".to_string(),
                    description: "Backup output path".to_string(),
                    required: true,
                    default: None,
                },
            ],
            options: vec![OptionDoc {
                short: Some("-c".to_string()),
                long: "--compress".to_string(),
                description: "Enable compression".to_string(),
                value_name: None,
                default: None,
            }],
            examples: vec![ExampleDoc {
                description: "Create compressed backup".to_string(),
                command: "oxirs tdbbackup --compress dataset.tdb backup.tdb.gz".to_string(),
            }],
            see_also: vec!["tdbcompact".to_string()],
        }
    }

    fn tdbcompact_command() -> CommandDoc {
        CommandDoc {
            name: "tdbcompact".to_string(),
            description: "Compact and optimize TDB2 datasets".to_string(),
            usage: "oxirs tdbcompact [OPTIONS] <DATASET>".to_string(),
            arguments: vec![ArgumentDoc {
                name: "DATASET".to_string(),
                description: "Dataset to compact".to_string(),
                required: true,
                default: None,
            }],
            options: vec![],
            examples: vec![ExampleDoc {
                description: "Compact dataset".to_string(),
                command: "oxirs tdbcompact dataset.tdb".to_string(),
            }],
            see_also: vec!["tdbstats".to_string()],
        }
    }
}

impl Default for DocGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_format_parsing() {
        assert_eq!(
            "markdown".parse::<DocFormat>().unwrap(),
            DocFormat::Markdown
        );
        assert_eq!("md".parse::<DocFormat>().unwrap(), DocFormat::Markdown);
        assert_eq!("html".parse::<DocFormat>().unwrap(), DocFormat::Html);
        assert_eq!("man".parse::<DocFormat>().unwrap(), DocFormat::Man);
        assert_eq!("text".parse::<DocFormat>().unwrap(), DocFormat::Text);
        assert!("invalid".parse::<DocFormat>().is_err());
    }

    #[test]
    fn test_generate_markdown() {
        let generator = DocGenerator::new();
        let markdown = generator.generate(DocFormat::Markdown).unwrap();

        assert!(markdown.contains("# OxiRS CLI Reference"));
        assert!(markdown.contains("## query"));
        assert!(markdown.contains("### Usage"));
        assert!(markdown.contains("### Examples"));
    }

    #[test]
    fn test_generate_html() {
        let generator = DocGenerator::new();
        let html = generator.generate(DocFormat::Html).unwrap();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<title>OxiRS CLI Reference</title>"));
        assert!(html.contains("query"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_generate_man() {
        let generator = DocGenerator::new();
        let man = generator.generate(DocFormat::Man).unwrap();

        assert!(man.contains(".TH OXIRS"));
        assert!(man.contains(".SH NAME"));
        assert!(man.contains(".SH DESCRIPTION"));
    }

    #[test]
    fn test_generate_text() {
        let generator = DocGenerator::new();
        let text = generator.generate(DocFormat::Text).unwrap();

        assert!(text.contains("OxiRS CLI REFERENCE"));
        assert!(text.contains("COMMAND: QUERY"));
        assert!(text.contains("Usage:"));
    }

    #[test]
    fn test_all_commands_documented() {
        let generator = DocGenerator::new();
        assert!(generator.commands.len() >= 10);

        let command_names: Vec<&str> = generator.commands.iter().map(|c| c.name.as_str()).collect();
        assert!(command_names.contains(&"query"));
        assert!(command_names.contains(&"update"));
        assert!(command_names.contains(&"import"));
        assert!(command_names.contains(&"export"));
    }

    #[test]
    fn test_command_has_examples() {
        let generator = DocGenerator::new();
        let query_cmd = generator
            .commands
            .iter()
            .find(|c| c.name == "query")
            .unwrap();

        assert!(!query_cmd.examples.is_empty());
        assert!(query_cmd.examples[0].command.contains("oxirs query"));
    }
}
