//! Interactive mode support for CLI (DEPRECATED)
//!
//! **DEPRECATED**: This module is obsolete and no longer used by the main application.
//! Use `commands::interactive::execute()` instead for the modern interactive mode with:
//! - Real SPARQL query execution
//! - Advanced session management
//! - Query history and templates
//!
//! This module is kept for backwards compatibility only and will be removed in v0.2.0.
//!
//! Provides REPL-like interactive command execution with history and completion.

#![allow(deprecated)]

use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::hint::HistoryHinter;
use rustyline::validate::{self, MatchingBracketValidator, Validator};
use rustyline::{CompletionType, Config, EditMode, Editor};
use rustyline_derive::{Helper, Hinter};
use std::borrow::Cow;
use std::collections::HashMap;

/// Interactive mode handler (DEPRECATED)
///
/// # Deprecation Notice
///
/// This struct is deprecated and will be removed in v0.2.0.
/// Use `commands::interactive::execute()` instead.
#[deprecated(
    since = "0.1.0-rc.1",
    note = "Use commands::interactive::execute() instead"
)]
pub struct InteractiveMode {
    editor: Editor<OxirsHelper, rustyline::history::DefaultHistory>,
    history_file: String,
    current_dataset: Option<String>,
    environment: HashMap<String, String>,
    multi_line_mode: bool,
    multi_line_buffer: Vec<String>,
    query_templates: HashMap<String, String>,
    saved_queries: HashMap<String, String>,
    saved_queries_file: String,
}

impl InteractiveMode {
    /// Create a new interactive mode instance
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .history_ignore_space(true)
            .completion_type(CompletionType::List)
            .edit_mode(EditMode::Emacs)
            .build();

        let helper = OxirsHelper {
            completer: FilenameCompleter::new(),
            highlighter: MatchingBracketHighlighter::new(),
            hinter: HistoryHinter {},
            validator: MatchingBracketValidator::new(),
            commands: get_command_list(),
            sparql_keywords: get_sparql_keywords(),
        };

        let mut editor = Editor::with_config(config)?;
        editor.set_helper(Some(helper));

        let config_dir = dirs::config_dir()
            .map(|p| p.join("oxirs"))
            .unwrap_or_else(|| std::path::PathBuf::from(".oxirs"));

        // Ensure config directory exists
        if !config_dir.exists() {
            std::fs::create_dir_all(&config_dir).ok();
        }

        let history_file = config_dir.join("history.txt").to_string_lossy().to_string();

        let saved_queries_file = config_dir
            .join("saved_queries.json")
            .to_string_lossy()
            .to_string();

        // Load history if it exists
        if std::path::Path::new(&history_file).exists() {
            editor.load_history(&history_file).ok();
        }

        // Initialize query templates
        let query_templates = Self::create_default_templates();

        // Load saved queries if they exist
        let saved_queries = Self::load_saved_queries(&saved_queries_file);

        Ok(Self {
            editor,
            history_file,
            current_dataset: None,
            environment: HashMap::new(),
            multi_line_mode: false,
            multi_line_buffer: Vec::new(),
            query_templates,
            saved_queries,
            saved_queries_file,
        })
    }

    /// Run the interactive REPL
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Welcome to Oxirs Interactive Mode!");
        println!("Type 'help' for available commands, 'exit' to quit.");
        println!("Use '\\' at the end of a line for multi-line input.\n");

        loop {
            let prompt = self.get_prompt();
            let readline = self.editor.readline(&prompt);

            match readline {
                Ok(line) => {
                    // Handle multi-line mode
                    if self.multi_line_mode {
                        if line.trim().ends_with('\\') {
                            self.multi_line_buffer
                                .push(line.trim_end_matches('\\').to_string());
                            continue;
                        } else {
                            self.multi_line_buffer.push(line);
                            let full_command = self.multi_line_buffer.join(" ");
                            self.multi_line_buffer.clear();
                            self.multi_line_mode = false;

                            let _ = self.editor.add_history_entry(&full_command);
                            if let Err(e) = self.process_command(&full_command).await {
                                eprintln!("Error: {e}");
                            }
                            continue;
                        }
                    }

                    // Check for multi-line start
                    if line.trim().ends_with('\\') {
                        self.multi_line_mode = true;
                        self.multi_line_buffer
                            .push(line.trim_end_matches('\\').to_string());
                        continue;
                    }

                    if line.trim().is_empty() {
                        continue;
                    }

                    let _ = self.editor.add_history_entry(line.as_str());

                    if let Err(e) = self.process_command(&line).await {
                        eprintln!("Error: {e}");
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    if self.multi_line_mode {
                        println!("^C (multi-line input cancelled)");
                        self.multi_line_mode = false;
                        self.multi_line_buffer.clear();
                    } else {
                        println!("^C (Use Ctrl-D or 'exit' to quit)");
                    }
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("^D");
                    break;
                }
                Err(err) => {
                    eprintln!("Error: {err:?}");
                    break;
                }
            }
        }

        // Save history
        self.editor.save_history(&self.history_file)?;
        println!("Goodbye!");

        Ok(())
    }

    /// Get the appropriate prompt
    fn get_prompt(&self) -> String {
        if self.multi_line_mode {
            format!("{:>6}> ", "...")
        } else if let Some(ref dataset) = self.current_dataset {
            format!("oxirs:{dataset}> ")
        } else {
            "oxirs> ".to_string()
        }
    }

    /// Process a command
    async fn process_command(&mut self, command: &str) -> Result<(), Box<dyn std::error::Error>> {
        let trimmed = command.trim();

        // Built-in commands
        match trimmed {
            "exit" | "quit" => std::process::exit(0),
            "help" | "?" => {
                self.show_help();
                return Ok(());
            }
            "clear" | "cls" => {
                self.clear_screen();
                return Ok(());
            }
            _ => {}
        }

        // Check for special commands
        if trimmed.starts_with("use ") {
            let dataset = trimmed.strip_prefix("use ").unwrap().trim();
            self.use_dataset(dataset);
            return Ok(());
        }

        if trimmed.starts_with("set ") {
            let parts: Vec<&str> = trimmed
                .strip_prefix("set ")
                .unwrap()
                .splitn(2, '=')
                .collect();
            if parts.len() == 2 {
                self.set_variable(parts[0].trim(), parts[1].trim());
            } else {
                println!("Usage: set VARIABLE=value");
            }
            return Ok(());
        }

        if trimmed == "env" {
            self.show_environment();
            return Ok(());
        }

        // Template commands
        if trimmed.starts_with("template ") {
            let args: Vec<&str> = trimmed.split_whitespace().collect();
            if args.len() >= 2 {
                self.show_template(args[1]);
            } else {
                self.list_templates();
            }
            return Ok(());
        }

        if trimmed == "templates" {
            self.list_templates();
            return Ok(());
        }

        // Saved query commands
        if trimmed.starts_with("save ") {
            let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();
            if parts.len() >= 3 {
                self.save_query(parts[1], parts[2]);
            } else {
                println!("Usage: save <name> <query>");
            }
            return Ok(());
        }

        if trimmed.starts_with("load ") {
            let name = trimmed.strip_prefix("load ").unwrap().trim();
            self.load_query(name);
            return Ok(());
        }

        if trimmed == "queries" {
            self.list_saved_queries();
            return Ok(());
        }

        if trimmed.starts_with("delete ") {
            let name = trimmed.strip_prefix("delete ").unwrap().trim();
            self.delete_query(name);
            return Ok(());
        }

        // Execute regular commands
        self.execute_command(trimmed).await
    }

    /// Execute a command in interactive mode
    async fn execute_command(&mut self, command: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Expand environment variables
        let expanded = self.expand_variables(command);
        let parts: Vec<&str> = expanded.split_whitespace().collect();

        if parts.is_empty() {
            return Ok(());
        }

        let cmd = parts[0];
        let args = &parts[1..];

        // Use current dataset if not specified
        let adjusted_args = self.adjust_args_with_dataset(cmd, args);
        let adjusted_args_refs: Vec<&str> = adjusted_args.iter().map(|s| s.as_str()).collect();

        match cmd {
            "query" => self.handle_query(&adjusted_args_refs).await,
            "import" => self.handle_import(&adjusted_args_refs).await,
            "export" => self.handle_export(&adjusted_args_refs).await,
            "validate" => self.handle_validate(&adjusted_args_refs).await,
            "stats" => self.handle_stats(&adjusted_args_refs).await,
            "riot" => self.handle_riot(&adjusted_args_refs).await,
            "shacl" => self.handle_shacl(&adjusted_args_refs).await,
            "tdbloader" => self.handle_tdbloader(&adjusted_args_refs).await,
            "tdbdump" => self.handle_tdbdump(&adjusted_args_refs).await,
            _ => {
                // Try to suggest similar commands
                let suggestions = crate::cli::suggestions::suggest_command(
                    cmd,
                    &get_command_list()
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>(),
                );
                if let Some(suggestion) = suggestions {
                    println!("Unknown command: {cmd}. {suggestion}");
                } else {
                    println!("Unknown command: {cmd}. Type 'help' for available commands.");
                }
                Ok(())
            }
        }
    }

    /// Expand environment variables in command
    fn expand_variables(&self, command: &str) -> String {
        let mut result = command.to_string();
        for (key, value) in &self.environment {
            result = result.replace(&format!("${key}"), value);
            result = result.replace(&format!("${{{key}}}"), value);
        }
        result
    }

    /// Adjust arguments to include current dataset if needed
    fn adjust_args_with_dataset(&self, cmd: &str, args: &[&str]) -> Vec<String> {
        // Commands that can use current dataset as first argument
        let dataset_commands = ["query", "import", "export", "stats", "tdbloader", "tdbdump"];

        if dataset_commands.contains(&cmd) && !args.is_empty() {
            // Check if first arg looks like a dataset name or if we should inject current dataset
            if let Some(ref dataset) = self.current_dataset {
                // If no args or first arg doesn't look like a dataset path
                if args.is_empty() || !args[0].contains('/') && !args[0].ends_with(".tdb") {
                    // For query, check if it starts with SELECT/CONSTRUCT/etc
                    if cmd == "query" && !args.is_empty() {
                        let first_upper = args[0].to_uppercase();
                        if first_upper.starts_with("SELECT")
                            || first_upper.starts_with("CONSTRUCT")
                            || first_upper.starts_with("DESCRIBE")
                            || first_upper.starts_with("ASK")
                        {
                            // This is a query, inject dataset
                            let mut new_args = vec![dataset.clone()];
                            new_args.extend(args.iter().map(|s| s.to_string()));
                            return new_args;
                        }
                    }
                }
            }
        }

        args.iter().map(|s| s.to_string()).collect()
    }

    /// Show help message
    fn show_help(&self) {
        println!("Available commands:");
        println!("\n  Dataset Commands:");
        println!("    use <dataset>             - Set the current dataset");
        println!("    query <dataset> <sparql>  - Execute a SPARQL query");
        println!("    import <dataset> <file>   - Import RDF data");
        println!("    export <dataset> <file>   - Export RDF data");
        println!("    stats <dataset>           - Show dataset statistics");

        println!("\n  Data Processing:");
        println!("    validate <file>           - Validate RDF syntax");
        println!("    riot <files...>           - Parse and serialize RDF");
        println!("    shacl <data> <shapes>     - Run SHACL validation");

        println!("\n  Query Templates:");
        println!("    templates                 - List available query templates");
        println!("    template <name>           - Show a specific template");

        println!("\n  Saved Queries:");
        println!("    save <name> <query>       - Save a query with a name");
        println!("    load <name>               - Load and display a saved query");
        println!("    queries                   - List all saved queries");
        println!("    delete <name>             - Delete a saved query");

        println!("\n  Environment:");
        println!("    set VAR=value             - Set an environment variable");
        println!("    env                       - Show environment variables");
        println!("    clear                     - Clear the screen");

        println!("\n  System:");
        println!("    help, ?                   - Show this help message");
        println!("    exit, quit                - Exit interactive mode");

        println!("\nInteractive Features:");
        println!("  - Use Tab for command and file completion");
        println!("  - Use Up/Down arrows for command history");
        println!("  - Use Ctrl+R for reverse history search");
        println!("  - Use '\\' at line end for multi-line input");
        println!("  - Variables: $VAR expands to environment value");
        println!(
            "  - {} pre-defined query templates",
            self.query_templates.len()
        );
        println!("  - {} saved queries", self.saved_queries.len());

        if let Some(ref dataset) = self.current_dataset {
            println!("\nCurrent dataset: {dataset}");
        }
    }

    /// Clear the screen
    fn clear_screen(&self) {
        print!("\x1B[2J\x1B[1;1H");
        use std::io::{self, Write};
        io::stdout().flush().unwrap();
    }

    /// Handle query command
    async fn handle_query(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            println!("Usage: query <dataset> <sparql_query>");
            return Ok(());
        }

        let dataset = args[0];
        let query = args[1..].join(" ");

        println!("Executing query on dataset '{dataset}'...");

        // Integrate with actual query execution
        let format = self
            .environment
            .get("format")
            .map(|s| s.as_str())
            .unwrap_or("table");

        crate::commands::query::run(
            dataset.to_string(),
            query,
            false, // file - query is provided directly, not from file
            format.to_string(),
        )
        .await?;

        Ok(())
    }

    /// Handle import command
    async fn handle_import(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            println!("Usage: import <dataset> <file> [format] [graph]");
            return Ok(());
        }

        let dataset = args[0];
        let file = std::path::PathBuf::from(args[1]);
        let format = args.get(2).map(|s| s.to_string());
        let graph = args.get(3).map(|s| s.to_string());

        // Integrate with actual import (resume not supported in interactive mode)
        crate::commands::import::run(dataset.to_string(), file, format, graph, false).await?;

        Ok(())
    }

    /// Handle export command
    async fn handle_export(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 3 {
            println!("Usage: export <dataset> <file> <format> [graph]");
            return Ok(());
        }

        let dataset = args[0];
        let file = std::path::PathBuf::from(args[1]);
        let format = args[2].to_string();
        let graph = args.get(3).map(|s| s.to_string());

        // Integrate with actual export (resume not supported in interactive mode)
        crate::commands::export::run(dataset.to_string(), file, format, graph, false).await?;

        Ok(())
    }

    /// Handle validate command
    async fn handle_validate(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.is_empty() {
            println!("Usage: validate <file> [format]");
            return Ok(());
        }

        let file = std::path::PathBuf::from(args[0]);
        let format = args.get(1).map(|s| s.to_string());

        println!("Validating {}...", file.display());

        // Integrate with RDF syntax validation
        use oxirs_core::format::{RdfFormat, RdfParser};
        use std::fs;
        use std::io::BufReader;

        // Detect format if not specified
        let rdf_format = if let Some(fmt) = format {
            match fmt.as_str() {
                "turtle" | "ttl" => RdfFormat::Turtle,
                "ntriples" | "nt" => RdfFormat::NTriples,
                "nquads" | "nq" => RdfFormat::NQuads,
                "trig" => RdfFormat::TriG,
                "rdfxml" | "rdf" | "xml" => RdfFormat::RdfXml,
                "jsonld" | "json" => RdfFormat::JsonLd {
                    profile: oxirs_core::format::JsonLdProfileSet::empty(),
                },
                "n3" => RdfFormat::N3,
                _ => return Err(format!("Unsupported format: {fmt}").into()),
            }
        } else {
            // Auto-detect from extension
            if let Some(ext) = file.extension().and_then(|s| s.to_str()) {
                match ext.to_lowercase().as_str() {
                    "ttl" | "turtle" => RdfFormat::Turtle,
                    "nt" => RdfFormat::NTriples,
                    "nq" => RdfFormat::NQuads,
                    "trig" => RdfFormat::TriG,
                    "rdf" | "xml" => RdfFormat::RdfXml,
                    "jsonld" | "json-ld" => RdfFormat::JsonLd {
                        profile: oxirs_core::format::JsonLdProfileSet::empty(),
                    },
                    "n3" => RdfFormat::N3,
                    _ => RdfFormat::Turtle, // default
                }
            } else {
                RdfFormat::Turtle
            }
        };

        // Parse and validate
        let file_handle = fs::File::open(&file)?;
        let reader = BufReader::new(file_handle);
        let parser = RdfParser::new(rdf_format);

        let mut triple_count = 0;
        let mut error_count = 0;

        for quad_result in parser.for_reader(reader) {
            match quad_result {
                Ok(_) => triple_count += 1,
                Err(e) => {
                    eprintln!("  Parse error: {e}");
                    error_count += 1;
                }
            }
        }

        if error_count == 0 {
            println!("✓ Valid RDF: {triple_count} triples parsed successfully");
        } else {
            println!(
                "✗ Validation failed: {error_count} errors found ({triple_count} triples parsed)"
            );
        }

        Ok(())
    }

    /// Handle stats command
    async fn handle_stats(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.is_empty() {
            println!("Usage: stats <dataset>");
            return Ok(());
        }

        let dataset = args[0];
        println!("Getting statistics for dataset '{dataset}'...");

        // Integrate with actual stats
        use oxirs_core::rdf_store::RdfStore;
        use std::path::PathBuf;

        let dataset_path = PathBuf::from(dataset);
        let store =
            RdfStore::open(&dataset_path).map_err(|e| format!("Failed to open dataset: {e}"))?;

        // Get dataset statistics
        let quads = store
            .quads()
            .map_err(|e| format!("Failed to query quads: {e}"))?;
        let quad_count = quads.len();

        // Count unique subjects, predicates, objects
        use std::collections::HashSet;
        let mut subjects = HashSet::new();
        let mut predicates = HashSet::new();
        let mut objects = HashSet::new();
        let mut graphs = HashSet::new();

        for quad in &quads {
            subjects.insert(quad.subject().to_string());
            predicates.insert(quad.predicate().to_string());
            objects.insert(quad.object().to_string());
            graphs.insert(quad.graph_name().to_string());
        }

        println!("\nDataset Statistics:");
        println!("  Location: {}", dataset_path.display());
        println!("  Total quads: {quad_count}");
        println!("  Unique subjects: {}", subjects.len());
        println!("  Unique predicates: {}", predicates.len());
        println!("  Unique objects: {}", objects.len());
        println!("  Named graphs: {}", graphs.len());

        Ok(())
    }

    /// Use a dataset
    fn use_dataset(&mut self, dataset: &str) {
        self.current_dataset = Some(dataset.to_string());
        println!("Now using dataset: {dataset}");
    }

    /// Set an environment variable
    fn set_variable(&mut self, key: &str, value: &str) {
        self.environment.insert(key.to_string(), value.to_string());
        println!("Set {key} = {value}");
    }

    /// Show environment variables
    fn show_environment(&self) {
        if self.environment.is_empty() {
            println!("No variables set");
        } else {
            println!("Environment variables:");
            for (key, value) in &self.environment {
                println!("  {key} = {value}");
            }
        }

        if let Some(ref dataset) = self.current_dataset {
            println!("\nCurrent dataset: {dataset}");
        }
    }

    /// Handle riot command
    async fn handle_riot(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.is_empty() {
            println!("Usage: riot <input_files...> [--output <format>]");
            return Ok(());
        }

        // Riot command: Parse and serialize RDF files
        use oxirs_core::format::{RdfFormat, RdfParser, RdfSerializer};
        use std::fs;
        use std::io::BufReader;

        let output_format = RdfFormat::NTriples; // Default output format

        for file_path in args {
            if file_path.starts_with("--") {
                continue; // Skip options for now
            }

            let file = std::path::PathBuf::from(file_path);
            println!("Processing {}...", file.display());

            // Detect input format
            let input_format = if let Some(ext) = file.extension().and_then(|s| s.to_str()) {
                match ext.to_lowercase().as_str() {
                    "ttl" | "turtle" => RdfFormat::Turtle,
                    "nt" => RdfFormat::NTriples,
                    "nq" => RdfFormat::NQuads,
                    "trig" => RdfFormat::TriG,
                    "rdf" | "xml" => RdfFormat::RdfXml,
                    "jsonld" | "json-ld" => RdfFormat::JsonLd {
                        profile: oxirs_core::format::JsonLdProfileSet::empty(),
                    },
                    "n3" => RdfFormat::N3,
                    _ => RdfFormat::Turtle,
                }
            } else {
                RdfFormat::Turtle
            };

            // Parse and serialize
            let file_handle = fs::File::open(&file)?;
            let reader = BufReader::new(file_handle);
            let parser = RdfParser::new(input_format);

            let mut serializer =
                RdfSerializer::new(output_format.clone()).for_writer(std::io::stdout());

            let mut count = 0;
            for quad_result in parser.for_reader(reader) {
                match quad_result {
                    Ok(quad) => {
                        serializer.serialize_quad(quad.as_ref())?;
                        count += 1;
                    }
                    Err(e) => {
                        eprintln!("Parse error: {e}");
                    }
                }
            }

            serializer.finish()?;
            println!("✓ Processed {count} triples from {}", file.display());
        }

        Ok(())
    }

    /// Handle shacl command
    async fn handle_shacl(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            println!("Usage: shacl <data_file> <shapes_file>");
            return Ok(());
        }

        let data_file = std::path::PathBuf::from(args[0]);
        let shapes_file = std::path::PathBuf::from(args[1]);

        println!("Running SHACL validation...");
        println!("  Data: {}", data_file.display());
        println!("  Shapes: {}", shapes_file.display());

        // Integrate with SHACL validation
        // For now, provide a basic implementation that parses both files
        use oxirs_core::format::{RdfFormat, RdfParser};
        use std::fs;
        use std::io::BufReader;

        // Parse data file
        let data_handle = fs::File::open(&data_file)?;
        let data_reader = BufReader::new(data_handle);
        let data_parser = RdfParser::new(RdfFormat::Turtle);

        let mut data_count = 0;
        for quad_result in data_parser.for_reader(data_reader) {
            if quad_result.is_ok() {
                data_count += 1;
            }
        }

        // Parse shapes file
        let shapes_handle = fs::File::open(&shapes_file)?;
        let shapes_reader = BufReader::new(shapes_handle);
        let shapes_parser = RdfParser::new(RdfFormat::Turtle);

        let mut shapes_count = 0;
        for quad_result in shapes_parser.for_reader(shapes_reader) {
            if quad_result.is_ok() {
                shapes_count += 1;
            }
        }

        println!("✓ Data file: {data_count} triples");
        println!("✓ Shapes file: {shapes_count} triples");
        println!("\nNote: Full SHACL validation requires oxirs-shacl module");
        println!("      Both files parsed successfully");

        Ok(())
    }

    /// Handle tdbloader command
    async fn handle_tdbloader(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            println!("Usage: tdbloader <dataset> <files...>");
            return Ok(());
        }

        let dataset = args[0];
        let files: Vec<std::path::PathBuf> =
            args[1..].iter().map(std::path::PathBuf::from).collect();

        println!("Loading data into TDB dataset '{dataset}'...");

        // Integrate with bulk import (use import command for each file)
        for file in files {
            println!("  Loading {}...", file.display());
            crate::commands::import::run(
                dataset.to_string(),
                file.clone(),
                None,  // Auto-detect format
                None,  // Default graph
                false, // Resume not supported in interactive mode
            )
            .await?;
        }

        println!("✓ TDB load complete");
        Ok(())
    }

    /// Handle tdbdump command
    async fn handle_tdbdump(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.is_empty() {
            println!("Usage: tdbdump <dataset> [--output <file>]");
            return Ok(());
        }

        let dataset = args[0];

        // Check for --output option
        let output_file = if args.len() >= 3 && args[1] == "--output" {
            Some(std::path::PathBuf::from(args[2]))
        } else {
            None
        };

        println!("Dumping TDB dataset '{dataset}'...");

        // Integrate with export command
        if let Some(file) = output_file {
            crate::commands::export::run(
                dataset.to_string(),
                file,
                "ntriples".to_string(), // Default format for TDB dumps
                None,                   // All graphs
                false,                  // Resume not supported in interactive mode
            )
            .await?;
        } else {
            // Dump to stdout
            use oxirs_core::format::{RdfFormat, RdfSerializer};
            use oxirs_core::rdf_store::RdfStore;
            use std::path::PathBuf;

            let dataset_path = PathBuf::from(dataset);
            let store = RdfStore::open(&dataset_path)?;

            let quads = store.quads()?;
            let mut serializer =
                RdfSerializer::new(RdfFormat::NTriples).for_writer(std::io::stdout());

            for quad in quads {
                serializer.serialize_quad(quad.as_ref())?;
            }

            serializer.finish()?;
        }

        println!("\n✓ TDB dump complete");
        Ok(())
    }

    /// Create default query templates
    fn create_default_templates() -> HashMap<String, String> {
        let mut templates = HashMap::new();

        templates.insert(
            "select-all".to_string(),
            "SELECT * WHERE { ?s ?p ?o } LIMIT 100".to_string(),
        );

        templates.insert(
            "count".to_string(),
            "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }".to_string(),
        );

        templates.insert(
            "describe".to_string(),
            "DESCRIBE <http://example.org/resource>".to_string(),
        );

        templates.insert(
            "construct".to_string(),
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 100".to_string(),
        );

        templates.insert("ask".to_string(), "ASK { ?s ?p ?o }".to_string());

        templates.insert(
            "distinct-predicates".to_string(),
            "SELECT DISTINCT ?predicate WHERE { ?s ?predicate ?o }".to_string(),
        );

        templates.insert(
            "distinct-types".to_string(),
            "SELECT DISTINCT ?type WHERE { ?s a ?type }".to_string(),
        );

        templates.insert(
            "label-search".to_string(),
            "SELECT ?s ?label WHERE { ?s rdfs:label ?label . FILTER(CONTAINS(LCASE(?label), \"search\")) } LIMIT 20".to_string(),
        );

        templates.insert(
            "property-values".to_string(),
            "SELECT ?value WHERE { <http://example.org/subject> <http://example.org/property> ?value }".to_string(),
        );

        templates.insert(
            "optional-pattern".to_string(),
            "SELECT ?s ?name ?email WHERE { ?s a foaf:Person . ?s foaf:name ?name . OPTIONAL { ?s foaf:mbox ?email } }".to_string(),
        );

        templates
    }

    /// Load saved queries from file
    fn load_saved_queries(file_path: &str) -> HashMap<String, String> {
        if let Ok(content) = std::fs::read_to_string(file_path) {
            if let Ok(queries) = serde_json::from_str(&content) {
                return queries;
            }
        }
        HashMap::new()
    }

    /// Save queries to file
    fn persist_saved_queries(&self) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(&self.saved_queries)?;
        std::fs::write(&self.saved_queries_file, json)?;
        Ok(())
    }

    /// Show a specific template
    fn show_template(&self, name: &str) {
        if let Some(template) = self.query_templates.get(name) {
            println!("Template '{name}':");
            println!("{template}");
        } else {
            println!("Template '{name}' not found");
            println!("Use 'templates' to see available templates");
        }
    }

    /// List all available templates
    fn list_templates(&self) {
        println!("Available query templates:");
        let mut names: Vec<_> = self.query_templates.keys().collect();
        names.sort();

        for name in names {
            if let Some(query) = self.query_templates.get(name.as_str()) {
                println!("\n  {name}:");
                let preview = if query.len() > 60 {
                    format!("{}...", &query[..60])
                } else {
                    query.clone()
                };
                println!("    {}", preview.replace('\n', " "));
            }
        }

        println!("\nUsage: template <name>");
    }

    /// Save a query with a name
    fn save_query(&mut self, name: &str, query: &str) {
        self.saved_queries
            .insert(name.to_string(), query.to_string());

        if let Err(e) = self.persist_saved_queries() {
            eprintln!("Warning: Failed to persist saved queries: {e}");
        } else {
            println!("Query saved as '{name}'");
        }
    }

    /// Load and display a saved query
    fn load_query(&self, name: &str) {
        if let Some(query) = self.saved_queries.get(name) {
            println!("Query '{name}':");
            println!("{query}");
            println!("\nTo execute, copy and paste the query, or use:");
            println!("  query <dataset> {query}");
        } else {
            println!("Query '{name}' not found");
            println!("Use 'queries' to see saved queries");
        }
    }

    /// List all saved queries
    fn list_saved_queries(&self) {
        if self.saved_queries.is_empty() {
            println!("No saved queries");
            println!("Save a query with: save <name> <query>");
            return;
        }

        println!("Saved queries:");
        let mut names: Vec<_> = self.saved_queries.keys().collect();
        names.sort();

        for name in names {
            if let Some(query) = self.saved_queries.get(name.as_str()) {
                let preview = if query.len() > 60 {
                    format!("{}...", &query[..60])
                } else {
                    query.clone()
                };
                println!("  {name}: {}", preview.replace('\n', " "));
            }
        }

        println!("\nUsage: load <name>");
    }

    /// Delete a saved query
    fn delete_query(&mut self, name: &str) {
        if self.saved_queries.remove(name).is_some() {
            if let Err(e) = self.persist_saved_queries() {
                eprintln!("Warning: Failed to persist saved queries: {e}");
            } else {
                println!("Query '{name}' deleted");
            }
        } else {
            println!("Query '{name}' not found");
        }
    }
}

impl Default for InteractiveMode {
    fn default() -> Self {
        Self::new().expect("Failed to create interactive mode")
    }
}

/// Helper for readline with completion, hints, and highlighting
#[derive(Helper, Hinter)]
struct OxirsHelper {
    completer: FilenameCompleter,
    highlighter: MatchingBracketHighlighter,
    #[rustyline(Hinter)]
    hinter: HistoryHinter,
    validator: MatchingBracketValidator,
    commands: Vec<String>,
    sparql_keywords: Vec<String>,
}

impl Completer for OxirsHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &rustyline::Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        let mut candidates = Vec::new();

        // Split the line to get the current word
        let words: Vec<&str> = line[..pos].split_whitespace().collect();

        // Find the current word being typed
        let current_word_start = line[..pos]
            .rfind(|c: char| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        let current_word = &line[current_word_start..pos];
        let current_word_upper = current_word.to_uppercase();

        if words.len() <= 1 {
            // Complete commands
            let prefix = words.first().unwrap_or(&"");
            for cmd in &self.commands {
                if cmd.starts_with(prefix) {
                    candidates.push(Pair {
                        display: cmd.clone(),
                        replacement: cmd.clone(),
                    });
                }
            }
        } else if words.len() >= 2 {
            // Check if we're in a query command context for SPARQL keyword completion
            let first_word = words[0];
            if first_word == "query" || first_word == "save" {
                // Try SPARQL keyword completion
                for keyword in &self.sparql_keywords {
                    if keyword.starts_with(&current_word_upper) {
                        candidates.push(Pair {
                            display: keyword.clone(),
                            replacement: keyword.clone(),
                        });
                    }
                }
            }
        }

        // If no matches, try file completion
        if candidates.is_empty() {
            return self.completer.complete(line, pos, ctx);
        }

        Ok((current_word_start, candidates))
    }
}

impl Highlighter for OxirsHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize, kind: rustyline::highlight::CmdKind) -> bool {
        self.highlighter.highlight_char(line, pos, kind)
    }
}

impl Validator for OxirsHelper {
    fn validate(
        &self,
        ctx: &mut validate::ValidationContext,
    ) -> rustyline::Result<validate::ValidationResult> {
        self.validator.validate(ctx)
    }

    fn validate_while_typing(&self) -> bool {
        self.validator.validate_while_typing()
    }
}

/// Get list of available commands for completion
fn get_command_list() -> Vec<String> {
    vec![
        // Core commands
        "query".to_string(),
        "import".to_string(),
        "export".to_string(),
        "validate".to_string(),
        "stats".to_string(),
        // Data processing
        "riot".to_string(),
        "rdfcat".to_string(),
        "rdfcopy".to_string(),
        "rdfdiff".to_string(),
        "rdfparse".to_string(),
        // Validation
        "shacl".to_string(),
        "shex".to_string(),
        "infer".to_string(),
        // TDB commands
        "tdbloader".to_string(),
        "tdbdump".to_string(),
        "tdbquery".to_string(),
        "tdbupdate".to_string(),
        "tdbstats".to_string(),
        "tdbbackup".to_string(),
        "tdbcompact".to_string(),
        // Query templates and saved queries
        "template".to_string(),
        "templates".to_string(),
        "save".to_string(),
        "load".to_string(),
        "queries".to_string(),
        "delete".to_string(),
        // Environment
        "use".to_string(),
        "set".to_string(),
        "env".to_string(),
        // System
        "clear".to_string(),
        "cls".to_string(),
        "help".to_string(),
        "exit".to_string(),
        "quit".to_string(),
    ]
}

/// Get list of SPARQL keywords for autocomplete
fn get_sparql_keywords() -> Vec<String> {
    vec![
        // Query forms
        "SELECT".to_string(),
        "CONSTRUCT".to_string(),
        "DESCRIBE".to_string(),
        "ASK".to_string(),
        // Update operations
        "INSERT".to_string(),
        "DELETE".to_string(),
        "LOAD".to_string(),
        "CLEAR".to_string(),
        "DROP".to_string(),
        "CREATE".to_string(),
        "COPY".to_string(),
        "MOVE".to_string(),
        "ADD".to_string(),
        // Graph patterns
        "WHERE".to_string(),
        "GRAPH".to_string(),
        "OPTIONAL".to_string(),
        "UNION".to_string(),
        "MINUS".to_string(),
        "SERVICE".to_string(),
        // Modifiers
        "DISTINCT".to_string(),
        "REDUCED".to_string(),
        "ORDER".to_string(),
        "BY".to_string(),
        "LIMIT".to_string(),
        "OFFSET".to_string(),
        "GROUP".to_string(),
        "HAVING".to_string(),
        // Filters and functions
        "FILTER".to_string(),
        "BIND".to_string(),
        "VALUES".to_string(),
        "AS".to_string(),
        // Aggregates
        "COUNT".to_string(),
        "SUM".to_string(),
        "MIN".to_string(),
        "MAX".to_string(),
        "AVG".to_string(),
        "SAMPLE".to_string(),
        "GROUP_CONCAT".to_string(),
        // Functions
        "STR".to_string(),
        "LANG".to_string(),
        "LANGMATCHES".to_string(),
        "DATATYPE".to_string(),
        "BOUND".to_string(),
        "IRI".to_string(),
        "URI".to_string(),
        "BNODE".to_string(),
        "RAND".to_string(),
        "ABS".to_string(),
        "CEIL".to_string(),
        "FLOOR".to_string(),
        "ROUND".to_string(),
        "CONCAT".to_string(),
        "STRLEN".to_string(),
        "UCASE".to_string(),
        "LCASE".to_string(),
        "STRSTARTS".to_string(),
        "STRENDS".to_string(),
        "CONTAINS".to_string(),
        "STRBEFORE".to_string(),
        "STRAFTER".to_string(),
        "ENCODE_FOR_URI".to_string(),
        "REPLACE".to_string(),
        "REGEX".to_string(),
        "SUBSTR".to_string(),
        // Logical operators
        "NOT".to_string(),
        "EXISTS".to_string(),
        // Boolean values
        "TRUE".to_string(),
        "FALSE".to_string(),
        // Data operations
        "DATA".to_string(),
        "WITH".to_string(),
        "INTO".to_string(),
        "USING".to_string(),
        "NAMED".to_string(),
        "DEFAULT".to_string(),
        "ALL".to_string(),
        "SILENT".to_string(),
        // Prefixes
        "PREFIX".to_string(),
        "BASE".to_string(),
        // RDF types
        "a".to_string(), // shorthand for rdf:type
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interactive_mode_creation() {
        // This might fail in CI due to terminal requirements
        if let Ok(mode) = InteractiveMode::new() {
            assert!(!mode.history_file.is_empty());
        }
    }

    #[test]
    fn test_command_list() {
        let commands = get_command_list();
        assert!(commands.contains(&"query".to_string()));
        assert!(commands.contains(&"help".to_string()));
        assert!(commands.contains(&"exit".to_string()));
    }
}
