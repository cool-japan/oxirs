//! Interactive mode support for CLI
//!
//! Provides REPL-like interactive command execution with history and completion.

use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::hint::HistoryHinter;
use rustyline::validate::{self, MatchingBracketValidator, Validator};
use rustyline::{CompletionType, Config, EditMode, Editor};
use rustyline_derive::{Helper, Hinter};
use std::borrow::Cow;
use std::collections::HashMap;

/// Interactive mode handler
pub struct InteractiveMode {
    editor: Editor<OxideHelper, rustyline::history::DefaultHistory>,
    history_file: String,
    current_dataset: Option<String>,
    environment: HashMap<String, String>,
    multi_line_mode: bool,
    multi_line_buffer: Vec<String>,
}

impl InteractiveMode {
    /// Create a new interactive mode instance
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = Config::builder()
            .history_ignore_space(true)
            .completion_type(CompletionType::List)
            .edit_mode(EditMode::Emacs)
            .build();

        let helper = OxideHelper {
            completer: FilenameCompleter::new(),
            highlighter: MatchingBracketHighlighter::new(),
            hinter: HistoryHinter {},
            validator: MatchingBracketValidator::new(),
            commands: get_command_list(),
        };

        let mut editor = Editor::with_config(config)?;
        editor.set_helper(Some(helper));

        let history_file = dirs::config_dir()
            .map(|p| p.join("oxide").join("history.txt"))
            .unwrap_or_else(|| ".oxide_history".into())
            .to_string_lossy()
            .to_string();

        // Load history if it exists
        if std::path::Path::new(&history_file).exists() {
            editor.load_history(&history_file).ok();
        }

        Ok(Self {
            editor,
            history_file,
            current_dataset: None,
            environment: HashMap::new(),
            multi_line_mode: false,
            multi_line_buffer: Vec::new(),
        })
    }

    /// Run the interactive REPL
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Welcome to Oxide Interactive Mode!");
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
                                eprintln!("Error: {}", e);
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
                        eprintln!("Error: {}", e);
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
                    eprintln!("Error: {:?}", err);
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
            format!("oxide:{}> ", dataset)
        } else {
            "oxide> ".to_string()
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
                    println!("Unknown command: {}. {}", cmd, suggestion);
                } else {
                    println!(
                        "Unknown command: {}. Type 'help' for available commands.",
                        cmd
                    );
                }
                Ok(())
            }
        }
    }

    /// Expand environment variables in command
    fn expand_variables(&self, command: &str) -> String {
        let mut result = command.to_string();
        for (key, value) in &self.environment {
            result = result.replace(&format!("${}", key), value);
            result = result.replace(&format!("${{{}}}", key), value);
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

        if let Some(ref dataset) = self.current_dataset {
            println!("\nCurrent dataset: {}", dataset);
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

        println!("Executing query on dataset '{}'...", dataset);
        println!("Query: {}", query);

        // TODO: Integrate with actual query execution
        println!("Query execution not yet implemented in interactive mode");

        Ok(())
    }

    /// Handle import command
    async fn handle_import(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            println!("Usage: import <dataset> <file>");
            return Ok(());
        }

        let dataset = args[0];
        let file = args[1];

        println!("Importing {} into dataset '{}'...", file, dataset);

        // TODO: Integrate with actual import
        println!("Import not yet implemented in interactive mode");

        Ok(())
    }

    /// Handle export command
    async fn handle_export(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            println!("Usage: export <dataset> <file>");
            return Ok(());
        }

        let dataset = args[0];
        let file = args[1];

        println!("Exporting dataset '{}' to {}...", dataset, file);

        // TODO: Integrate with actual export
        println!("Export not yet implemented in interactive mode");

        Ok(())
    }

    /// Handle validate command
    async fn handle_validate(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.is_empty() {
            println!("Usage: validate <file>");
            return Ok(());
        }

        let file = args[0];
        println!("Validating {}...", file);

        // TODO: Integrate with actual validation
        println!("Validation not yet implemented in interactive mode");

        Ok(())
    }

    /// Handle stats command
    async fn handle_stats(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.is_empty() {
            println!("Usage: stats <dataset>");
            return Ok(());
        }

        let dataset = args[0];
        println!("Getting statistics for dataset '{}'...", dataset);

        // TODO: Integrate with actual stats
        println!("Stats not yet implemented in interactive mode");

        Ok(())
    }

    /// Use a dataset
    fn use_dataset(&mut self, dataset: &str) {
        self.current_dataset = Some(dataset.to_string());
        println!("Now using dataset: {}", dataset);
    }

    /// Set an environment variable
    fn set_variable(&mut self, key: &str, value: &str) {
        self.environment.insert(key.to_string(), value.to_string());
        println!("Set {} = {}", key, value);
    }

    /// Show environment variables
    fn show_environment(&self) {
        if self.environment.is_empty() {
            println!("No variables set");
        } else {
            println!("Environment variables:");
            for (key, value) in &self.environment {
                println!("  {} = {}", key, value);
            }
        }

        if let Some(ref dataset) = self.current_dataset {
            println!("\nCurrent dataset: {}", dataset);
        }
    }

    /// Handle riot command
    async fn handle_riot(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.is_empty() {
            println!("Usage: riot <input_files...> [--output <format>]");
            return Ok(());
        }

        println!("Running riot with args: {}", args.join(" "));
        // TODO: Integrate with actual riot command
        println!("Riot command not yet fully integrated in interactive mode");
        Ok(())
    }

    /// Handle shacl command  
    async fn handle_shacl(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            println!("Usage: shacl <data_file> <shapes_file>");
            return Ok(());
        }

        println!("Running SHACL validation...");
        // TODO: Integrate with actual shacl command
        println!("SHACL command not yet fully integrated in interactive mode");
        Ok(())
    }

    /// Handle tdbloader command
    async fn handle_tdbloader(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.len() < 2 {
            println!("Usage: tdbloader <dataset> <files...>");
            return Ok(());
        }

        println!("Loading data into TDB...");
        // TODO: Integrate with actual tdbloader command
        println!("TDB loader not yet fully integrated in interactive mode");
        Ok(())
    }

    /// Handle tdbdump command
    async fn handle_tdbdump(&self, args: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
        if args.is_empty() {
            println!("Usage: tdbdump <dataset> [--output <file>]");
            return Ok(());
        }

        println!("Dumping TDB dataset...");
        // TODO: Integrate with actual tdbdump command
        println!("TDB dump not yet fully integrated in interactive mode");
        Ok(())
    }
}

impl Default for InteractiveMode {
    fn default() -> Self {
        Self::new().expect("Failed to create interactive mode")
    }
}

/// Helper for readline with completion, hints, and highlighting
#[derive(Helper, Hinter)]
struct OxideHelper {
    completer: FilenameCompleter,
    highlighter: MatchingBracketHighlighter,
    #[rustyline(Hinter)]
    hinter: HistoryHinter,
    validator: MatchingBracketValidator,
    commands: Vec<String>,
}

impl Completer for OxideHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &rustyline::Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        // First try command completion
        let mut candidates = Vec::new();

        // Split the line to get the current word
        let words: Vec<&str> = line[..pos].split_whitespace().collect();

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
        }

        // If no command matches, try file completion
        if candidates.is_empty() {
            return self.completer.complete(line, pos, ctx);
        }

        Ok((0, candidates))
    }
}

impl Highlighter for OxideHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_char(&self, line: &str, pos: usize, forced: bool) -> bool {
        self.highlighter.highlight_char(line, pos, forced)
    }
}

impl Validator for OxideHelper {
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
