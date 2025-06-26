//! Interactive mode support for CLI
//!
//! Provides REPL-like interactive command execution with history and completion.

use rustyline::error::ReadlineError;
use rustyline::{CompletionType, Config, EditMode, Editor};
use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::highlight::{Highlighter, MatchingBracketHighlighter};
use rustyline::hint::HistoryHinter;
use rustyline::validate::{self, MatchingBracketValidator, Validator};
use rustyline_derive::{Completer, Helper, Highlighter, Hinter, Validator};
use std::borrow::Cow;

/// Interactive mode handler
pub struct InteractiveMode {
    editor: Editor<OxideHelper, rustyline::history::DefaultHistory>,
    history_file: String,
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
        })
    }

    /// Run the interactive REPL
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Welcome to Oxide Interactive Mode!");
        println!("Type 'help' for available commands, 'exit' to quit.\n");

        loop {
            let readline = self.editor.readline("oxide> ");
            match readline {
                Ok(line) => {
                    if line.trim().is_empty() {
                        continue;
                    }

                    self.editor.add_history_entry(line.as_str());

                    let trimmed = line.trim();
                    match trimmed {
                        "exit" | "quit" => break,
                        "help" => self.show_help(),
                        "clear" => self.clear_screen(),
                        _ => {
                            if let Err(e) = self.execute_command(trimmed).await {
                                eprintln!("Error: {}", e);
                            }
                        }
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
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

    /// Execute a command in interactive mode
    async fn execute_command(&self, command: &str) -> Result<(), Box<dyn std::error::Error>> {
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        let cmd = parts[0];
        let args = &parts[1..];

        match cmd {
            "query" => self.handle_query(args).await,
            "import" => self.handle_import(args).await,
            "export" => self.handle_export(args).await,
            "validate" => self.handle_validate(args).await,
            "stats" => self.handle_stats(args).await,
            _ => {
                println!("Unknown command: {}. Type 'help' for available commands.", cmd);
                Ok(())
            }
        }
    }

    /// Show help message
    fn show_help(&self) {
        println!("Available commands:");
        println!("  query <dataset> <sparql>  - Execute a SPARQL query");
        println!("  import <dataset> <file>   - Import RDF data");
        println!("  export <dataset> <file>   - Export RDF data");
        println!("  validate <file>           - Validate RDF syntax");
        println!("  stats <dataset>           - Show dataset statistics");
        println!("  clear                     - Clear the screen");
        println!("  help                      - Show this help message");
        println!("  exit/quit                 - Exit interactive mode");
        println!("\nTips:");
        println!("  - Use Tab for command and file completion");
        println!("  - Use Up/Down arrows for command history");
        println!("  - Use Ctrl+R for reverse history search");
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
            let prefix = words.get(0).unwrap_or(&"");
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
        "query".to_string(),
        "import".to_string(),
        "export".to_string(),
        "validate".to_string(),
        "stats".to_string(),
        "clear".to_string(),
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