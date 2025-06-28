//! Enhanced CLI Infrastructure
//!
//! Provides advanced argument validation, interactive mode support,
//! progress tracking, and improved user experience features.

use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, Write};
use std::time::Duration;

pub mod completion;
pub mod error;
pub mod help;
pub mod interactive;
pub mod logging;
pub mod output;
pub mod progress;
pub mod validation;

pub use completion::{CommandCompletionProvider, CompletionContext, CompletionProvider};
pub use error::{CliError, CliResult};
pub use help::{HelpCategory, HelpProvider};
pub use interactive::InteractiveMode;
pub use logging::{
    init_logging, CommandLogger, DataLogger, LogConfig, LogFormat, PerfLogger, QueryLogger,
};
pub use output::{ColorScheme, OutputFormatter};
pub use progress::{ProgressTracker, ProgressType};
pub use validation::ArgumentValidator;

/// CLI Context for managing global state
pub struct CliContext {
    pub verbose: bool,
    pub quiet: bool,
    pub no_color: bool,
    pub interactive: bool,
    pub profile: Option<String>,
    pub output_formatter: OutputFormatter,
    pub progress_tracker: Option<ProgressTracker>,
}

impl CliContext {
    pub fn new() -> Self {
        let no_color = std::env::var("NO_COLOR").is_ok() || !atty::is(atty::Stream::Stdout);

        Self {
            verbose: false,
            quiet: false,
            no_color,
            interactive: false,
            profile: None,
            output_formatter: OutputFormatter::new(no_color),
            progress_tracker: None,
        }
    }

    /// Initialize from CLI arguments
    pub fn from_cli(verbose: bool, quiet: bool, no_color: bool) -> Self {
        let mut ctx = Self::new();
        ctx.verbose = verbose;
        ctx.quiet = quiet;
        ctx.no_color = no_color || ctx.no_color;
        ctx.output_formatter = OutputFormatter::new(ctx.no_color);
        ctx
    }

    /// Check if we should show output
    pub fn should_show_output(&self) -> bool {
        !self.quiet
    }

    /// Check if we should show verbose output
    pub fn should_show_verbose(&self) -> bool {
        self.verbose && !self.quiet
    }

    /// Start a progress operation
    pub fn start_progress(&mut self, total: Option<u64>, message: &str) -> Option<ProgressBar> {
        if self.quiet || !self.should_show_output() {
            return None;
        }

        let pb = match total {
            Some(len) => {
                let pb = ProgressBar::new(len);
                pb.set_style(
                    ProgressStyle::default_bar()
                        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                        .unwrap()
                        .progress_chars("=>-"),
                );
                pb
            }
            None => {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} {msg}")
                        .unwrap(),
                );
                pb.enable_steady_tick(Duration::from_millis(100));
                pb
            }
        };

        pb.set_message(message.to_string());
        Some(pb)
    }

    /// Print an info message
    pub fn info(&self, message: &str) {
        if self.should_show_output() {
            self.output_formatter.info(message);
        }
    }

    /// Print a success message
    pub fn success(&self, message: &str) {
        if self.should_show_output() {
            self.output_formatter.success(message);
        }
    }

    /// Print a warning message
    pub fn warn(&self, message: &str) {
        if self.should_show_output() {
            self.output_formatter.warn(message);
        }
    }

    /// Print an error message
    pub fn error(&self, message: &str) {
        self.output_formatter.error(message);
    }

    /// Print verbose output
    pub fn verbose(&self, message: &str) {
        if self.should_show_verbose() {
            self.output_formatter.verbose(message);
        }
    }

    /// Prompt user for confirmation (returns true if yes)
    pub fn confirm(&self, prompt: &str) -> io::Result<bool> {
        if !self.interactive {
            return Ok(true); // Non-interactive mode assumes yes
        }

        print!("{} [y/N]: ", prompt);
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        Ok(input.trim().to_lowercase() == "y" || input.trim().to_lowercase() == "yes")
    }

    /// Prompt user for input
    pub fn prompt(&self, prompt: &str) -> io::Result<String> {
        print!("{}: ", prompt);
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        Ok(input.trim().to_string())
    }
}

impl Default for CliContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Command suggestions
pub mod suggestions {
    use strsim::levenshtein;

    /// Find similar commands for suggestions
    pub fn find_similar_commands(input: &str, commands: &[&str], threshold: usize) -> Vec<String> {
        let mut suggestions: Vec<(String, usize)> = commands
            .iter()
            .map(|&cmd| (cmd.to_string(), levenshtein(input, cmd)))
            .filter(|(_, distance)| *distance <= threshold)
            .collect();

        suggestions.sort_by_key(|(_, distance)| *distance);
        suggestions.into_iter().map(|(cmd, _)| cmd).collect()
    }

    /// Suggest commands based on user input
    pub fn suggest_command(input: &str, commands: &[&str]) -> Option<String> {
        let similar = find_similar_commands(input, commands, 3);
        if !similar.is_empty() {
            Some(format!("Did you mean: {}?", similar.join(", ")))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_context_creation() {
        let ctx = CliContext::new();
        assert!(!ctx.verbose);
        assert!(!ctx.quiet);
        assert!(!ctx.interactive);
    }

    #[test]
    fn test_output_control() {
        let mut ctx = CliContext::new();
        assert!(ctx.should_show_output());

        ctx.quiet = true;
        assert!(!ctx.should_show_output());
        assert!(!ctx.should_show_verbose());

        ctx.quiet = false;
        ctx.verbose = true;
        assert!(ctx.should_show_verbose());
    }

    #[test]
    fn test_command_suggestions() {
        use suggestions::*;

        let commands = vec!["query", "update", "import", "export"];
        let similar = find_similar_commands("qeury", &commands, 3);
        assert_eq!(similar, vec!["query"]);

        let suggestion = suggest_command("improt", &commands);
        assert!(suggestion.is_some());
        assert!(suggestion.unwrap().contains("import"));
    }
}
