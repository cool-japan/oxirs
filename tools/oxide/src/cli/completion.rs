//! Advanced auto-completion support for CLI
//!
//! Provides context-aware completion for commands, arguments, and file paths.

use clap::Command;
use clap_complete::{generate, Generator, Shell};
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Generate shell completion scripts
pub fn generate_completion<G: Generator>(
    gen: G,
    app: &mut Command,
    name: &str,
    out: &mut dyn io::Write,
) {
    generate(gen, app, name, out);
}

/// Print completions for specified shell
pub fn print_completions(shell: Shell, app: &mut Command) {
    generate_completion(shell, app, "oxide", &mut io::stdout());
}

/// Completion context for tracking state
pub struct CompletionContext {
    /// Current command being typed
    pub command: Option<String>,
    /// Current subcommand
    pub subcommand: Option<String>,
    /// Arguments provided so far
    pub args: Vec<String>,
    /// Current word being completed
    pub current_word: String,
    /// Position in the command line
    pub position: usize,
}

impl CompletionContext {
    /// Parse a command line into completion context
    pub fn from_line(line: &str, position: usize) -> Self {
        let prefix = &line[..position];
        let words: Vec<String> = shlex::split(prefix).unwrap_or_default();

        let (command, subcommand, args) = if words.len() > 1 {
            let cmd = words[0].clone();
            if words.len() > 2 {
                (Some(cmd), Some(words[1].clone()), words[2..].to_vec())
            } else {
                (Some(cmd), None, words[1..].to_vec())
            }
        } else {
            (words.first().cloned(), None, vec![])
        };

        let current_word = if line[..position].ends_with(' ') {
            String::new()
        } else {
            words.last().cloned().unwrap_or_default()
        };

        Self {
            command,
            subcommand,
            args,
            current_word,
            position,
        }
    }
}

/// Completion provider trait
pub trait CompletionProvider {
    /// Get completions for the current context
    fn get_completions(&self, context: &CompletionContext) -> Vec<CompletionItem>;
}

/// A single completion item
#[derive(Debug, Clone)]
pub struct CompletionItem {
    /// The text to insert
    pub replacement: String,
    /// Display text (may include description)
    pub display: String,
    /// Optional description
    pub description: Option<String>,
    /// Completion type (for styling)
    pub completion_type: CompletionType,
}

/// Type of completion for styling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompletionType {
    Command,
    Subcommand,
    Argument,
    Option,
    File,
    Directory,
    Value,
    Variable,
}

/// Command completion provider
pub struct CommandCompletionProvider {
    commands: HashMap<String, CommandInfo>,
}

/// Information about a command for completion
#[allow(dead_code)]
struct CommandInfo {
    description: String,
    subcommands: Vec<String>,
    options: Vec<OptionInfo>,
    positional_args: Vec<ArgInfo>,
}

/// Information about an option
#[allow(dead_code)]
struct OptionInfo {
    short: Option<char>,
    long: String,
    description: String,
    takes_value: bool,
    possible_values: Vec<String>,
}

/// Information about a positional argument
#[allow(dead_code)]
struct ArgInfo {
    name: String,
    description: String,
    completion_hint: CompletionHint,
}

/// Hints for argument completion
#[derive(Clone)]
#[allow(dead_code)]
enum CompletionHint {
    File { extensions: Vec<String> },
    Directory,
    Dataset,
    Format { values: Vec<String> },
    Url,
    Custom { values: Vec<String> },
}

impl CommandCompletionProvider {
    /// Create a new command completion provider
    pub fn new() -> Self {
        let mut provider = Self {
            commands: HashMap::new(),
        };
        provider.initialize_commands();
        provider
    }

    /// Initialize command information
    fn initialize_commands(&mut self) {
        // Query command
        self.commands.insert(
            "query".to_string(),
            CommandInfo {
                description: "Execute SPARQL query".to_string(),
                subcommands: vec![],
                options: vec![
                    OptionInfo {
                        short: Some('f'),
                        long: "file".to_string(),
                        description: "Read query from file".to_string(),
                        takes_value: false,
                        possible_values: vec![],
                    },
                    OptionInfo {
                        short: Some('o'),
                        long: "output".to_string(),
                        description: "Output format".to_string(),
                        takes_value: true,
                        possible_values: vec![
                            "json".to_string(),
                            "csv".to_string(),
                            "tsv".to_string(),
                            "table".to_string(),
                        ],
                    },
                ],
                positional_args: vec![
                    ArgInfo {
                        name: "dataset".to_string(),
                        description: "Target dataset".to_string(),
                        completion_hint: CompletionHint::Dataset,
                    },
                    ArgInfo {
                        name: "query".to_string(),
                        description: "SPARQL query".to_string(),
                        completion_hint: CompletionHint::Custom {
                            values: vec!["SELECT * WHERE { ?s ?p ?o } LIMIT 10".to_string()],
                        },
                    },
                ],
            },
        );

        // Import command
        self.commands.insert(
            "import".to_string(),
            CommandInfo {
                description: "Import RDF data".to_string(),
                subcommands: vec![],
                options: vec![
                    OptionInfo {
                        short: Some('f'),
                        long: "format".to_string(),
                        description: "Input format".to_string(),
                        takes_value: true,
                        possible_values: vec![
                            "turtle".to_string(),
                            "ntriples".to_string(),
                            "rdfxml".to_string(),
                            "jsonld".to_string(),
                        ],
                    },
                    OptionInfo {
                        short: Some('g'),
                        long: "graph".to_string(),
                        description: "Named graph URI".to_string(),
                        takes_value: true,
                        possible_values: vec![],
                    },
                ],
                positional_args: vec![
                    ArgInfo {
                        name: "dataset".to_string(),
                        description: "Target dataset".to_string(),
                        completion_hint: CompletionHint::Dataset,
                    },
                    ArgInfo {
                        name: "file".to_string(),
                        description: "Input file".to_string(),
                        completion_hint: CompletionHint::File {
                            extensions: vec![
                                "ttl".to_string(),
                                "nt".to_string(),
                                "rdf".to_string(),
                                "xml".to_string(),
                                "jsonld".to_string(),
                            ],
                        },
                    },
                ],
            },
        );

        // Riot command
        self.commands.insert(
            "riot".to_string(),
            CommandInfo {
                description: "RDF I/O tool".to_string(),
                subcommands: vec![],
                options: vec![
                    OptionInfo {
                        short: None,
                        long: "output".to_string(),
                        description: "Output format".to_string(),
                        takes_value: true,
                        possible_values: vec![
                            "turtle".to_string(),
                            "ntriples".to_string(),
                            "rdfxml".to_string(),
                            "jsonld".to_string(),
                            "trig".to_string(),
                            "nquads".to_string(),
                        ],
                    },
                    OptionInfo {
                        short: None,
                        long: "out".to_string(),
                        description: "Output file".to_string(),
                        takes_value: true,
                        possible_values: vec![],
                    },
                    OptionInfo {
                        short: None,
                        long: "validate".to_string(),
                        description: "Validate only".to_string(),
                        takes_value: false,
                        possible_values: vec![],
                    },
                ],
                positional_args: vec![ArgInfo {
                    name: "files".to_string(),
                    description: "Input files".to_string(),
                    completion_hint: CompletionHint::File {
                        extensions: vec![
                            "ttl".to_string(),
                            "nt".to_string(),
                            "rdf".to_string(),
                            "xml".to_string(),
                            "jsonld".to_string(),
                        ],
                    },
                }],
            },
        );
    }
}

impl CompletionProvider for CommandCompletionProvider {
    fn get_completions(&self, context: &CompletionContext) -> Vec<CompletionItem> {
        let mut completions = Vec::new();

        // If no command yet, complete commands
        if context.command.is_none()
            || (context.args.is_empty() && !context.current_word.is_empty())
        {
            for (cmd, info) in &self.commands {
                if cmd.starts_with(&context.current_word) {
                    completions.push(CompletionItem {
                        replacement: cmd.clone(),
                        display: format!("{:<20} {}", cmd, info.description),
                        description: Some(info.description.clone()),
                        completion_type: CompletionType::Command,
                    });
                }
            }
            return completions;
        }

        // Complete based on command context
        if let Some(ref cmd) = context.command {
            if let Some(cmd_info) = self.commands.get(cmd) {
                // Complete options
                if context.current_word.starts_with('-') {
                    for opt in &cmd_info.options {
                        if let Some(short) = opt.short {
                            let short_opt = format!("-{}", short);
                            if short_opt.starts_with(&context.current_word) {
                                completions.push(CompletionItem {
                                    replacement: short_opt.clone(),
                                    display: format!("{:<20} {}", short_opt, opt.description),
                                    description: Some(opt.description.clone()),
                                    completion_type: CompletionType::Option,
                                });
                            }
                        }

                        let long_opt = format!("--{}", opt.long);
                        if long_opt.starts_with(&context.current_word) {
                            completions.push(CompletionItem {
                                replacement: long_opt.clone(),
                                display: format!("{:<20} {}", long_opt, opt.description),
                                description: Some(opt.description.clone()),
                                completion_type: CompletionType::Option,
                            });
                        }
                    }
                } else {
                    // Complete positional arguments
                    let arg_index = context
                        .args
                        .iter()
                        .filter(|arg| !arg.starts_with('-'))
                        .count();

                    if let Some(arg_info) = cmd_info.positional_args.get(arg_index) {
                        completions.extend(self.complete_argument(arg_info, &context.current_word));
                    }
                }
            }
        }

        completions
    }
}

impl CommandCompletionProvider {
    /// Complete a specific argument
    fn complete_argument(&self, arg_info: &ArgInfo, current: &str) -> Vec<CompletionItem> {
        match &arg_info.completion_hint {
            CompletionHint::File { extensions } => complete_files(current, Some(extensions)),
            CompletionHint::Directory => complete_directories(current),
            CompletionHint::Dataset => complete_datasets(current),
            CompletionHint::Format { values } => values
                .iter()
                .filter(|v| v.starts_with(current))
                .map(|v| CompletionItem {
                    replacement: v.clone(),
                    display: v.clone(),
                    description: None,
                    completion_type: CompletionType::Value,
                })
                .collect(),
            CompletionHint::Url => {
                if current.is_empty() {
                    vec![CompletionItem {
                        replacement: "http://".to_string(),
                        display: "http://".to_string(),
                        description: Some("HTTP URL".to_string()),
                        completion_type: CompletionType::Value,
                    }]
                } else {
                    vec![]
                }
            }
            CompletionHint::Custom { values } => values
                .iter()
                .filter(|v| current.is_empty() || v.contains(current))
                .map(|v| CompletionItem {
                    replacement: v.clone(),
                    display: v.clone(),
                    description: None,
                    completion_type: CompletionType::Value,
                })
                .collect(),
        }
    }
}

/// Complete file paths
fn complete_files(prefix: &str, extensions: Option<&Vec<String>>) -> Vec<CompletionItem> {
    let (dir, file_prefix) = if prefix.contains('/') {
        let path = Path::new(prefix);
        let parent = path.parent().unwrap_or(Path::new("."));
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        (parent, file_name)
    } else {
        (Path::new("."), prefix)
    };

    let mut completions = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if !name_str.starts_with(file_prefix) {
                continue;
            }

            if path.is_dir() {
                let display = format!("{}/", name_str);
                completions.push(CompletionItem {
                    replacement: display.clone(),
                    display,
                    description: Some("Directory".to_string()),
                    completion_type: CompletionType::Directory,
                });
            } else if let Some(exts) = extensions {
                if let Some(ext) = path.extension() {
                    if exts.iter().any(|e| e == &ext.to_string_lossy().to_string()) {
                        completions.push(CompletionItem {
                            replacement: name_str.to_string(),
                            display: name_str.to_string(),
                            description: Some(format!("{} file", ext.to_string_lossy())),
                            completion_type: CompletionType::File,
                        });
                    }
                }
            } else {
                completions.push(CompletionItem {
                    replacement: name_str.to_string(),
                    display: name_str.to_string(),
                    description: Some("File".to_string()),
                    completion_type: CompletionType::File,
                });
            }
        }
    }

    completions
}

/// Complete directory paths
fn complete_directories(prefix: &str) -> Vec<CompletionItem> {
    let (dir, dir_prefix) = if prefix.contains('/') {
        let path = Path::new(prefix);
        let parent = path.parent().unwrap_or(Path::new("."));
        let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        (parent, dir_name)
    } else {
        (Path::new("."), prefix)
    };

    let mut completions = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.filter_map(Result::ok) {
            if entry.path().is_dir() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                if name_str.starts_with(dir_prefix) {
                    let display = format!("{}/", name_str);
                    completions.push(CompletionItem {
                        replacement: display.clone(),
                        display,
                        description: Some("Directory".to_string()),
                        completion_type: CompletionType::Directory,
                    });
                }
            }
        }
    }

    completions
}

/// Complete dataset names
fn complete_datasets(prefix: &str) -> Vec<CompletionItem> {
    // TODO: This should look for actual datasets in the workspace
    let datasets = vec!["mykg", "test-db", "production"];

    datasets
        .into_iter()
        .filter(|d| d.starts_with(prefix))
        .map(|d| CompletionItem {
            replacement: d.to_string(),
            display: d.to_string(),
            description: Some("Dataset".to_string()),
            completion_type: CompletionType::Value,
        })
        .collect()
}

/// Shell-specific completion generator
pub mod shell {
    use super::*;
    use clap::CommandFactory;
    use clap_complete::{generate, Shell};
    use std::io;

    /// Generate completion script for a specific shell
    pub fn generate_completion_script(shell: Shell, app: &mut Command) {
        generate(shell, app, "oxide", &mut io::stdout());
    }

    /// Install completion for the current shell
    pub fn install_completion(shell: Shell) -> Result<(), Box<dyn std::error::Error>> {
        match shell {
            Shell::Bash => install_bash_completion(),
            Shell::Zsh => install_zsh_completion(),
            Shell::Fish => install_fish_completion(),
            _ => Err("Unsupported shell".into()),
        }
    }

    fn install_bash_completion() -> Result<(), Box<dyn std::error::Error>> {
        let completion_dir = dirs::home_dir()
            .ok_or("Could not find home directory")?
            .join(".local/share/bash-completion/completions");

        std::fs::create_dir_all(&completion_dir)?;

        let completion_file = completion_dir.join("oxide");
        let mut app = crate::Cli::command();
        let mut file = std::fs::File::create(completion_file)?;
        generate(Shell::Bash, &mut app, "oxide", &mut file);

        println!("Bash completion installed. Restart your shell or run:");
        println!("  source ~/.local/share/bash-completion/completions/oxide");

        Ok(())
    }

    fn install_zsh_completion() -> Result<(), Box<dyn std::error::Error>> {
        let completion_dir = dirs::home_dir()
            .ok_or("Could not find home directory")?
            .join(".zsh/completions");

        std::fs::create_dir_all(&completion_dir)?;

        let completion_file = completion_dir.join("_oxide");
        let mut app = crate::Cli::command();
        let mut file = std::fs::File::create(completion_file)?;
        generate(Shell::Zsh, &mut app, "oxide", &mut file);

        println!("Zsh completion installed. Add this to your ~/.zshrc:");
        println!("  fpath=(~/.zsh/completions $fpath)");
        println!("  autoload -U compinit && compinit");

        Ok(())
    }

    fn install_fish_completion() -> Result<(), Box<dyn std::error::Error>> {
        let completion_dir = dirs::config_dir()
            .ok_or("Could not find config directory")?
            .join("fish/completions");

        std::fs::create_dir_all(&completion_dir)?;

        let completion_file = completion_dir.join("oxide.fish");
        let mut app = crate::Cli::command();
        let mut file = std::fs::File::create(completion_file)?;
        generate(Shell::Fish, &mut app, "oxide", &mut file);

        println!("Fish completion installed. It should work immediately.");

        Ok(())
    }
}

impl Default for CommandCompletionProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_context_parsing() {
        let ctx = CompletionContext::from_line("oxide query mydb SEL", 20);
        assert_eq!(ctx.command, Some("oxide".to_string()));
        assert_eq!(ctx.subcommand, Some("query".to_string()));
        assert_eq!(ctx.current_word, "SEL");
    }

    #[test]
    fn test_command_completion() {
        let provider = CommandCompletionProvider::new();
        let ctx = CompletionContext {
            command: None,
            subcommand: None,
            args: vec![],
            current_word: "que".to_string(),
            position: 3,
        };

        let completions = provider.get_completions(&ctx);
        assert!(completions.iter().any(|c| c.replacement == "query"));
    }
}
