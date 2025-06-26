//! Enhanced help system with better documentation
//!
//! Provides context-aware help, examples, and tutorials for CLI commands.

use std::collections::HashMap;
use colored::*;
use textwrap::{wrap, Options};
use unicode_width::UnicodeWidthStr;

/// Help topic categories
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HelpCategory {
    GettingStarted,
    DataManagement,
    Querying,
    Validation,
    Storage,
    Configuration,
    Troubleshooting,
    Examples,
}

/// Enhanced help provider
pub struct HelpProvider {
    commands: HashMap<String, CommandHelp>,
    topics: HashMap<String, TopicHelp>,
    examples: HashMap<String, Vec<Example>>,
    terminal_width: usize,
}

/// Detailed help for a command
#[derive(Clone)]
pub struct CommandHelp {
    pub name: String,
    pub description: String,
    pub long_description: Option<String>,
    pub usage: Vec<String>,
    pub arguments: Vec<ArgumentHelp>,
    pub options: Vec<OptionHelp>,
    pub examples: Vec<Example>,
    pub see_also: Vec<String>,
    pub category: HelpCategory,
}

/// Help for command arguments
#[derive(Clone)]
pub struct ArgumentHelp {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub value_type: String,
    pub default: Option<String>,
}

/// Help for command options
#[derive(Clone)]
pub struct OptionHelp {
    pub short: Option<char>,
    pub long: String,
    pub description: String,
    pub value_name: Option<String>,
    pub possible_values: Vec<String>,
    pub default: Option<String>,
}

/// Example usage
#[derive(Clone)]
pub struct Example {
    pub description: String,
    pub command: String,
    pub output: Option<String>,
    pub explanation: Option<String>,
}

/// Help topic
pub struct TopicHelp {
    pub title: String,
    pub content: String,
    pub examples: Vec<Example>,
    pub see_also: Vec<String>,
}

impl HelpProvider {
    /// Create a new help provider
    pub fn new() -> Self {
        let terminal_width = terminal_size::terminal_size()
            .map(|(w, _)| w.0 as usize)
            .unwrap_or(80);
            
        let mut provider = Self {
            commands: HashMap::new(),
            topics: HashMap::new(),
            examples: HashMap::new(),
            terminal_width,
        };
        
        provider.initialize_help();
        provider
    }
    
    /// Initialize help content
    fn initialize_help(&mut self) {
        self.init_command_help();
        self.init_topic_help();
        self.init_examples();
    }
    
    /// Initialize command help
    fn init_command_help(&mut self) {
        // Query command
        self.commands.insert("query".to_string(), CommandHelp {
            name: "query".to_string(),
            description: "Execute SPARQL queries against RDF datasets".to_string(),
            long_description: Some(
                "The query command allows you to execute SPARQL queries against local or remote RDF datasets. \
                It supports all SPARQL 1.1 query forms (SELECT, CONSTRUCT, DESCRIBE, ASK) and can output \
                results in various formats.".to_string()
            ),
            usage: vec![
                "oxide query <DATASET> <QUERY>".to_string(),
                "oxide query <DATASET> -f <QUERY_FILE>".to_string(),
                "oxide query <DATASET> <QUERY> -o json > results.json".to_string(),
            ],
            arguments: vec![
                ArgumentHelp {
                    name: "DATASET".to_string(),
                    description: "The target dataset to query (path or name)".to_string(),
                    required: true,
                    value_type: "STRING".to_string(),
                    default: None,
                },
                ArgumentHelp {
                    name: "QUERY".to_string(),
                    description: "The SPARQL query to execute".to_string(),
                    required: true,
                    value_type: "STRING".to_string(),
                    default: None,
                },
            ],
            options: vec![
                OptionHelp {
                    short: Some('f'),
                    long: "file".to_string(),
                    description: "Read query from a file instead of command line".to_string(),
                    value_name: Some("FILE".to_string()),
                    possible_values: vec![],
                    default: None,
                },
                OptionHelp {
                    short: Some('o'),
                    long: "output".to_string(),
                    description: "Output format for results".to_string(),
                    value_name: Some("FORMAT".to_string()),
                    possible_values: vec!["table".to_string(), "json".to_string(), "csv".to_string(), "tsv".to_string(), "xml".to_string()],
                    default: Some("table".to_string()),
                },
            ],
            examples: vec![
                Example {
                    description: "Simple SELECT query".to_string(),
                    command: r#"oxide query mydb "SELECT * WHERE { ?s ?p ?o } LIMIT 10""#.to_string(),
                    output: Some("Shows first 10 triples in table format".to_string()),
                    explanation: Some("Queries all triples and limits results to 10".to_string()),
                },
                Example {
                    description: "Query from file with JSON output".to_string(),
                    command: "oxide query mydb -f complex_query.rq -o json".to_string(),
                    output: None,
                    explanation: Some("Executes query from file and outputs JSON".to_string()),
                },
                Example {
                    description: "Count all triples".to_string(),
                    command: r#"oxide query mydb "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }""#.to_string(),
                    output: None,
                    explanation: Some("Returns the total number of triples in the dataset".to_string()),
                },
            ],
            see_also: vec!["update".to_string(), "arq".to_string(), "rsparql".to_string()],
            category: HelpCategory::Querying,
        });
        
        // Import command
        self.commands.insert("import".to_string(), CommandHelp {
            name: "import".to_string(),
            description: "Import RDF data into a dataset".to_string(),
            long_description: Some(
                "Import RDF data from files in various formats into a dataset. Supports automatic format \
                detection based on file extensions and content type. Can import into named graphs and \
                handle large files efficiently.".to_string()
            ),
            usage: vec![
                "oxide import <DATASET> <FILE>".to_string(),
                "oxide import <DATASET> <FILE> -f turtle".to_string(),
                "oxide import <DATASET> <FILE> -g http://example.org/graph".to_string(),
            ],
            arguments: vec![
                ArgumentHelp {
                    name: "DATASET".to_string(),
                    description: "Target dataset for import".to_string(),
                    required: true,
                    value_type: "STRING".to_string(),
                    default: None,
                },
                ArgumentHelp {
                    name: "FILE".to_string(),
                    description: "RDF file to import".to_string(),
                    required: true,
                    value_type: "FILE".to_string(),
                    default: None,
                },
            ],
            options: vec![
                OptionHelp {
                    short: Some('f'),
                    long: "format".to_string(),
                    description: "Input format (auto-detected if not specified)".to_string(),
                    value_name: Some("FORMAT".to_string()),
                    possible_values: vec!["turtle".to_string(), "ntriples".to_string(), "rdfxml".to_string(), "jsonld".to_string()],
                    default: None,
                },
                OptionHelp {
                    short: Some('g'),
                    long: "graph".to_string(),
                    description: "Named graph URI for import".to_string(),
                    value_name: Some("URI".to_string()),
                    possible_values: vec![],
                    default: Some("default".to_string()),
                },
            ],
            examples: vec![
                Example {
                    description: "Import Turtle file".to_string(),
                    command: "oxide import mydb data.ttl".to_string(),
                    output: None,
                    explanation: Some("Auto-detects Turtle format from .ttl extension".to_string()),
                },
                Example {
                    description: "Import into named graph".to_string(),
                    command: "oxide import mydb ontology.rdf -g http://example.org/ontology".to_string(),
                    output: None,
                    explanation: Some("Imports RDF/XML file into a specific named graph".to_string()),
                },
            ],
            see_also: vec!["export".to_string(), "tdbloader".to_string(), "riot".to_string()],
            category: HelpCategory::DataManagement,
        });
    }
    
    /// Initialize help topics
    fn init_topic_help(&mut self) {
        // Getting Started topic
        self.topics.insert("getting-started".to_string(), TopicHelp {
            title: "Getting Started with Oxide".to_string(),
            content: r#"
Oxide is the command-line interface for OxiRS, providing comprehensive tools for RDF data management,
SPARQL operations, and semantic web development.

## Quick Start

1. Initialize a new dataset:
   oxide init mydb --format tdb2

2. Import some data:
   oxide import mydb data.ttl

3. Query your data:
   oxide query mydb "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

## Key Concepts

- **Dataset**: A collection of RDF graphs stored together
- **Graph**: A set of RDF triples (default graph or named graphs)
- **SPARQL**: Query language for RDF data
- **Formats**: Turtle (.ttl), N-Triples (.nt), RDF/XML (.rdf), JSON-LD (.jsonld)

## Common Workflows

### Data Import and Export
Import data → Transform → Query → Export results

### Data Validation
Import data → Validate with SHACL → Fix issues → Re-validate

### Development Workflow
Create dataset → Import ontology → Import data → Test queries → Deploy
"#.to_string(),
            examples: vec![
                Example {
                    description: "Complete workflow example".to_string(),
                    command: "oxide init myproject && oxide import myproject schema.ttl && oxide import myproject data.ttl".to_string(),
                    output: None,
                    explanation: Some("Creates a dataset and imports schema and data".to_string()),
                },
            ],
            see_also: vec!["init".to_string(), "import".to_string(), "query".to_string()],
        });
        
        // SPARQL Help topic
        self.topics.insert("sparql".to_string(), TopicHelp {
            title: "SPARQL Query Language".to_string(),
            content: r#"
SPARQL is the standard query language for RDF data. Oxide supports SPARQL 1.1 Query and Update.

## Query Forms

### SELECT - Return tabular results
SELECT ?subject ?predicate ?object
WHERE { ?subject ?predicate ?object }

### CONSTRUCT - Build new RDF graph
CONSTRUCT { ?s rdfs:label ?label }
WHERE { ?s ?p ?label . FILTER(LANG(?label) = "en") }

### ASK - Test if pattern exists
ASK WHERE { ?x rdf:type foaf:Person }

### DESCRIBE - Get information about resources
DESCRIBE <http://example.org/resource>

## Common Patterns

### Optional Patterns
SELECT ?person ?name ?email
WHERE {
  ?person a foaf:Person ;
          foaf:name ?name .
  OPTIONAL { ?person foaf:mbox ?email }
}

### Filtering
SELECT ?title
WHERE {
  ?book dc:title ?title ;
        dc:date ?date .
  FILTER(?date > "2020-01-01"^^xsd:date)
}

### Aggregation
SELECT ?author (COUNT(?book) as ?bookCount)
WHERE { ?book dc:creator ?author }
GROUP BY ?author
ORDER BY DESC(?bookCount)
"#.to_string(),
            examples: vec![
                Example {
                    description: "Find all classes".to_string(),
                    command: r#"oxide query mydb "SELECT DISTINCT ?class WHERE { ?s a ?class }""#.to_string(),
                    output: None,
                    explanation: Some("Lists all RDF types used in the dataset".to_string()),
                },
            ],
            see_also: vec!["query".to_string(), "update".to_string(), "arq".to_string()],
        });
    }
    
    /// Initialize examples
    fn init_examples(&mut self) {
        let query_examples = vec![
            Example {
                description: "Basic pattern matching".to_string(),
                command: r#"oxide query mydb "SELECT * WHERE { ?s ?p ?o } LIMIT 10""#.to_string(),
                output: Some("Returns first 10 triples".to_string()),
                explanation: Some("The most basic SPARQL query - matches all triples".to_string()),
            },
            Example {
                description: "Find specific type".to_string(),
                command: r#"oxide query mydb "SELECT ?person WHERE { ?person a foaf:Person }""#.to_string(),
                output: Some("Lists all foaf:Person instances".to_string()),
                explanation: Some("Uses 'a' as shorthand for rdf:type".to_string()),
            },
            Example {
                description: "Property paths".to_string(),
                command: r#"oxide query mydb "SELECT ?descendant WHERE { ?descendant rdfs:subClassOf+ ?ancestor }""#.to_string(),
                output: Some("Finds transitive subclasses".to_string()),
                explanation: Some("The + operator means one or more steps".to_string()),
            },
        ];
        
        self.examples.insert("query".to_string(), query_examples);
    }
    
    /// Display command help
    pub fn show_command_help(&self, command: &str) {
        if let Some(help) = self.commands.get(command) {
            self.print_command_help(help);
        } else {
            println!("{}", format!("No help available for command: {}", command).red());
            self.suggest_similar_commands(command);
        }
    }
    
    /// Print formatted command help
    fn print_command_help(&self, help: &CommandHelp) {
        // Header
        println!("{}", help.name.to_uppercase().bold());
        println!("{}", self.wrap_text(&help.description));
        
        if let Some(ref long_desc) = help.long_description {
            println!();
            println!("{}", self.wrap_text(long_desc));
        }
        
        // Usage
        println!("\n{}", "USAGE:".yellow());
        for usage in &help.usage {
            println!("    {}", usage);
        }
        
        // Arguments
        if !help.arguments.is_empty() {
            println!("\n{}", "ARGUMENTS:".yellow());
            for arg in &help.arguments {
                let required = if arg.required { "" } else { " (optional)" };
                println!("    {:<20} {}{}", 
                    format!("<{}>", arg.name).green(),
                    arg.description,
                    required.dimmed()
                );
                if let Some(ref default) = arg.default {
                    println!("    {:<20} Default: {}", "", default.dimmed());
                }
            }
        }
        
        // Options
        if !help.options.is_empty() {
            println!("\n{}", "OPTIONS:".yellow());
            for opt in &help.options {
                let short = opt.short.map(|c| format!("-{}, ", c)).unwrap_or_default();
                let long = format!("--{}", opt.long);
                let value = opt.value_name.as_ref().map(|v| format!(" <{}>", v)).unwrap_or_default();
                
                println!("    {:<4}{:<20} {}", 
                    short.green(),
                    format!("{}{}", long, value).green(),
                    opt.description
                );
                
                if !opt.possible_values.is_empty() {
                    println!("    {:<25} Possible values: {}", 
                        "",
                        opt.possible_values.join(", ").dimmed()
                    );
                }
                
                if let Some(ref default) = opt.default {
                    println!("    {:<25} Default: {}", "", default.dimmed());
                }
            }
        }
        
        // Examples
        if !help.examples.is_empty() {
            println!("\n{}", "EXAMPLES:".yellow());
            for (i, example) in help.examples.iter().enumerate() {
                if i > 0 { println!(); }
                println!("    # {}", example.description.dimmed());
                println!("    {}", example.command.cyan());
                
                if let Some(ref output) = example.output {
                    println!("    {}", format!("→ {}", output).dimmed());
                }
                
                if let Some(ref explanation) = example.explanation {
                    println!("    {}", self.wrap_text_indent(explanation, 4).dimmed());
                }
            }
        }
        
        // See also
        if !help.see_also.is_empty() {
            println!("\n{}", "SEE ALSO:".yellow());
            println!("    {}", help.see_also.join(", ").cyan());
        }
    }
    
    /// Display topic help
    pub fn show_topic_help(&self, topic: &str) {
        if let Some(help) = self.topics.get(topic) {
            self.print_topic_help(help);
        } else {
            println!("{}", format!("No help available for topic: {}", topic).red());
            self.list_available_topics();
        }
    }
    
    /// Print formatted topic help
    fn print_topic_help(&self, help: &TopicHelp) {
        println!("{}", help.title.bold());
        println!("{}", "=".repeat(help.title.width()).dimmed());
        println!("{}", help.content);
        
        if !help.examples.is_empty() {
            println!("\n{}", "Examples:".yellow());
            for example in &help.examples {
                println!("\n# {}", example.description.dimmed());
                println!("{}", example.command.cyan());
                
                if let Some(ref explanation) = example.explanation {
                    println!("{}", self.wrap_text_indent(explanation, 0).dimmed());
                }
            }
        }
        
        if !help.see_also.is_empty() {
            println!("\n{}", "See also:".yellow());
            println!("{}", help.see_also.join(", ").cyan());
        }
    }
    
    /// List all available commands by category
    pub fn list_commands(&self) {
        let mut by_category: HashMap<HelpCategory, Vec<&CommandHelp>> = HashMap::new();
        
        for cmd in self.commands.values() {
            by_category.entry(cmd.category.clone()).or_default().push(cmd);
        }
        
        println!("{}", "Available Commands".bold());
        println!("{}", "=================".dimmed());
        
        let categories = [
            (HelpCategory::GettingStarted, "Getting Started"),
            (HelpCategory::DataManagement, "Data Management"),
            (HelpCategory::Querying, "Querying"),
            (HelpCategory::Validation, "Validation"),
            (HelpCategory::Storage, "Storage"),
            (HelpCategory::Configuration, "Configuration"),
        ];
        
        for (category, title) in &categories {
            if let Some(commands) = by_category.get(category) {
                println!("\n{}", title.yellow());
                for cmd in commands {
                    println!("  {:<20} {}", cmd.name.green(), cmd.description);
                }
            }
        }
        
        println!("\n{}", "For more help on a specific command:".dimmed());
        println!("{}", "  oxide help <COMMAND>".cyan());
        println!("\n{}", "For help on a topic:".dimmed());
        println!("{}", "  oxide help <TOPIC>".cyan());
    }
    
    /// List available help topics
    pub fn list_available_topics(&self) {
        println!("\n{}", "Available help topics:".yellow());
        for topic in self.topics.keys() {
            println!("  {}", topic.cyan());
        }
    }
    
    /// Suggest similar commands
    fn suggest_similar_commands(&self, input: &str) {
        let commands: Vec<&str> = self.commands.keys().map(|s| s.as_str()).collect();
        if let Some(suggestion) = crate::cli::suggestions::suggest_command(input, &commands) {
            println!("\n{}", suggestion.dimmed());
        }
    }
    
    /// Wrap text to terminal width
    fn wrap_text(&self, text: &str) -> String {
        let options = Options::new(self.terminal_width)
            .subsequent_indent("")
            .break_words(false);
        wrap(text, options).join("\n")
    }
    
    /// Wrap text with indentation
    fn wrap_text_indent(&self, text: &str, indent: usize) -> String {
        let indent_str = " ".repeat(indent);
        let options = Options::new(self.terminal_width - indent)
            .initial_indent(&indent_str)
            .subsequent_indent(&indent_str)
            .break_words(false);
        wrap(text, options).join("\n")
    }
    
    /// Search help content
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let mut results = Vec::new();
        let query_lower = query.to_lowercase();
        
        // Search commands
        for (name, help) in &self.commands {
            let mut score = 0;
            
            if name.contains(&query_lower) {
                score += 10;
            }
            
            if help.description.to_lowercase().contains(&query_lower) {
                score += 5;
            }
            
            if let Some(ref long_desc) = help.long_description {
                if long_desc.to_lowercase().contains(&query_lower) {
                    score += 3;
                }
            }
            
            for example in &help.examples {
                if example.command.to_lowercase().contains(&query_lower) {
                    score += 2;
                }
            }
            
            if score > 0 {
                results.push(SearchResult {
                    result_type: SearchResultType::Command,
                    name: name.clone(),
                    description: help.description.clone(),
                    score,
                });
            }
        }
        
        // Search topics
        for (name, help) in &self.topics {
            let mut score = 0;
            
            if name.contains(&query_lower) {
                score += 10;
            }
            
            if help.title.to_lowercase().contains(&query_lower) {
                score += 7;
            }
            
            if help.content.to_lowercase().contains(&query_lower) {
                score += 3;
            }
            
            if score > 0 {
                results.push(SearchResult {
                    result_type: SearchResultType::Topic,
                    name: name.clone(),
                    description: help.title.clone(),
                    score,
                });
            }
        }
        
        // Sort by score
        results.sort_by(|a, b| b.score.cmp(&a.score));
        results
    }
    
    /// Display search results
    pub fn show_search_results(&self, query: &str, results: &[SearchResult]) {
        if results.is_empty() {
            println!("{}", format!("No results found for: {}", query).yellow());
            return;
        }
        
        println!("{}", format!("Search results for: {}", query).bold());
        println!("{}", "=".repeat(30).dimmed());
        
        for result in results.iter().take(10) {
            let type_str = match result.result_type {
                SearchResultType::Command => "[CMD]".green(),
                SearchResultType::Topic => "[TOPIC]".blue(),
            };
            
            println!("{} {:<20} {}", type_str, result.name.bold(), result.description);
        }
        
        if results.len() > 10 {
            println!("\n{}", format!("... and {} more results", results.len() - 10).dimmed());
        }
    }
}

/// Search result
pub struct SearchResult {
    pub result_type: SearchResultType,
    pub name: String,
    pub description: String,
    pub score: i32,
}

/// Type of search result
pub enum SearchResultType {
    Command,
    Topic,
}

/// Generate markdown documentation
pub mod markdown {
    use super::*;
    use std::io::Write;
    
    /// Generate markdown documentation for all commands
    pub fn generate_command_docs(provider: &HelpProvider, output: &mut dyn Write) -> std::io::Result<()> {
        writeln!(output, "# Oxide CLI Command Reference\n")?;
        
        let mut commands: Vec<_> = provider.commands.values().collect();
        commands.sort_by_key(|c| &c.name);
        
        for cmd in commands {
            generate_command_doc(cmd, output)?;
            writeln!(output)?;
        }
        
        Ok(())
    }
    
    /// Generate markdown for a single command
    fn generate_command_doc(cmd: &CommandHelp, output: &mut dyn Write) -> std::io::Result<()> {
        writeln!(output, "## {}\n", cmd.name)?;
        writeln!(output, "{}\n", cmd.description)?;
        
        if let Some(ref long_desc) = cmd.long_description {
            writeln!(output, "{}\n", long_desc)?;
        }
        
        writeln!(output, "### Usage\n")?;
        writeln!(output, "```bash")?;
        for usage in &cmd.usage {
            writeln!(output, "{}", usage)?;
        }
        writeln!(output, "```\n")?;
        
        if !cmd.arguments.is_empty() {
            writeln!(output, "### Arguments\n")?;
            for arg in &cmd.arguments {
                let required = if arg.required { " *(required)*" } else { " *(optional)*" };
                writeln!(output, "- **{}**{} - {}", arg.name, required, arg.description)?;
                if let Some(ref default) = arg.default {
                    writeln!(output, "  - Default: `{}`", default)?;
                }
            }
            writeln!(output)?;
        }
        
        if !cmd.options.is_empty() {
            writeln!(output, "### Options\n")?;
            for opt in &cmd.options {
                let short = opt.short.map(|c| format!("-{}, ", c)).unwrap_or_default();
                writeln!(output, "- **{}--{}** - {}", short, opt.long, opt.description)?;
                
                if let Some(ref value) = opt.value_name {
                    writeln!(output, "  - Value: `<{}>`", value)?;
                }
                
                if !opt.possible_values.is_empty() {
                    writeln!(output, "  - Possible values: `{}`", opt.possible_values.join("`, `"))?;
                }
                
                if let Some(ref default) = opt.default {
                    writeln!(output, "  - Default: `{}`", default)?;
                }
            }
            writeln!(output)?;
        }
        
        if !cmd.examples.is_empty() {
            writeln!(output, "### Examples\n")?;
            for example in &cmd.examples {
                writeln!(output, "**{}**", example.description)?;
                writeln!(output, "```bash")?;
                writeln!(output, "{}", example.command)?;
                writeln!(output, "```")?;
                
                if let Some(ref explanation) = example.explanation {
                    writeln!(output, "\n{}\n", explanation)?;
                }
            }
        }
        
        if !cmd.see_also.is_empty() {
            writeln!(output, "### See Also\n")?;
            writeln!(output, "{}\n", cmd.see_also.join(", "))?;
        }
        
        Ok(())
    }
}

impl Default for HelpProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_help_provider_creation() {
        let provider = HelpProvider::new();
        assert!(!provider.commands.is_empty());
        assert!(!provider.topics.is_empty());
    }
    
    #[test]
    fn test_search_functionality() {
        let provider = HelpProvider::new();
        let results = provider.search("query");
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.name == "query"));
    }
    
    #[test]
    fn test_markdown_generation() {
        let provider = HelpProvider::new();
        let mut output = Vec::new();
        markdown::generate_command_docs(&provider, &mut output).unwrap();
        let content = String::from_utf8(output).unwrap();
        assert!(content.contains("# Oxide CLI Command Reference"));
        assert!(content.contains("## query"));
    }
}