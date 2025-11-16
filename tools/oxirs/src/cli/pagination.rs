//! Result Set Pagination for Interactive REPL
//!
//! Provides paginated viewing of large SPARQL query results with navigation
//! commands and configurable page sizes.

use crate::cli::formatters::{Binding, QueryResults};
use crate::cli::CliResult;
use std::io::{self, Write};

/// Configuration for pagination behavior
#[derive(Debug, Clone)]
pub struct PaginationConfig {
    /// Number of results per page (default: 20)
    pub page_size: usize,
    /// Whether to show page numbers (default: true)
    pub show_page_numbers: bool,
    /// Whether to show navigation hints (default: true)
    pub show_navigation_hints: bool,
    /// Whether to show result count (default: true)
    pub show_result_count: bool,
}

impl Default for PaginationConfig {
    fn default() -> Self {
        Self {
            page_size: 20,
            show_page_numbers: true,
            show_navigation_hints: true,
            show_result_count: true,
        }
    }
}

impl PaginationConfig {
    /// Create configuration for small result sets (larger page size)
    pub fn small() -> Self {
        Self {
            page_size: 50,
            ..Default::default()
        }
    }

    /// Create configuration for large result sets (smaller page size)
    pub fn large() -> Self {
        Self {
            page_size: 10,
            ..Default::default()
        }
    }

    /// Create configuration with custom page size
    pub fn with_page_size(page_size: usize) -> Self {
        Self {
            page_size,
            ..Default::default()
        }
    }
}

/// Paginated view of query results
pub struct ResultPaginator {
    /// All results to paginate
    results: QueryResults,
    /// Current page (0-indexed)
    current_page: usize,
    /// Configuration
    config: PaginationConfig,
}

impl ResultPaginator {
    /// Create a new result paginator
    pub fn new(results: QueryResults) -> Self {
        Self {
            results,
            current_page: 0,
            config: PaginationConfig::default(),
        }
    }

    /// Create a paginator with custom configuration
    pub fn with_config(results: QueryResults, config: PaginationConfig) -> Self {
        Self {
            results,
            current_page: 0,
            config,
        }
    }

    /// Get the total number of results
    pub fn total_results(&self) -> usize {
        self.results.bindings.len()
    }

    /// Get the total number of pages
    pub fn total_pages(&self) -> usize {
        (self.total_results() + self.config.page_size - 1) / self.config.page_size
    }

    /// Get the current page number (1-indexed)
    pub fn current_page_number(&self) -> usize {
        self.current_page + 1
    }

    /// Check if there is a next page
    pub fn has_next_page(&self) -> bool {
        self.current_page < self.total_pages() - 1
    }

    /// Check if there is a previous page
    pub fn has_previous_page(&self) -> bool {
        self.current_page > 0
    }

    /// Move to the next page
    pub fn next_page(&mut self) -> bool {
        if self.has_next_page() {
            self.current_page += 1;
            true
        } else {
            false
        }
    }

    /// Move to the previous page
    pub fn previous_page(&mut self) -> bool {
        if self.has_previous_page() {
            self.current_page -= 1;
            true
        } else {
            false
        }
    }

    /// Move to the first page
    pub fn first_page(&mut self) {
        self.current_page = 0;
    }

    /// Move to the last page
    pub fn last_page(&mut self) {
        if self.total_pages() > 0 {
            self.current_page = self.total_pages() - 1;
        }
    }

    /// Move to a specific page (0-indexed)
    pub fn go_to_page(&mut self, page: usize) -> bool {
        if page < self.total_pages() {
            self.current_page = page;
            true
        } else {
            false
        }
    }

    /// Get the bindings for the current page
    pub fn current_page_bindings(&self) -> &[Binding] {
        let start = self.current_page * self.config.page_size;
        let end = ((self.current_page + 1) * self.config.page_size).min(self.total_results());
        &self.results.bindings[start..end]
    }

    /// Get results for the current page
    pub fn current_page_results(&self) -> QueryResults {
        QueryResults {
            variables: self.results.variables.clone(),
            bindings: self.current_page_bindings().to_vec(),
        }
    }

    /// Display the current page with navigation hints
    pub fn display_page(&self) -> CliResult<()> {
        let page_results = self.current_page_results();

        // Show result count if enabled
        if self.config.show_result_count {
            println!(
                "\n{} result{} total",
                self.total_results(),
                if self.total_results() == 1 { "" } else { "s" }
            );
        }

        // Show page numbers if enabled
        if self.config.show_page_numbers && self.total_pages() > 1 {
            println!(
                "Page {} of {} (showing {}-{} of {})",
                self.current_page_number(),
                self.total_pages(),
                self.current_page * self.config.page_size + 1,
                (self.current_page * self.config.page_size + page_results.bindings.len()),
                self.total_results()
            );
        }

        // Display the bindings (using simple table format)
        println!();
        self.display_bindings(&page_results)?;

        // Show navigation hints if enabled
        if self.config.show_navigation_hints && self.total_pages() > 1 {
            println!();
            let mut hints = Vec::new();

            if self.has_previous_page() {
                hints.push("[p]revious");
            }
            if self.has_next_page() {
                hints.push("[n]ext");
            }
            hints.push("[f]irst");
            hints.push("[l]ast");
            hints.push("[g]oto");
            hints.push("[q]uit");

            println!("Navigation: {}", hints.join(" | "));
        }

        Ok(())
    }

    /// Display bindings in a simple table format
    fn display_bindings(&self, results: &QueryResults) -> CliResult<()> {
        if results.bindings.is_empty() {
            println!("(no results on this page)");
            return Ok(());
        }

        // Calculate column widths
        let mut col_widths: Vec<usize> = results.variables.iter().map(|v| v.len() + 1).collect();

        for binding in &results.bindings {
            for (i, _var) in results.variables.iter().enumerate() {
                if i < binding.values.len() {
                    if let Some(term) = &binding.values[i] {
                        let term_str = self.format_term(term);
                        col_widths[i] = col_widths[i].max(term_str.len() + 2);
                    }
                }
            }
        }

        // Print header
        print!("│");
        for (i, var) in results.variables.iter().enumerate() {
            print!(" {:width$}│", var, width = col_widths[i]);
        }
        println!();

        // Print separator
        print!("├");
        for width in &col_widths {
            print!("{}┼", "─".repeat(width + 1));
        }
        // Replace last ┼ with ┤
        println!("\x08┤");

        // Print rows
        for binding in &results.bindings {
            print!("│");
            for (i, _var) in results.variables.iter().enumerate() {
                let term_str = if i < binding.values.len() {
                    binding.values[i]
                        .as_ref()
                        .map(|t| self.format_term(t))
                        .unwrap_or_else(|| String::from(""))
                } else {
                    String::from("")
                };
                print!(" {:width$}│", term_str, width = col_widths[i]);
            }
            println!();
        }

        Ok(())
    }

    /// Format an RDF term for display
    fn format_term(&self, term: &crate::cli::formatters::RdfTerm) -> String {
        use crate::cli::formatters::RdfTerm;
        match term {
            RdfTerm::Uri { value } => {
                // Try to show local name for readability
                if let Some(hash_pos) = value.rfind('#') {
                    value[hash_pos + 1..].to_string()
                } else if let Some(slash_pos) = value.rfind('/') {
                    value[slash_pos + 1..].to_string()
                } else {
                    value.clone()
                }
            }
            RdfTerm::Literal {
                value,
                lang: Some(lang),
                ..
            } => format!("\"{}\"@{}", value, lang),
            RdfTerm::Literal {
                value,
                datatype: Some(dt),
                ..
            } => format!("\"{}\"^^{}", value, dt),
            RdfTerm::Literal { value, .. } => format!("\"{}\"", value),
            RdfTerm::Bnode { value } => format!("_:{}", value),
        }
    }

    /// Interactive navigation loop
    pub fn interactive_navigate(&mut self) -> CliResult<()> {
        loop {
            // Display current page
            self.display_page()?;

            // If only one page, don't prompt for navigation
            if self.total_pages() <= 1 {
                return Ok(());
            }

            // Prompt for navigation command
            print!("\nCommand: ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;

            match input.trim().to_lowercase().as_str() {
                "n" | "next" => {
                    if !self.next_page() {
                        println!("Already on last page");
                    }
                }
                "p" | "prev" | "previous" => {
                    if !self.previous_page() {
                        println!("Already on first page");
                    }
                }
                "f" | "first" => {
                    self.first_page();
                }
                "l" | "last" => {
                    self.last_page();
                }
                "g" | "goto" => {
                    print!("Page number (1-{}): ", self.total_pages());
                    io::stdout().flush()?;

                    let mut page_input = String::new();
                    io::stdin().read_line(&mut page_input)?;

                    match page_input.trim().parse::<usize>() {
                        Ok(page) if page > 0 && page <= self.total_pages() => {
                            self.go_to_page(page - 1);
                        }
                        _ => {
                            println!("Invalid page number");
                        }
                    }
                }
                "q" | "quit" | "exit" => {
                    return Ok(());
                }
                "" => {
                    // Enter with no input = next page
                    if !self.next_page() {
                        return Ok(());
                    }
                }
                _ => {
                    println!("Unknown command. Use n/p/f/l/g/q or press Enter for next page");
                }
            }
        }
    }
}

/// Pagination navigation command
#[derive(Debug, Clone, PartialEq)]
pub enum NavigationCommand {
    Next,
    Previous,
    First,
    Last,
    GoTo(usize),
    Quit,
}

impl NavigationCommand {
    /// Parse a command from user input
    pub fn parse(input: &str) -> Option<Self> {
        match input.trim().to_lowercase().as_str() {
            "n" | "next" => Some(NavigationCommand::Next),
            "p" | "prev" | "previous" => Some(NavigationCommand::Previous),
            "f" | "first" => Some(NavigationCommand::First),
            "l" | "last" => Some(NavigationCommand::Last),
            "q" | "quit" | "exit" => Some(NavigationCommand::Quit),
            s if s.starts_with("g ") || s.starts_with("goto ") => {
                let parts: Vec<&str> = s.split_whitespace().collect();
                if parts.len() == 2 {
                    parts[1].parse::<usize>().ok().map(NavigationCommand::GoTo)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::formatters::{Binding, RdfTerm};

    fn create_test_results(num_bindings: usize) -> QueryResults {
        let variables = vec!["x".to_string(), "y".to_string()];
        let mut bindings = Vec::new();

        for i in 0..num_bindings {
            let values = vec![
                Some(RdfTerm::Uri {
                    value: format!("http://example.org/x{}", i),
                }),
                Some(RdfTerm::Literal {
                    value: i.to_string(),
                    lang: None,
                    datatype: None,
                }),
            ];
            bindings.push(Binding { values });
        }

        QueryResults {
            variables,
            bindings,
        }
    }

    #[test]
    fn test_paginator_creation() {
        let results = create_test_results(50);
        let paginator = ResultPaginator::new(results);

        assert_eq!(paginator.total_results(), 50);
        assert_eq!(paginator.current_page_number(), 1);
    }

    #[test]
    fn test_page_calculation() {
        let results = create_test_results(50);
        let config = PaginationConfig::with_page_size(20);
        let paginator = ResultPaginator::with_config(results, config);

        assert_eq!(paginator.total_pages(), 3);
        assert_eq!(paginator.current_page_bindings().len(), 20);
    }

    #[test]
    fn test_navigation() {
        let results = create_test_results(50);
        let mut paginator = ResultPaginator::new(results);

        assert_eq!(paginator.current_page_number(), 1);
        assert!(!paginator.has_previous_page());
        assert!(paginator.has_next_page());

        assert!(paginator.next_page());
        assert_eq!(paginator.current_page_number(), 2);

        assert!(paginator.previous_page());
        assert_eq!(paginator.current_page_number(), 1);
    }

    #[test]
    fn test_first_last_page() {
        let results = create_test_results(50);
        let mut paginator = ResultPaginator::new(results);

        paginator.last_page();
        assert_eq!(paginator.current_page_number(), paginator.total_pages());

        paginator.first_page();
        assert_eq!(paginator.current_page_number(), 1);
    }

    #[test]
    fn test_goto_page() {
        let results = create_test_results(50);
        let mut paginator = ResultPaginator::new(results);

        assert!(paginator.go_to_page(1));
        assert_eq!(paginator.current_page_number(), 2);

        assert!(!paginator.go_to_page(10));
        assert_eq!(paginator.current_page_number(), 2);
    }

    #[test]
    fn test_page_boundaries() {
        let results = create_test_results(25);
        let config = PaginationConfig::with_page_size(10);
        let mut paginator = ResultPaginator::with_config(results, config);

        assert_eq!(paginator.total_pages(), 3);

        // First page: 10 items
        assert_eq!(paginator.current_page_bindings().len(), 10);

        // Second page: 10 items
        paginator.next_page();
        assert_eq!(paginator.current_page_bindings().len(), 10);

        // Third page: 5 items (remainder)
        paginator.next_page();
        assert_eq!(paginator.current_page_bindings().len(), 5);
    }

    #[test]
    fn test_single_page() {
        let results = create_test_results(10);
        let config = PaginationConfig::with_page_size(20);
        let paginator = ResultPaginator::with_config(results, config);

        assert_eq!(paginator.total_pages(), 1);
        assert!(!paginator.has_next_page());
        assert!(!paginator.has_previous_page());
    }

    #[test]
    fn test_empty_results() {
        let results = create_test_results(0);
        let paginator = ResultPaginator::new(results);

        assert_eq!(paginator.total_results(), 0);
        assert_eq!(paginator.total_pages(), 0);
        assert_eq!(paginator.current_page_bindings().len(), 0);
    }

    #[test]
    fn test_navigation_command_parsing() {
        assert_eq!(NavigationCommand::parse("n"), Some(NavigationCommand::Next));
        assert_eq!(
            NavigationCommand::parse("next"),
            Some(NavigationCommand::Next)
        );
        assert_eq!(
            NavigationCommand::parse("p"),
            Some(NavigationCommand::Previous)
        );
        assert_eq!(
            NavigationCommand::parse("prev"),
            Some(NavigationCommand::Previous)
        );
        assert_eq!(
            NavigationCommand::parse("f"),
            Some(NavigationCommand::First)
        );
        assert_eq!(NavigationCommand::parse("l"), Some(NavigationCommand::Last));
        assert_eq!(NavigationCommand::parse("q"), Some(NavigationCommand::Quit));
        assert_eq!(
            NavigationCommand::parse("g 5"),
            Some(NavigationCommand::GoTo(5))
        );
        assert_eq!(
            NavigationCommand::parse("goto 10"),
            Some(NavigationCommand::GoTo(10))
        );
        assert_eq!(NavigationCommand::parse("invalid"), None);
    }

    #[test]
    fn test_config_presets() {
        let small_config = PaginationConfig::small();
        assert_eq!(small_config.page_size, 50);

        let large_config = PaginationConfig::large();
        assert_eq!(large_config.page_size, 10);

        let custom_config = PaginationConfig::with_page_size(100);
        assert_eq!(custom_config.page_size, 100);
    }
}
