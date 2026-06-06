//! Types for the interactive REPL mode
//!
//! Contains structs, enums, and helper types used by the interactive SPARQL shell.

use crate::cli::sparql_autocomplete::SparqlAutocompleteProvider;
use rustyline_derive::{Helper, Highlighter, Validator};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// SPARQL query helper for readline with enhanced context-aware completion
#[derive(Helper, Highlighter, Validator)]
pub(crate) struct SparqlHelper {
    pub(crate) autocomplete_provider: SparqlAutocompleteProvider,
}

impl SparqlHelper {
    pub(crate) fn new() -> Self {
        Self {
            autocomplete_provider: SparqlAutocompleteProvider::new(),
        }
    }
}

impl rustyline::completion::Completer for SparqlHelper {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        use crate::cli::completion::{CompletionContext, CompletionProvider};

        // Create completion context from the current line
        let context = CompletionContext::from_line(line, pos);

        // Get completions from the SPARQL autocomplete provider
        let completions = self.autocomplete_provider.get_completions(&context);

        // Find start position for completion (start of current word)
        let line_before = &line[..pos];
        let start = line_before
            .rfind(char::is_whitespace)
            .map(|i| i + 1)
            .unwrap_or(0);

        // Convert CompletionItem to rustyline's String candidates
        let candidates: Vec<String> = completions
            .into_iter()
            .map(|item| item.replacement)
            .collect();

        Ok((start, candidates))
    }
}

impl rustyline::hint::Hinter for SparqlHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        if pos < line.len() {
            return None;
        }

        let line_upper = line.to_uppercase();

        // Provide hints based on current context
        if line_upper.starts_with("SELECT") && !line_upper.contains("WHERE") {
            Some(" WHERE { ?s ?p ?o }".to_string())
        } else if line_upper.starts_with("PREFIX")
            && line.matches(':').count() == 1
            && !line.contains('<')
        {
            Some(" <http://example.org/>".to_string())
        } else if line_upper.ends_with("WHERE") {
            Some(" { ?s ?p ?o }".to_string())
        } else {
            None
        }
    }
}

/// Query session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct QuerySession {
    /// Session name
    pub(crate) name: String,
    /// Dataset connected to
    pub(crate) dataset: String,
    /// Queries executed in this session
    pub(crate) queries: Vec<String>,
    /// Timestamp of session creation
    pub(crate) created_at: String,
    /// Last modified timestamp
    pub(crate) modified_at: String,
}

impl QuerySession {
    /// Create a new session
    pub(crate) fn new(name: String, dataset: String) -> Self {
        let now = chrono::Local::now().to_rfc3339();
        Self {
            name,
            dataset,
            queries: Vec::new(),
            created_at: now.clone(),
            modified_at: now,
        }
    }

    /// Add a query to the session
    pub(crate) fn add_query(&mut self, query: String) {
        self.queries.push(query);
        self.modified_at = chrono::Local::now().to_rfc3339();
    }

    /// Save session to file
    pub(crate) fn save_to_file(&self, path: &PathBuf) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize session: {}", e))?;
        fs::write(path, json).map_err(|e| format!("Failed to write session file: {}", e))?;
        Ok(())
    }

    /// Load session from file
    pub(crate) fn load_from_file(path: &PathBuf) -> Result<Self, String> {
        let json =
            fs::read_to_string(path).map_err(|e| format!("Failed to read session file: {}", e))?;
        let session: QuerySession = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse session file: {}", e))?;
        Ok(session)
    }

    /// Clear all queries from the session
    pub(crate) fn clear(&mut self) {
        self.queries.clear();
        self.modified_at = chrono::Local::now().to_rfc3339();
    }
}
