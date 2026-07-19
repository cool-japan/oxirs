//! REPL extension commands for the interactive SPARQL shell.
//!
//! Wires the standalone REPL-support modules (`query_bookmarks`,
//! `result_export`, `ascii_diagram`, `dataset_manager`, `visual_query_builder`,
//! `fuzzy_history`, and `schema_autocomplete`) into the live interactive loop
//! driven by [`crate::commands::interactive_session::execute`].
//!
//! The main loop hands every line that starts with `:` to
//! [`dispatch_colon_command`], which parses the meta-command and mutates the
//! shared [`ReplState`]. Commands that need to run a SPARQL query (for example
//! `:bookmark run` and `:visual`) do not execute it here — they return
//! [`ColonOutcome::Inject`] so the query flows back through the loop's normal
//! execution and result-capture path, keeping a single execution site.

use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use oxirs_core::rdf_store::RdfStore;

use crate::cli::ascii_diagram::{AsciiDiagramGenerator, DiagramConfig, DiagramTriple, LayoutStyle};
use crate::cli::dataset_manager::DatasetManager;
use crate::cli::error::CliResult;
use crate::cli::formatters::QueryResults as FormatterQueryResults;
use crate::cli::fuzzy_history::FuzzyHistorySearch;
use crate::cli::query_bookmarks::{BookmarkConfig, BookmarkManager};
use crate::cli::result_export::{ExportFormat, ResultExporter};
use crate::cli::schema_autocomplete::SchemaAutocompleteProvider;
use crate::cli::visual_query_builder::VisualQueryBuilder;

/// Outcome of dispatching a `:`-prefixed REPL meta-command.
#[derive(Debug)]
pub enum ColonOutcome {
    /// The command was handled fully; the loop should read the next line.
    Handled,
    /// The command produced a SPARQL query that should be executed as if the
    /// user had typed it (used by `:bookmark run` and `:visual`).
    Inject(String),
    /// The active dataset changed; the loop should refresh schema completion.
    StoreChanged,
    /// The token after `:` was not a recognized command.
    Unknown,
}

/// Shared mutable state backing the interactive REPL extension commands.
///
/// Owns the active store handle (shared with the schema autocomplete provider),
/// the multi-dataset manager, the bookmark store, the fuzzy history searcher,
/// and the "last result" slots used by `:export` and `:diagram`.
pub struct ReplState {
    /// Active dataset store, shared with [`Self::schema`] via the `Arc`.
    pub store: Arc<RwLock<RdfStore>>,
    /// Name of the currently active dataset.
    pub active_dataset: String,
    /// Registered datasets and the active-connection pointer.
    pub datasets: DatasetManager,
    /// Persistent query bookmarks.
    pub bookmarks: BookmarkManager,
    /// Fuzzy history searcher for `:hsearch`.
    pub fuzzy: FuzzyHistorySearch,
    /// Schema discovery provider feeding schema-aware tab completion.
    pub schema: SchemaAutocompleteProvider,
    /// Last query result (any form), for `:export`.
    pub last_result: Option<FormatterQueryResults>,
    /// Last CONSTRUCT/DESCRIBE graph, for `:diagram`.
    pub last_graph: Option<Vec<DiagramTriple>>,
}

impl ReplState {
    /// Build REPL state around an already-opened store.
    ///
    /// `dataset_location` is the on-disk path of the initial dataset so it can
    /// be re-selected later with `:dataset use`. `base_dir` is the session state
    /// directory; bookmarks persist to `base_dir/bookmarks.json`.
    pub fn new(
        store: Arc<RwLock<RdfStore>>,
        dataset_name: String,
        dataset_location: String,
        base_dir: &Path,
    ) -> CliResult<Self> {
        let mut datasets = DatasetManager::new();
        datasets.connect(dataset_name.clone(), dataset_location)?;

        let bookmarks = BookmarkManager::with_config(BookmarkConfig::with_path(
            base_dir.join("bookmarks.json"),
        ))?;

        let schema = SchemaAutocompleteProvider::new(Arc::clone(&store));

        Ok(Self {
            store,
            active_dataset: dataset_name,
            datasets,
            bookmarks,
            fuzzy: FuzzyHistorySearch::new(),
            schema,
            last_result: None,
            last_graph: None,
        })
    }

    /// Execute a SPARQL query against the active store.
    ///
    /// Errors are surfaced as strings; the store is never bypassed and no
    /// fabricated result is returned on failure.
    pub fn query(&self, sparql: &str) -> Result<oxirs_core::rdf_store::OxirsQueryResults, String> {
        let guard = self.store.read().map_err(|e| e.to_string())?;
        guard.query(sparql).map_err(|e| e.to_string())
    }

    /// Collect the class and property URIs discovered from the active dataset,
    /// used to feed schema-aware completion.
    pub fn schema_terms(&self) -> Vec<String> {
        match self.schema.get_schema() {
            Ok(schema) => {
                let mut terms: Vec<String> =
                    Vec::with_capacity(schema.classes.len() + schema.properties.len());
                terms.extend(schema.classes.iter().cloned());
                terms.extend(schema.properties.iter().cloned());
                terms
            }
            Err(_) => Vec::new(),
        }
    }
}

/// Parse and dispatch a `:`-prefixed REPL meta-command.
///
/// `history` is the session's query history (oldest first) used by `:hsearch`.
pub fn dispatch_colon_command(
    state: &mut ReplState,
    history: &[String],
    line: &str,
) -> ColonOutcome {
    let trimmed = line.trim();
    let without = trimmed.strip_prefix(':').unwrap_or(trimmed);
    let mut parts = without.splitn(2, char::is_whitespace);
    let command = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    match command {
        "help" | "h" => {
            print_extension_help();
            ColonOutcome::Handled
        }
        "bookmark" | "bm" => handle_bookmark(state, rest),
        "export" => handle_export(state, rest),
        "diagram" => handle_diagram(state, rest),
        "dataset" | "ds" => handle_dataset(state, rest),
        "visual" => handle_visual(),
        "hsearch" => handle_hsearch(state, history, rest),
        _ => ColonOutcome::Unknown,
    }
}

fn handle_bookmark(state: &mut ReplState, rest: &str) -> ColonOutcome {
    let mut parts = rest.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim();

    match sub {
        "add" => {
            let mut inner = arg.splitn(2, char::is_whitespace);
            let name = inner.next().unwrap_or("").trim();
            let query = inner.next().unwrap_or("").trim();
            if name.is_empty() || query.is_empty() {
                eprintln!("Usage: :bookmark add <name> <query>");
                return ColonOutcome::Handled;
            }
            match state.bookmarks.save(name.to_string(), query.to_string()) {
                Ok(_) => println!("Bookmark '{name}' saved"),
                Err(e) => eprintln!("Error saving bookmark: {e}"),
            }
            ColonOutcome::Handled
        }
        "list" | "ls" => {
            let bookmarks = state.bookmarks.list();
            if bookmarks.is_empty() {
                println!("No bookmarks saved. Add one with :bookmark add <name> <query>");
            } else {
                println!("Bookmarks ({}):", bookmarks.len());
                for bookmark in bookmarks {
                    let first_line = bookmark.query.lines().next().unwrap_or("");
                    println!(
                        "  {} (used {}x): {}",
                        bookmark.name, bookmark.use_count, first_line
                    );
                }
            }
            ColonOutcome::Handled
        }
        "run" => {
            if arg.is_empty() {
                eprintln!("Usage: :bookmark run <name>");
                return ColonOutcome::Handled;
            }
            match state.bookmarks.get_query(arg) {
                Ok(query) => {
                    println!("Running bookmark '{arg}'");
                    ColonOutcome::Inject(query)
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    ColonOutcome::Handled
                }
            }
        }
        "rm" | "delete" | "del" => {
            if arg.is_empty() {
                eprintln!("Usage: :bookmark rm <name>");
                return ColonOutcome::Handled;
            }
            match state.bookmarks.delete(arg) {
                Ok(_) => println!("Bookmark '{arg}' removed"),
                Err(e) => eprintln!("Error: {e}"),
            }
            ColonOutcome::Handled
        }
        _ => {
            eprintln!("Usage: :bookmark <add|list|run|rm> ...");
            ColonOutcome::Handled
        }
    }
}

fn handle_export(state: &mut ReplState, rest: &str) -> ColonOutcome {
    let mut parts = rest.splitn(2, char::is_whitespace);
    let format_str = parts.next().unwrap_or("").trim();
    let path_str = parts.next().unwrap_or("").trim();

    if format_str.is_empty() || path_str.is_empty() {
        eprintln!("Usage: :export <csv|json|html|xlsx> <path>");
        return ColonOutcome::Handled;
    }

    let format = match ExportFormat::parse(format_str) {
        Some(format) => format,
        None => {
            eprintln!("Unknown export format '{format_str}' (use csv, json, html, or xlsx)");
            return ColonOutcome::Handled;
        }
    };

    let results = match &state.last_result {
        Some(results) => results,
        None => {
            eprintln!("No query result to export. Run a query first.");
            return ColonOutcome::Handled;
        }
    };

    let exporter = ResultExporter::new(format);
    match exporter.export_to_file(results, Path::new(path_str)) {
        Ok(_) => println!(
            "Exported {} row(s) to {} as {}",
            results.bindings.len(),
            path_str,
            format
        ),
        Err(e) => eprintln!("Export failed: {e}"),
    }
    ColonOutcome::Handled
}

fn handle_diagram(state: &mut ReplState, rest: &str) -> ColonOutcome {
    let style = match rest.trim().to_lowercase().as_str() {
        "" | "tree" => LayoutStyle::Tree,
        "graph" => LayoutStyle::Graph,
        "compact" => LayoutStyle::Compact,
        "list" => LayoutStyle::List,
        other => {
            eprintln!("Unknown diagram style '{other}' (use tree, graph, compact, or list)");
            return ColonOutcome::Handled;
        }
    };

    let triples: Vec<DiagramTriple> = match &state.last_graph {
        Some(graph) => graph.clone(),
        None => match sample_dataset_triples(&state.store, 50) {
            Ok(triples) => triples,
            Err(e) => {
                eprintln!("Error reading dataset for diagram: {e}");
                return ColonOutcome::Handled;
            }
        },
    };

    if triples.is_empty() {
        println!(
            "No triples to diagram. Run a CONSTRUCT/DESCRIBE query or load data into the dataset."
        );
        return ColonOutcome::Handled;
    }

    let config = DiagramConfig {
        style,
        ..DiagramConfig::default()
    };
    let generator = AsciiDiagramGenerator::new(config);
    let mut stdout = std::io::stdout();
    if let Err(e) = generator.generate(&triples, &mut stdout) {
        eprintln!("Diagram generation failed: {e}");
    }
    ColonOutcome::Handled
}

fn handle_dataset(state: &mut ReplState, rest: &str) -> ColonOutcome {
    let mut parts = rest.splitn(2, char::is_whitespace);
    let sub = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim();

    match sub {
        "add" => {
            let mut inner = arg.splitn(2, char::is_whitespace);
            let name = inner.next().unwrap_or("").trim();
            let location = inner.next().unwrap_or("").trim();
            if name.is_empty() || location.is_empty() {
                eprintln!("Usage: :dataset add <name> <path>");
                return ColonOutcome::Handled;
            }
            match state
                .datasets
                .connect(name.to_string(), location.to_string())
            {
                Ok(_) => println!("Dataset '{name}' registered at {location}"),
                Err(e) => eprintln!("Error: {e}"),
            }
            ColonOutcome::Handled
        }
        "use" | "switch" => {
            if arg.is_empty() {
                eprintln!("Usage: :dataset use <name>");
                return ColonOutcome::Handled;
            }
            let location = match state.datasets.get(arg) {
                Some(connection) => connection.location.clone(),
                None => {
                    eprintln!("Dataset '{arg}' not found. Register it first with :dataset add.");
                    return ColonOutcome::Handled;
                }
            };

            let path = PathBuf::from(&location);
            if !path.is_dir() {
                eprintln!("Dataset location is not a directory: {location}");
                return ColonOutcome::Handled;
            }

            let new_store = match RdfStore::open(&path) {
                Ok(store) => store,
                Err(e) => {
                    eprintln!("Failed to open dataset '{arg}': {e}");
                    return ColonOutcome::Handled;
                }
            };

            if let Err(e) = state.datasets.switch(arg) {
                eprintln!("Error: {e}");
                return ColonOutcome::Handled;
            }

            match state.store.write() {
                Ok(mut guard) => *guard = new_store,
                Err(e) => {
                    eprintln!("Store lock error: {e}");
                    return ColonOutcome::Handled;
                }
            }

            state.active_dataset = arg.to_string();
            state.last_result = None;
            state.last_graph = None;
            state.schema.invalidate_cache();
            println!("Switched to dataset '{arg}'");
            ColonOutcome::StoreChanged
        }
        "list" | "ls" => {
            let active = state.datasets.active_name();
            let connections = state.datasets.list();
            if connections.is_empty() {
                println!("No datasets registered.");
            } else {
                println!("Datasets ({}):", connections.len());
                for connection in connections {
                    let marker = if Some(&connection.name) == active.as_ref() {
                        "*"
                    } else {
                        " "
                    };
                    println!(
                        "  {} {} -> {}",
                        marker, connection.name, connection.location
                    );
                }
            }
            ColonOutcome::Handled
        }
        _ => {
            eprintln!("Usage: :dataset <add|use|list> ...");
            ColonOutcome::Handled
        }
    }
}

fn handle_visual() -> ColonOutcome {
    let mut builder = VisualQueryBuilder::new();
    match builder.build_interactive() {
        Ok(query) => {
            if query.trim().is_empty() {
                eprintln!("Visual builder produced an empty query.");
                ColonOutcome::Handled
            } else {
                ColonOutcome::Inject(query)
            }
        }
        Err(e) => {
            eprintln!("Visual query builder cancelled: {e}");
            ColonOutcome::Handled
        }
    }
}

fn handle_hsearch(state: &ReplState, history: &[String], rest: &str) -> ColonOutcome {
    if rest.is_empty() {
        eprintln!("Usage: :hsearch <term>");
        return ColonOutcome::Handled;
    }

    // `FuzzyHistorySearch` expects the history most-recent-first; the session
    // stores queries oldest-first, so reverse before searching.
    let recent_first: Vec<String> = history.iter().rev().cloned().collect();
    let matches = state.fuzzy.search(rest, &recent_first);

    if matches.is_empty() {
        println!("No history entries fuzzy-match '{rest}'");
        return ColonOutcome::Handled;
    }

    println!("Fuzzy history matches for '{rest}':");
    for fuzzy_match in &matches {
        let first_line = fuzzy_match.query.lines().next().unwrap_or("");
        println!("  [{:>3.0}%] {}", fuzzy_match.score * 100.0, first_line);
    }
    ColonOutcome::Handled
}

/// Sample up to `limit` triples from the active store for diagram fallback.
fn sample_dataset_triples(
    store: &Arc<RwLock<RdfStore>>,
    limit: usize,
) -> Result<Vec<DiagramTriple>, String> {
    let guard = store.read().map_err(|e| e.to_string())?;
    let triples = guard.triples().map_err(|e| e.to_string())?;
    Ok(triples
        .iter()
        .take(limit)
        .map(|triple| DiagramTriple {
            subject: triple.subject().to_string(),
            predicate: triple.predicate().to_string(),
            object: triple.object().to_string(),
        })
        .collect())
}

/// Print the `:`-command reference shown by `:help` (and appended to `.help`).
pub fn print_extension_help() {
    println!("REPL Extension Commands (: prefix):");
    println!("  :bookmark add <name> <query>   Save a query bookmark");
    println!("  :bookmark list                 List saved bookmarks");
    println!("  :bookmark run <name>           Load and execute a bookmarked query");
    println!("  :bookmark rm <name>            Delete a bookmark");
    println!("  :export <format> <path>        Export the LAST result (csv|json|html|xlsx)");
    println!("  :diagram [style]               ASCII diagram of the last graph or a sample");
    println!("                                 (styles: tree, graph, compact, list)");
    println!("  :dataset add <name> <path>     Register another dataset");
    println!("  :dataset use <name>            Switch the active dataset");
    println!("  :dataset list                  List registered datasets");
    println!("  :visual                        Build a query interactively into the buffer");
    println!("  :hsearch <term>                Fuzzy-search the session query history");
    println!("  :help                          Show this extension help");
    println!("  Tab                            Schema-aware completion (classes/properties)");
    println!();
}
