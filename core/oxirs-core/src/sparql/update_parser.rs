//! SPARQL UPDATE statement parser.
//!
//! Parses SPARQL 1.1 Update operations including INSERT DATA, DELETE DATA,
//! DELETE/INSERT WHERE, LOAD, CLEAR, DROP, CREATE, COPY, MOVE, and ADD.
//! Reports parse errors with position information.

use std::collections::HashMap;
use std::fmt;

// ────────────────────────────────────────────────────────────────────────────
// Error types
// ────────────────────────────────────────────────────────────────────────────

/// Error produced when parsing a SPARQL Update request fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateParseError {
    /// Human-readable description of the error.
    pub message: String,
    /// Byte offset in the input where the error was detected.
    pub position: usize,
    /// Optional line number (1-based) for display purposes.
    pub line: Option<usize>,
    /// Optional column number (1-based) for display purposes.
    pub column: Option<usize>,
}

impl UpdateParseError {
    /// Create a new parse error at the given byte offset.
    pub fn new(message: impl Into<String>, position: usize) -> Self {
        Self {
            message: message.into(),
            position,
            line: None,
            column: None,
        }
    }

    /// Attach line/column information derived from the original source.
    pub fn with_location(mut self, line: usize, column: usize) -> Self {
        self.line = Some(line);
        self.column = Some(column);
        self
    }
}

impl fmt::Display for UpdateParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.line, self.column) {
            (Some(ln), Some(col)) => {
                write!(f, "update parse error at {}:{}: {}", ln, col, self.message)
            }
            _ => write!(
                f,
                "update parse error at byte {}: {}",
                self.position, self.message
            ),
        }
    }
}

impl std::error::Error for UpdateParseError {}

/// Compute (line, column) from a byte offset and the original source text.
pub(crate) fn line_col(source: &str, byte_offset: usize) -> (usize, usize) {
    let prefix = &source[..byte_offset.min(source.len())];
    let line = prefix.chars().filter(|&c| c == '\n').count() + 1;
    let last_newline = prefix.rfind('\n').map(|p| p + 1).unwrap_or(0);
    let col = byte_offset.saturating_sub(last_newline) + 1;
    (line, col)
}

// ────────────────────────────────────────────────────────────────────────────
// AST types
// ────────────────────────────────────────────────────────────────────────────

/// A single triple pattern inside a SPARQL Update template or WHERE clause.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriplePattern {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl TriplePattern {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// Target graph specification used in CLEAR / DROP / COPY / MOVE / ADD.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphTarget {
    /// A specific named graph identified by IRI.
    Graph(String),
    /// The default graph.
    Default,
    /// All named graphs.
    Named,
    /// Default graph + all named graphs.
    All,
}

impl fmt::Display for GraphTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphTarget::Graph(iri) => write!(f, "GRAPH <{}>", iri),
            GraphTarget::Default => write!(f, "DEFAULT"),
            GraphTarget::Named => write!(f, "NAMED"),
            GraphTarget::All => write!(f, "ALL"),
        }
    }
}

/// A parsed SPARQL Update operation.
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateOperation {
    /// `INSERT DATA { triples }`
    InsertData {
        /// Triples to insert.
        triples: Vec<TriplePattern>,
        /// Optional target graph IRI.
        graph: Option<String>,
    },
    /// `DELETE DATA { triples }`
    DeleteData {
        /// Triples to delete.
        triples: Vec<TriplePattern>,
        /// Optional target graph IRI.
        graph: Option<String>,
    },
    /// `DELETE { del } INSERT { ins } WHERE { pattern }`
    DeleteInsertWhere {
        delete_triples: Vec<TriplePattern>,
        insert_triples: Vec<TriplePattern>,
        where_triples: Vec<TriplePattern>,
        graph: Option<String>,
    },
    /// `LOAD <iri> [INTO GRAPH <target>]`
    Load {
        source_uri: String,
        target_graph: Option<String>,
        silent: bool,
    },
    /// `CLEAR target`
    Clear { target: GraphTarget, silent: bool },
    /// `DROP target`
    Drop { target: GraphTarget, silent: bool },
    /// `CREATE GRAPH <iri>`
    CreateGraph { graph_iri: String, silent: bool },
    /// `COPY source TO target`
    Copy {
        source: GraphTarget,
        destination: GraphTarget,
        silent: bool,
    },
    /// `MOVE source TO target`
    Move {
        source: GraphTarget,
        destination: GraphTarget,
        silent: bool,
    },
    /// `ADD source TO target`
    Add {
        source: GraphTarget,
        destination: GraphTarget,
        silent: bool,
    },
}

impl UpdateOperation {
    /// A short human-readable label for the operation kind.
    pub fn kind_label(&self) -> &'static str {
        match self {
            UpdateOperation::InsertData { .. } => "INSERT DATA",
            UpdateOperation::DeleteData { .. } => "DELETE DATA",
            UpdateOperation::DeleteInsertWhere { .. } => "DELETE/INSERT WHERE",
            UpdateOperation::Load { .. } => "LOAD",
            UpdateOperation::Clear { .. } => "CLEAR",
            UpdateOperation::Drop { .. } => "DROP",
            UpdateOperation::CreateGraph { .. } => "CREATE GRAPH",
            UpdateOperation::Copy { .. } => "COPY",
            UpdateOperation::Move { .. } => "MOVE",
            UpdateOperation::Add { .. } => "ADD",
        }
    }
}

/// A fully parsed SPARQL Update request (one or more operations separated by `;`).
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateRequest {
    /// PREFIX declarations.
    pub prefixes: HashMap<String, String>,
    /// Ordered list of update operations.
    pub operations: Vec<UpdateOperation>,
}

// ────────────────────────────────────────────────────────────────────────────
// Parser
// ────────────────────────────────────────────────────────────────────────────

/// SPARQL UPDATE parser.
pub struct UpdateParser {
    /// Namespace prefix map built from PREFIX declarations.
    prefixes: HashMap<String, String>,
}

impl Default for UpdateParser {
    fn default() -> Self {
        Self::new()
    }
}

impl UpdateParser {
    /// Create a new parser with an empty prefix map.
    pub fn new() -> Self {
        Self {
            prefixes: HashMap::new(),
        }
    }

    /// Create a parser pre-loaded with prefix definitions.
    pub fn with_prefixes(prefixes: HashMap<String, String>) -> Self {
        Self { prefixes }
    }

    /// Returns the parser's current namespace prefix map.
    pub fn prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }

    /// Parse a full SPARQL Update request string.
    pub fn parse(&mut self, input: &str) -> Result<UpdateRequest, UpdateParseError> {
        let mut pos = 0usize;
        let mut operations = Vec::new();

        // Parse prologue (BASE / PREFIX declarations).
        pos = self.skip_ws(input, pos);
        pos = self.parse_prologue(input, pos)?;

        // Parse one or more update operations separated by `;`.
        loop {
            pos = self.skip_ws(input, pos);
            if pos >= input.len() {
                break;
            }

            let op = self.parse_operation(input, &mut pos)?;
            operations.push(op);

            pos = self.skip_ws(input, pos);
            if pos < input.len() && input.as_bytes().get(pos) == Some(&b';') {
                pos += 1; // consume separator
            }
        }

        if operations.is_empty() {
            let (ln, col) = line_col(input, pos);
            return Err(UpdateParseError::new("empty update request", pos).with_location(ln, col));
        }

        Ok(UpdateRequest {
            prefixes: self.prefixes.clone(),
            operations,
        })
    }

    // ── prologue ────────────────────────────────────────────────────────────

    fn parse_prologue(&mut self, input: &str, mut pos: usize) -> Result<usize, UpdateParseError> {
        loop {
            pos = self.skip_ws(input, pos);
            if self.match_keyword(input, pos, "PREFIX") {
                pos = self.consume_keyword(input, pos, "PREFIX")?;
                pos = self.skip_ws(input, pos);

                let (prefix, new_pos) = self.read_prefix_label(input, pos)?;
                pos = self.skip_ws(input, new_pos);

                let (iri, new_pos) = self.read_iri_ref(input, pos)?;
                pos = new_pos;

                self.prefixes.insert(prefix, iri);
            } else if self.match_keyword(input, pos, "BASE") {
                pos = self.consume_keyword(input, pos, "BASE")?;
                pos = self.skip_ws(input, pos);
                // Read the IRI but we don't use BASE resolution in this simple parser.
                let (_, new_pos) = self.read_iri_ref(input, pos)?;
                pos = new_pos;
            } else {
                break;
            }
        }
        Ok(pos)
    }

    // ── operation dispatch ──────────────────────────────────────────────────

    fn parse_operation(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.skip_ws(input, *pos);
        if *pos >= input.len() {
            let (ln, col) = line_col(input, *pos);
            return Err(
                UpdateParseError::new("unexpected end of input", *pos).with_location(ln, col)
            );
        }

        // Determine which operation keyword is at the current position.
        if self.match_keyword(input, *pos, "INSERT") {
            self.parse_insert(input, pos)
        } else if self.match_keyword(input, *pos, "DELETE") {
            self.parse_delete(input, pos)
        } else if self.match_keyword(input, *pos, "LOAD") {
            self.parse_load(input, pos)
        } else if self.match_keyword(input, *pos, "CLEAR") {
            self.parse_clear(input, pos)
        } else if self.match_keyword(input, *pos, "DROP") {
            self.parse_drop(input, pos)
        } else if self.match_keyword(input, *pos, "CREATE") {
            self.parse_create(input, pos)
        } else if self.match_keyword(input, *pos, "COPY") {
            self.parse_copy(input, pos)
        } else if self.match_keyword(input, *pos, "MOVE") {
            self.parse_move(input, pos)
        } else if self.match_keyword(input, *pos, "ADD") {
            self.parse_add(input, pos)
        } else {
            let (ln, col) = line_col(input, *pos);
            let snippet: String = input[*pos..].chars().take(20).collect();
            Err(UpdateParseError::new(
                format!("expected update keyword, found: '{}'", snippet),
                *pos,
            )
            .with_location(ln, col))
        }
    }

    // ── INSERT ──────────────────────────────────────────────────────────────

    fn parse_insert(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.consume_keyword(input, *pos, "INSERT")?;
        *pos = self.skip_ws(input, *pos);

        if self.match_keyword(input, *pos, "DATA") {
            *pos = self.consume_keyword(input, *pos, "DATA")?;
            *pos = self.skip_ws(input, *pos);

            let (graph, triples) = self.parse_quad_data(input, pos)?;
            Ok(UpdateOperation::InsertData { triples, graph })
        } else {
            // INSERT { ... } WHERE { ... }   (only insert part of delete/insert)
            let (ln, col) = line_col(input, *pos);
            Err(UpdateParseError::new(
                "expected DATA after INSERT (standalone INSERT without DELETE is not supported here; use DELETE/INSERT WHERE)",
                *pos,
            )
            .with_location(ln, col))
        }
    }

    // ── DELETE ──────────────────────────────────────────────────────────────

    fn parse_delete(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.consume_keyword(input, *pos, "DELETE")?;
        *pos = self.skip_ws(input, *pos);

        if self.match_keyword(input, *pos, "DATA") {
            *pos = self.consume_keyword(input, *pos, "DATA")?;
            *pos = self.skip_ws(input, *pos);

            let (graph, triples) = self.parse_quad_data(input, pos)?;
            Ok(UpdateOperation::DeleteData { triples, graph })
        } else if *pos < input.len() && input.as_bytes().get(*pos) == Some(&b'{') {
            // DELETE { ... } INSERT { ... } WHERE { ... }
            let delete_triples = self.parse_brace_block(input, pos)?;
            *pos = self.skip_ws(input, *pos);

            let mut insert_triples = Vec::new();
            if self.match_keyword(input, *pos, "INSERT") {
                *pos = self.consume_keyword(input, *pos, "INSERT")?;
                *pos = self.skip_ws(input, *pos);
                insert_triples = self.parse_brace_block(input, pos)?;
                *pos = self.skip_ws(input, *pos);
            }

            *pos = self.consume_keyword(input, *pos, "WHERE")?;
            *pos = self.skip_ws(input, *pos);
            let where_triples = self.parse_brace_block(input, pos)?;

            Ok(UpdateOperation::DeleteInsertWhere {
                delete_triples,
                insert_triples,
                where_triples,
                graph: None,
            })
        } else {
            let (ln, col) = line_col(input, *pos);
            Err(
                UpdateParseError::new("expected DATA or '{' after DELETE", *pos)
                    .with_location(ln, col),
            )
        }
    }

    // ── LOAD ────────────────────────────────────────────────────────────────

    fn parse_load(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.consume_keyword(input, *pos, "LOAD")?;
        *pos = self.skip_ws(input, *pos);

        let silent = self.try_consume_keyword(input, pos, "SILENT");
        *pos = self.skip_ws(input, *pos);

        let (source_uri, new_pos) = self.read_iri_ref(input, *pos)?;
        *pos = new_pos;
        *pos = self.skip_ws(input, *pos);

        let target_graph = if self.match_keyword(input, *pos, "INTO") {
            *pos = self.consume_keyword(input, *pos, "INTO")?;
            *pos = self.skip_ws(input, *pos);
            *pos = self.consume_keyword(input, *pos, "GRAPH")?;
            *pos = self.skip_ws(input, *pos);
            let (iri, new_pos) = self.read_iri_ref(input, *pos)?;
            *pos = new_pos;
            Some(iri)
        } else {
            None
        };

        Ok(UpdateOperation::Load {
            source_uri,
            target_graph,
            silent,
        })
    }

    // ── CLEAR / DROP ────────────────────────────────────────────────────────

    fn parse_clear(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.consume_keyword(input, *pos, "CLEAR")?;
        *pos = self.skip_ws(input, *pos);
        let silent = self.try_consume_keyword(input, pos, "SILENT");
        *pos = self.skip_ws(input, *pos);
        let target = self.parse_graph_ref_all(input, pos)?;
        Ok(UpdateOperation::Clear { target, silent })
    }

    fn parse_drop(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.consume_keyword(input, *pos, "DROP")?;
        *pos = self.skip_ws(input, *pos);
        let silent = self.try_consume_keyword(input, pos, "SILENT");
        *pos = self.skip_ws(input, *pos);
        let target = self.parse_graph_ref_all(input, pos)?;
        Ok(UpdateOperation::Drop { target, silent })
    }

    // ── CREATE ──────────────────────────────────────────────────────────────

    fn parse_create(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.consume_keyword(input, *pos, "CREATE")?;
        *pos = self.skip_ws(input, *pos);
        let silent = self.try_consume_keyword(input, pos, "SILENT");
        *pos = self.skip_ws(input, *pos);
        *pos = self.consume_keyword(input, *pos, "GRAPH")?;
        *pos = self.skip_ws(input, *pos);
        let (iri, new_pos) = self.read_iri_ref(input, *pos)?;
        *pos = new_pos;
        Ok(UpdateOperation::CreateGraph {
            graph_iri: iri,
            silent,
        })
    }

    // ── COPY / MOVE / ADD ───────────────────────────────────────────────────

    fn parse_copy(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.consume_keyword(input, *pos, "COPY")?;
        *pos = self.skip_ws(input, *pos);
        let silent = self.try_consume_keyword(input, pos, "SILENT");
        *pos = self.skip_ws(input, *pos);
        let source = self.parse_graph_or_default(input, pos)?;
        *pos = self.skip_ws(input, *pos);
        *pos = self.consume_keyword(input, *pos, "TO")?;
        *pos = self.skip_ws(input, *pos);
        let destination = self.parse_graph_or_default(input, pos)?;
        Ok(UpdateOperation::Copy {
            source,
            destination,
            silent,
        })
    }

    fn parse_move(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.consume_keyword(input, *pos, "MOVE")?;
        *pos = self.skip_ws(input, *pos);
        let silent = self.try_consume_keyword(input, pos, "SILENT");
        *pos = self.skip_ws(input, *pos);
        let source = self.parse_graph_or_default(input, pos)?;
        *pos = self.skip_ws(input, *pos);
        *pos = self.consume_keyword(input, *pos, "TO")?;
        *pos = self.skip_ws(input, *pos);
        let destination = self.parse_graph_or_default(input, pos)?;
        Ok(UpdateOperation::Move {
            source,
            destination,
            silent,
        })
    }

    fn parse_add(&self, input: &str, pos: &mut usize) -> Result<UpdateOperation, UpdateParseError> {
        *pos = self.consume_keyword(input, *pos, "ADD")?;
        *pos = self.skip_ws(input, *pos);
        let silent = self.try_consume_keyword(input, pos, "SILENT");
        *pos = self.skip_ws(input, *pos);
        let source = self.parse_graph_or_default(input, pos)?;
        *pos = self.skip_ws(input, *pos);
        *pos = self.consume_keyword(input, *pos, "TO")?;
        *pos = self.skip_ws(input, *pos);
        let destination = self.parse_graph_or_default(input, pos)?;
        Ok(UpdateOperation::Add {
            source,
            destination,
            silent,
        })
    }

    // ── graph references ────────────────────────────────────────────────────

    fn parse_graph_ref_all(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<GraphTarget, UpdateParseError> {
        *pos = self.skip_ws(input, *pos);
        if self.match_keyword(input, *pos, "ALL") {
            *pos = self.consume_keyword(input, *pos, "ALL")?;
            Ok(GraphTarget::All)
        } else if self.match_keyword(input, *pos, "DEFAULT") {
            *pos = self.consume_keyword(input, *pos, "DEFAULT")?;
            Ok(GraphTarget::Default)
        } else if self.match_keyword(input, *pos, "NAMED") {
            *pos = self.consume_keyword(input, *pos, "NAMED")?;
            Ok(GraphTarget::Named)
        } else if self.match_keyword(input, *pos, "GRAPH") {
            *pos = self.consume_keyword(input, *pos, "GRAPH")?;
            *pos = self.skip_ws(input, *pos);
            let (iri, new_pos) = self.read_iri_ref(input, *pos)?;
            *pos = new_pos;
            Ok(GraphTarget::Graph(iri))
        } else {
            let (ln, col) = line_col(input, *pos);
            Err(
                UpdateParseError::new("expected GRAPH, DEFAULT, NAMED, or ALL", *pos)
                    .with_location(ln, col),
            )
        }
    }

    fn parse_graph_or_default(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<GraphTarget, UpdateParseError> {
        *pos = self.skip_ws(input, *pos);
        if self.match_keyword(input, *pos, "DEFAULT") {
            *pos = self.consume_keyword(input, *pos, "DEFAULT")?;
            Ok(GraphTarget::Default)
        } else if self.match_keyword(input, *pos, "GRAPH") {
            *pos = self.consume_keyword(input, *pos, "GRAPH")?;
            *pos = self.skip_ws(input, *pos);
            let (iri, new_pos) = self.read_iri_ref(input, *pos)?;
            *pos = new_pos;
            Ok(GraphTarget::Graph(iri))
        } else {
            // Try to read a bare IRI as GRAPH <iri>
            if *pos < input.len() && input.as_bytes().get(*pos) == Some(&b'<') {
                let (iri, new_pos) = self.read_iri_ref(input, *pos)?;
                *pos = new_pos;
                Ok(GraphTarget::Graph(iri))
            } else {
                let (ln, col) = line_col(input, *pos);
                Err(
                    UpdateParseError::new("expected DEFAULT, GRAPH <iri>, or <iri>", *pos)
                        .with_location(ln, col),
                )
            }
        }
    }

    // ── quad / triple data blocks ───────────────────────────────────────────

    /// Parse QuadData per SPARQL 1.1 Update grammar:
    ///   QuadData ::= '{' Quads '}'
    ///   Quads    ::= TriplesTemplate? ( 'GRAPH' VarOrIri '{' TriplesTemplate? '}' '.'? TriplesTemplate? )*
    ///
    /// This handles both `{ triples }` (default graph) and
    /// `{ GRAPH <iri> { triples } }` (named graph).
    fn parse_quad_data(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<(Option<String>, Vec<TriplePattern>), UpdateParseError> {
        *pos = self.skip_ws(input, *pos);

        // Expect the outer opening brace of QuadData.
        if *pos >= input.len() || input.as_bytes().get(*pos) != Some(&b'{') {
            let (ln, col) = line_col(input, *pos);
            return Err(
                UpdateParseError::new("expected '{' to open quad data block", *pos)
                    .with_location(ln, col),
            );
        }
        *pos += 1; // consume outer '{'

        *pos = self.skip_ws(input, *pos);

        // Check whether the content starts with GRAPH keyword (named graph quad).
        if self.match_keyword(input, *pos, "GRAPH") {
            *pos = self.consume_keyword(input, *pos, "GRAPH")?;
            *pos = self.skip_ws(input, *pos);
            let (iri, new_pos) = self.read_iri_ref(input, *pos)?;
            *pos = new_pos;
            *pos = self.skip_ws(input, *pos);

            // Parse the inner brace block `{ triples }`.
            let triples = self.parse_brace_block(input, pos)?;

            // Consume optional trailing '.' after the GRAPH block.
            *pos = self.skip_ws(input, *pos);
            if *pos < input.len() && input.as_bytes().get(*pos) == Some(&b'.') {
                *pos += 1;
            }

            // Consume outer closing brace.
            *pos = self.skip_ws(input, *pos);
            if *pos >= input.len() || input.as_bytes().get(*pos) != Some(&b'}') {
                let (ln, col) = line_col(input, *pos);
                return Err(
                    UpdateParseError::new("expected '}' to close quad data block", *pos)
                        .with_location(ln, col),
                );
            }
            *pos += 1;

            Ok((Some(iri), triples))
        } else {
            // Default graph triples: parse triples until the outer '}'.
            let mut triples = Vec::new();
            loop {
                *pos = self.skip_ws(input, *pos);
                if *pos >= input.len() {
                    let (ln, col) = line_col(input, *pos);
                    return Err(UpdateParseError::new(
                        "unexpected end of input, expected '}'",
                        *pos,
                    )
                    .with_location(ln, col));
                }
                if input.as_bytes().get(*pos) == Some(&b'}') {
                    *pos += 1;
                    break;
                }

                let triple = self.parse_triple_pattern(input, pos)?;
                triples.push(triple);

                *pos = self.skip_ws(input, *pos);
                // Consume optional '.'
                if *pos < input.len() && input.as_bytes().get(*pos) == Some(&b'.') {
                    *pos += 1;
                }
            }

            Ok((None, triples))
        }
    }

    /// Parse `{ triple1 . triple2 . ... }`.
    fn parse_brace_block(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<Vec<TriplePattern>, UpdateParseError> {
        *pos = self.skip_ws(input, *pos);
        if *pos >= input.len() || input.as_bytes().get(*pos) != Some(&b'{') {
            let (ln, col) = line_col(input, *pos);
            return Err(UpdateParseError::new("expected '{'", *pos).with_location(ln, col));
        }
        *pos += 1; // consume '{'

        let mut triples = Vec::new();
        loop {
            *pos = self.skip_ws(input, *pos);
            if *pos >= input.len() {
                let (ln, col) = line_col(input, *pos);
                return Err(
                    UpdateParseError::new("unexpected end of input, expected '}'", *pos)
                        .with_location(ln, col),
                );
            }
            if input.as_bytes().get(*pos) == Some(&b'}') {
                *pos += 1;
                break;
            }

            let triple = self.parse_triple_pattern(input, pos)?;
            triples.push(triple);

            *pos = self.skip_ws(input, *pos);
            // Consume optional '.'
            if *pos < input.len() && input.as_bytes().get(*pos) == Some(&b'.') {
                *pos += 1;
            }
        }

        Ok(triples)
    }

    /// Parse a single triple pattern: subject predicate object.
    fn parse_triple_pattern(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<TriplePattern, UpdateParseError> {
        *pos = self.skip_ws(input, *pos);
        let subject = self.read_term(input, pos)?;
        *pos = self.skip_ws(input, *pos);
        let predicate = self.read_term(input, pos)?;
        *pos = self.skip_ws(input, *pos);
        let object = self.read_term(input, pos)?;
        Ok(TriplePattern::new(subject, predicate, object))
    }

    // ── low-level token readers ─────────────────────────────────────────────

    /// Read a single term (IRI, prefixed name, variable, literal, blank node, or 'a').
    fn read_term(&self, input: &str, pos: &mut usize) -> Result<String, UpdateParseError> {
        *pos = self.skip_ws(input, *pos);
        if *pos >= input.len() {
            let (ln, col) = line_col(input, *pos);
            return Err(
                UpdateParseError::new("unexpected end of input while reading term", *pos)
                    .with_location(ln, col),
            );
        }

        let ch = input.as_bytes()[*pos];

        // IRI reference
        if ch == b'<' {
            let (iri, new_pos) = self.read_iri_ref(input, *pos)?;
            *pos = new_pos;
            return Ok(format!("<{}>", iri));
        }

        // Variable
        if ch == b'?' || ch == b'$' {
            let start = *pos;
            *pos += 1; // skip ? or $
            while *pos < input.len() && is_name_char(input.as_bytes()[*pos]) {
                *pos += 1;
            }
            return Ok(input[start..*pos].to_string());
        }

        // Literal
        if ch == b'"' || ch == b'\'' {
            return self.read_literal(input, pos);
        }

        // Blank node
        if ch == b'_' && input.as_bytes().get(*pos + 1) == Some(&b':') {
            let start = *pos;
            *pos += 2;
            while *pos < input.len() && is_name_char(input.as_bytes()[*pos]) {
                *pos += 1;
            }
            return Ok(input[start..*pos].to_string());
        }

        // Keyword 'a' (rdf:type shorthand)
        if ch == b'a'
            && (*pos + 1 >= input.len()
                || !is_name_char(input.as_bytes().get(*pos + 1).copied().unwrap_or(b' ')))
        {
            *pos += 1;
            return Ok("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>".to_string());
        }

        // Prefixed name (prefix:local)
        let start = *pos;
        while *pos < input.len() && is_name_char(input.as_bytes()[*pos]) {
            *pos += 1;
        }
        if *pos < input.len() && input.as_bytes().get(*pos) == Some(&b':') {
            let prefix = &input[start..*pos];
            *pos += 1; // skip ':'
            let local_start = *pos;
            while *pos < input.len() && is_pname_local_char(input.as_bytes()[*pos]) {
                *pos += 1;
            }
            let local = &input[local_start..*pos];

            // Expand prefix if known
            if let Some(ns) = self.prefixes.get(prefix) {
                return Ok(format!("<{}{}>", ns, local));
            }
            return Ok(format!("{}:{}", prefix, local));
        }

        // Numeric literal
        *pos = start; // reset
        if ch.is_ascii_digit() || ch == b'+' || ch == b'-' {
            return self.read_numeric_literal(input, pos);
        }

        // Boolean literals
        if self.match_keyword(input, *pos, "true") {
            *pos += 4;
            return Ok("\"true\"^^<http://www.w3.org/2001/XMLSchema#boolean>".to_string());
        }
        if self.match_keyword(input, *pos, "false") {
            *pos += 5;
            return Ok("\"false\"^^<http://www.w3.org/2001/XMLSchema#boolean>".to_string());
        }

        let (ln, col) = line_col(input, *pos);
        let snippet: String = input[*pos..].chars().take(20).collect();
        Err(
            UpdateParseError::new(format!("unexpected token: '{}'", snippet), *pos)
                .with_location(ln, col),
        )
    }

    /// Read a string literal (supports double-quoted and single-quoted).
    fn read_literal(&self, input: &str, pos: &mut usize) -> Result<String, UpdateParseError> {
        let quote = input.as_bytes()[*pos];
        let start = *pos;
        *pos += 1; // skip opening quote
        let mut value = String::new();

        while *pos < input.len() {
            let ch = input.as_bytes()[*pos];
            if ch == b'\\' && *pos + 1 < input.len() {
                let esc = input.as_bytes()[*pos + 1];
                let escaped = match esc {
                    b'n' => '\n',
                    b't' => '\t',
                    b'\\' => '\\',
                    b'"' => '"',
                    b'\'' => '\'',
                    _ => {
                        *pos += 2;
                        continue;
                    }
                };
                value.push(escaped);
                *pos += 2;
            } else if ch == quote {
                *pos += 1; // skip closing quote
                           // Check for ^^<datatype> or @lang
                if *pos < input.len()
                    && input.as_bytes().get(*pos) == Some(&b'^')
                    && input.as_bytes().get(*pos + 1) == Some(&b'^')
                {
                    *pos += 2;
                    if input.as_bytes().get(*pos) == Some(&b'<') {
                        let (dt, new_pos) = self.read_iri_ref(input, *pos)?;
                        *pos = new_pos;
                        return Ok(format!("\"{}\"^^<{}>", value, dt));
                    }
                }
                if *pos < input.len() && input.as_bytes().get(*pos) == Some(&b'@') {
                    *pos += 1;
                    let lang_start = *pos;
                    while *pos < input.len()
                        && (input.as_bytes()[*pos].is_ascii_alphanumeric()
                            || input.as_bytes()[*pos] == b'-')
                    {
                        *pos += 1;
                    }
                    let lang = &input[lang_start..*pos];
                    return Ok(format!("\"{}\"@{}", value, lang));
                }
                return Ok(format!("\"{}\"", value));
            } else {
                value.push(ch as char);
                *pos += 1;
            }
        }

        let (ln, col) = line_col(input, start);
        Err(UpdateParseError::new("unterminated string literal", start).with_location(ln, col))
    }

    /// Read a numeric literal (integer or decimal).
    fn read_numeric_literal(
        &self,
        input: &str,
        pos: &mut usize,
    ) -> Result<String, UpdateParseError> {
        let start = *pos;
        // Optional sign
        if *pos < input.len() && (input.as_bytes()[*pos] == b'+' || input.as_bytes()[*pos] == b'-')
        {
            *pos += 1;
        }
        // Digits
        while *pos < input.len() && input.as_bytes()[*pos].is_ascii_digit() {
            *pos += 1;
        }
        // Optional decimal part
        let mut is_decimal = false;
        if *pos < input.len() && input.as_bytes().get(*pos) == Some(&b'.') {
            is_decimal = true;
            *pos += 1;
            while *pos < input.len() && input.as_bytes()[*pos].is_ascii_digit() {
                *pos += 1;
            }
        }
        // Optional exponent
        if *pos < input.len() && (input.as_bytes()[*pos] == b'e' || input.as_bytes()[*pos] == b'E')
        {
            is_decimal = true;
            *pos += 1;
            if *pos < input.len()
                && (input.as_bytes()[*pos] == b'+' || input.as_bytes()[*pos] == b'-')
            {
                *pos += 1;
            }
            while *pos < input.len() && input.as_bytes()[*pos].is_ascii_digit() {
                *pos += 1;
            }
        }

        if *pos == start {
            let (ln, col) = line_col(input, *pos);
            return Err(
                UpdateParseError::new("expected numeric literal", *pos).with_location(ln, col)
            );
        }

        let text = &input[start..*pos];
        if is_decimal {
            Ok(format!(
                "\"{}\"^^<http://www.w3.org/2001/XMLSchema#double>",
                text
            ))
        } else {
            Ok(format!(
                "\"{}\"^^<http://www.w3.org/2001/XMLSchema#integer>",
                text
            ))
        }
    }

    /// Read an IRI enclosed in `< >`. Returns the IRI without angle brackets.
    fn read_iri_ref(&self, input: &str, pos: usize) -> Result<(String, usize), UpdateParseError> {
        if pos >= input.len() || input.as_bytes().get(pos) != Some(&b'<') {
            let (ln, col) = line_col(input, pos);
            return Err(
                UpdateParseError::new("expected '<' to start IRI reference", pos)
                    .with_location(ln, col),
            );
        }
        let start = pos + 1;
        let mut end = start;
        while end < input.len() && input.as_bytes()[end] != b'>' {
            end += 1;
        }
        if end >= input.len() {
            let (ln, col) = line_col(input, pos);
            return Err(
                UpdateParseError::new("unterminated IRI reference", pos).with_location(ln, col)
            );
        }
        let iri = input[start..end].to_string();
        Ok((iri, end + 1))
    }

    /// Read a prefix label (e.g., `ex:` or `:`).
    fn read_prefix_label(
        &self,
        input: &str,
        pos: usize,
    ) -> Result<(String, usize), UpdateParseError> {
        let start = pos;
        let mut p = pos;
        while p < input.len() && input.as_bytes()[p] != b':' {
            p += 1;
        }
        if p >= input.len() {
            let (ln, col) = line_col(input, pos);
            return Err(
                UpdateParseError::new("expected ':' in prefix declaration", pos)
                    .with_location(ln, col),
            );
        }
        let prefix = input[start..p].trim().to_string();
        Ok((prefix, p + 1)) // skip ':'
    }

    // ── utility helpers ─────────────────────────────────────────────────────

    fn skip_ws(&self, input: &str, mut pos: usize) -> usize {
        let bytes = input.as_bytes();
        while pos < bytes.len() {
            if bytes[pos].is_ascii_whitespace() {
                pos += 1;
            } else if bytes[pos] == b'#' {
                // Skip line comment
                while pos < bytes.len() && bytes[pos] != b'\n' {
                    pos += 1;
                }
            } else {
                break;
            }
        }
        pos
    }

    fn match_keyword(&self, input: &str, pos: usize, kw: &str) -> bool {
        let end = pos + kw.len();
        if end > input.len() {
            return false;
        }
        if !input[pos..end].eq_ignore_ascii_case(kw) {
            return false;
        }
        // Ensure keyword boundary
        end >= input.len() || !is_name_char(input.as_bytes()[end])
    }

    fn consume_keyword(
        &self,
        input: &str,
        pos: usize,
        kw: &str,
    ) -> Result<usize, UpdateParseError> {
        if !self.match_keyword(input, pos, kw) {
            let (ln, col) = line_col(input, pos);
            let snippet: String = input[pos..].chars().take(20).collect();
            return Err(UpdateParseError::new(
                format!("expected keyword '{}', found: '{}'", kw, snippet),
                pos,
            )
            .with_location(ln, col));
        }
        Ok(pos + kw.len())
    }

    /// Try to consume an optional keyword. Returns `true` if found.
    fn try_consume_keyword(&self, input: &str, pos: &mut usize, kw: &str) -> bool {
        if self.match_keyword(input, *pos, kw) {
            *pos += kw.len();
            true
        } else {
            false
        }
    }
}

// ── character class helpers ─────────────────────────────────────────────────

fn is_name_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

fn is_pname_local_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
}

// ────────────────────────────────────────────────────────────────────────────
// Convenience functions
// ────────────────────────────────────────────────────────────────────────────

/// Parse a SPARQL Update request from a string.
pub fn parse_update(input: &str) -> Result<UpdateRequest, UpdateParseError> {
    let mut parser = UpdateParser::new();
    parser.parse(input)
}

/// Parse a SPARQL Update request with pre-defined prefixes.
pub fn parse_update_with_prefixes(
    input: &str,
    prefixes: HashMap<String, String>,
) -> Result<UpdateRequest, UpdateParseError> {
    let mut parser = UpdateParser::with_prefixes(prefixes);
    parser.parse(input)
}
