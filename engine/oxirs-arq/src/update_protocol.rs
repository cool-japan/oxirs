//! SPARQL 1.1 Update Protocol — standalone parser and in-memory executor
//!
//! This module provides a self-contained representation of all SPARQL 1.1
//! graph-update operations together with a text parser and an in-memory
//! store-backed executor.  It is intentionally independent of the
//! store-coupled `UpdateExecutor` in `update.rs` so that it can be used in
//! contexts where only an in-memory triple set is required (e.g. unit tests
//! and HTTP protocol adapters).
//!
//! ## Supported operations
//!
//! | SPARQL keyword | Variant |
//! |---|---|
//! | `INSERT DATA` | `SparqlUpdate::InsertData` |
//! | `DELETE DATA` | `SparqlUpdate::DeleteData` |
//! | `INSERT { … } WHERE { … }` | `SparqlUpdate::InsertWhere` |
//! | `DELETE { … } WHERE { … }` | `SparqlUpdate::DeleteWhere` |
//! | `DELETE { … } INSERT { … } WHERE { … }` | `SparqlUpdate::Modify` |
//! | `CREATE [SILENT] GRAPH <…>` | `SparqlUpdate::CreateGraph` |
//! | `DROP [SILENT] [GRAPH <…> \| DEFAULT \| NAMED \| ALL]` | `SparqlUpdate::DropGraph` |
//! | `CLEAR [SILENT] [GRAPH <…> \| DEFAULT \| NAMED \| ALL]` | `SparqlUpdate::ClearGraph` |
//! | `COPY [SILENT] <…> TO <…>` | `SparqlUpdate::CopyGraph` |
//! | `MOVE [SILENT] <…> TO <…>` | `SparqlUpdate::MoveGraph` |
//! | `ADD [SILENT] <…> TO <…>` | `SparqlUpdate::AddGraph` |
//! | `LOAD [SILENT] <…> [INTO GRAPH <…>]` | `SparqlUpdate::Load` |

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// A concrete RDF triple (no variables).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub s: String,
    pub p: String,
    pub o: String,
}

impl Triple {
    /// Convenience constructor.
    pub fn new(s: impl Into<String>, p: impl Into<String>, o: impl Into<String>) -> Self {
        Self {
            s: s.into(),
            p: p.into(),
            o: o.into(),
        }
    }
}

/// A position in a triple pattern – can be an IRI, a plain literal, a blank
/// node, or a variable (placeholder for pattern matching).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PatternTerm {
    Iri(String),
    Literal(String),
    Variable(String),
    BlankNode(String),
}

impl PatternTerm {
    /// Returns `true` when this term is a variable (used during template instantiation).
    pub fn is_variable(&self) -> bool {
        matches!(self, PatternTerm::Variable(_))
    }

    /// Returns the variable name if this is a `Variable` variant.
    pub fn variable_name(&self) -> Option<&str> {
        if let PatternTerm::Variable(name) = self {
            Some(name.as_str())
        } else {
            None
        }
    }
}

/// A triple pattern where any position may be a variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriplePattern {
    pub s: PatternTerm,
    pub p: PatternTerm,
    pub o: PatternTerm,
}

impl TriplePattern {
    /// Construct a new triple pattern.
    pub fn new(s: PatternTerm, p: PatternTerm, o: PatternTerm) -> Self {
        Self { s, o, p }
    }
}

// ---------------------------------------------------------------------------
// DROP / CLEAR target type
// ---------------------------------------------------------------------------

/// Scope qualifier for `DROP` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropType {
    /// A specific named graph identified by IRI.
    Graph,
    /// The default graph.
    Default,
    /// All named graphs.
    Named,
    /// Every graph in the dataset (default + all named).
    All,
}

/// Scope qualifier for `CLEAR` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClearType {
    /// A specific named graph identified by IRI.
    Graph,
    /// The default graph.
    Default,
    /// All named graphs.
    Named,
    /// Every graph in the dataset.
    All,
}

// ---------------------------------------------------------------------------
// Top-level update enum
// ---------------------------------------------------------------------------

/// A single SPARQL 1.1 update operation.
#[derive(Debug, Clone, PartialEq)]
pub enum SparqlUpdate {
    /// `INSERT DATA { … }` — adds concrete triples to the default graph.
    InsertData(Vec<Triple>),

    /// `DELETE DATA { … }` — removes concrete triples from the default graph.
    DeleteData(Vec<Triple>),

    /// `INSERT { template } WHERE { where_clause }` — pattern-based insert.
    InsertWhere {
        template: Vec<TriplePattern>,
        where_clause: Vec<TriplePattern>,
    },

    /// `DELETE { template } WHERE { where_clause }` — pattern-based delete.
    DeleteWhere {
        template: Vec<TriplePattern>,
        where_clause: Vec<TriplePattern>,
    },

    /// `DELETE { delete } INSERT { insert } WHERE { where_clause }` — combined modify.
    Modify {
        delete: Vec<TriplePattern>,
        insert: Vec<TriplePattern>,
        where_clause: Vec<TriplePattern>,
    },

    /// `CREATE [SILENT] GRAPH <iri>`.
    CreateGraph { iri: String, silent: bool },

    /// `DROP [SILENT] (GRAPH <iri> | DEFAULT | NAMED | ALL)`.
    DropGraph {
        iri: Option<String>,
        silent: bool,
        drop_type: DropType,
    },

    /// `CLEAR [SILENT] (GRAPH <iri> | DEFAULT | NAMED | ALL)`.
    ClearGraph {
        iri: Option<String>,
        silent: bool,
        clear_type: ClearType,
    },

    /// `COPY [SILENT] <source> TO <target>`.
    CopyGraph {
        source: String,
        target: String,
        silent: bool,
    },

    /// `MOVE [SILENT] <source> TO <target>`.
    MoveGraph {
        source: String,
        target: String,
        silent: bool,
    },

    /// `ADD [SILENT] <source> TO <target>`.
    AddGraph {
        source: String,
        target: String,
        silent: bool,
    },

    /// `LOAD [SILENT] <iri> [INTO GRAPH <into>]`.
    Load {
        iri: String,
        into: Option<String>,
        silent: bool,
    },
}

// ---------------------------------------------------------------------------
// Parse error
// ---------------------------------------------------------------------------

/// Error returned by `SparqlUpdateParser`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    /// Byte offset inside the input string where the error was detected.
    pub position: usize,
}

impl ParseError {
    fn at(position: usize, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            position,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "parse error at position {}: {}",
            self.position, self.message
        )
    }
}

impl std::error::Error for ParseError {}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// A lightweight tokenising parser for SPARQL 1.1 Update text.
///
/// The parser is intentionally simple (hand-written recursive descent on a
/// token stream) and covers the most common production patterns.  It does not
/// handle prefix declarations (`PREFIX`) or the `BASE` directive — callers
/// that require prefix resolution should expand prefixes before passing the
/// string to the parser.
pub struct SparqlUpdateParser;

impl SparqlUpdateParser {
    /// Parse zero or more semicolon-separated update operations from `input`.
    pub fn parse(input: &str) -> Result<Vec<SparqlUpdate>, ParseError> {
        let tokens = tokenise(input);
        let mut cursor = 0usize;
        let mut results = Vec::new();

        // Skip optional leading PREFIX declarations.
        skip_prefixes(&tokens, &mut cursor);

        while cursor < tokens.len() {
            skip_prefixes(&tokens, &mut cursor);
            if cursor >= tokens.len() {
                break;
            }
            let update = parse_one_operation(&tokens, &mut cursor)?;
            results.push(update);
            // Consume optional ';' separator.
            if cursor < tokens.len() && tokens[cursor] == ";" {
                cursor += 1;
            }
        }

        Ok(results)
    }

    /// Parse exactly one update operation from `input`.
    pub fn parse_one(input: &str) -> Result<SparqlUpdate, ParseError> {
        let mut updates = Self::parse(input)?;
        match updates.len() {
            0 => Err(ParseError::at(0, "no update operation found")),
            1 => Ok(updates.remove(0)),
            n => Err(ParseError::at(
                0,
                format!("expected exactly one operation, found {n}"),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Tokeniser
// ---------------------------------------------------------------------------

/// Produce a flat vector of tokens from a SPARQL update string.
///
/// Tokens are: keywords, IRIs (`<…>`), string literals (`"…"` / `'…'`),
/// blank node labels (`_:…`), variables (`?…` / `$…`), punctuation, and
/// bare identifiers.  Whitespace and `# comments` are discarded.
fn tokenise(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            // Skip whitespace
            c if c.is_whitespace() => i += 1,
            // Line comment
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            // IRI reference <…>
            '<' => {
                let mut tok = String::from('<');
                i += 1;
                while i < chars.len() && chars[i] != '>' {
                    tok.push(chars[i]);
                    i += 1;
                }
                if i < chars.len() {
                    tok.push('>');
                    i += 1;
                }
                tokens.push(tok);
            }
            // Double-quoted literal
            '"' => {
                let mut tok = String::from('"');
                i += 1;
                while i < chars.len() && chars[i] != '"' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        tok.push(chars[i]);
                        i += 1;
                    }
                    tok.push(chars[i]);
                    i += 1;
                }
                if i < chars.len() {
                    tok.push('"');
                    i += 1;
                }
                tokens.push(tok);
            }
            // Single-quoted literal
            '\'' => {
                let mut tok = String::from('\'');
                i += 1;
                while i < chars.len() && chars[i] != '\'' {
                    if chars[i] == '\\' && i + 1 < chars.len() {
                        tok.push(chars[i]);
                        i += 1;
                    }
                    tok.push(chars[i]);
                    i += 1;
                }
                if i < chars.len() {
                    tok.push('\'');
                    i += 1;
                }
                tokens.push(tok);
            }
            // Variable ?name or $name
            '?' | '$' => {
                let mut tok = String::from('?');
                i += 1;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    tok.push(chars[i]);
                    i += 1;
                }
                tokens.push(tok);
            }
            // Punctuation: { } ( ) . , ; ^^ @
            '{' | '}' | '(' | ')' | '.' | ',' | ';' => {
                tokens.push(chars[i].to_string());
                i += 1;
            }
            // ^^  datatype marker or ^ inverse path
            '^' => {
                if i + 1 < chars.len() && chars[i + 1] == '^' {
                    tokens.push("^^".to_string());
                    i += 2;
                } else {
                    tokens.push("^".to_string());
                    i += 1;
                }
            }
            // @ language tag
            '@' => {
                let mut tok = String::from('@');
                i += 1;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '-') {
                    tok.push(chars[i]);
                    i += 1;
                }
                tokens.push(tok);
            }
            // Bare identifier, keyword, prefixed name, or blank node label
            c if c.is_alphabetic() || c == '_' => {
                let mut tok = String::new();
                while i < chars.len()
                    && (chars[i].is_alphanumeric()
                        || chars[i] == '_'
                        || chars[i] == ':'
                        || chars[i] == '-')
                {
                    tok.push(chars[i]);
                    i += 1;
                }
                tokens.push(tok);
            }
            // Numbers / other characters — collect until whitespace or punctuation
            _ => {
                let mut tok = String::new();
                while i < chars.len()
                    && !chars[i].is_whitespace()
                    && !matches!(chars[i], '{' | '}' | '(' | ')' | ',' | ';')
                {
                    tok.push(chars[i]);
                    i += 1;
                }
                if !tok.is_empty() {
                    tokens.push(tok);
                }
            }
        }
    }

    tokens
}

// ---------------------------------------------------------------------------
// Parser helpers
// ---------------------------------------------------------------------------

fn peek(tokens: &[String], cursor: usize) -> Option<&str> {
    tokens.get(cursor).map(|s| s.as_str())
}

fn expect<'a>(
    tokens: &'a [String],
    cursor: &mut usize,
    expected: &str,
) -> Result<&'a str, ParseError> {
    match tokens.get(*cursor) {
        Some(tok) if tok.to_uppercase() == expected.to_uppercase() => {
            *cursor += 1;
            Ok(tok.as_str())
        }
        Some(tok) => Err(ParseError::at(
            *cursor,
            format!("expected '{expected}', found '{tok}'"),
        )),
        None => Err(ParseError::at(
            *cursor,
            format!("expected '{expected}', found end of input"),
        )),
    }
}

fn consume_keyword(tokens: &[String], cursor: &mut usize, keyword: &str) -> bool {
    match tokens.get(*cursor) {
        Some(tok) if tok.to_uppercase() == keyword.to_uppercase() => {
            *cursor += 1;
            true
        }
        _ => false,
    }
}

/// Consume the next token unconditionally and return it.
fn consume(tokens: &[String], cursor: &mut usize) -> Result<String, ParseError> {
    tokens
        .get(*cursor)
        .map(|t| {
            *cursor += 1;
            t.clone()
        })
        .ok_or_else(|| ParseError::at(*cursor, "unexpected end of input"))
}

/// Parse an IRI token of the form `<…>` and return the inner string.
fn parse_iri(tokens: &[String], cursor: &mut usize) -> Result<String, ParseError> {
    match tokens.get(*cursor) {
        Some(tok) if tok.starts_with('<') && tok.ends_with('>') => {
            let iri = tok[1..tok.len() - 1].to_string();
            *cursor += 1;
            Ok(iri)
        }
        Some(tok) => Err(ParseError::at(
            *cursor,
            format!("expected IRI, found '{tok}'"),
        )),
        None => Err(ParseError::at(*cursor, "expected IRI, found end of input")),
    }
}

/// Parse an optional `SILENT` keyword and return whether it was present.
fn parse_silent(tokens: &[String], cursor: &mut usize) -> bool {
    consume_keyword(tokens, cursor, "SILENT")
}

/// Skip zero or more `PREFIX` declarations.
fn skip_prefixes(tokens: &[String], cursor: &mut usize) {
    while let Some(tok) = tokens.get(*cursor) {
        if tok.to_uppercase() != "PREFIX" {
            break;
        }
        *cursor += 1; // consume PREFIX
                      // prefix name (e.g. "ex:" or ":")
        *cursor += 1;
        // IRI
        *cursor += 1;
    }
}

// ---------------------------------------------------------------------------
// Triple / pattern parsing
// ---------------------------------------------------------------------------

/// Parse a set of concrete triples inside `{ … }`.
/// Triples may be separated by `.` or `;` (simplified Turtle-like syntax).
fn parse_triple_block(tokens: &[String], cursor: &mut usize) -> Result<Vec<Triple>, ParseError> {
    expect(tokens, cursor, "{")?;
    let mut triples = Vec::new();

    while let Some(tok) = tokens.get(*cursor) {
        if tok == "}" {
            break;
        }
        if tok == "." {
            *cursor += 1;
            continue;
        }

        let s = parse_term_str(tokens, cursor)?;
        let p = parse_term_str(tokens, cursor)?;
        let o = parse_term_str(tokens, cursor)?;
        triples.push(Triple::new(s, p, o));

        // Optional trailing dot.
        if matches!(peek(tokens, *cursor), Some(".")) {
            *cursor += 1;
        }
    }

    expect(tokens, cursor, "}")?;
    Ok(triples)
}

/// Parse a set of triple patterns (may contain variables) inside `{ … }`.
fn parse_pattern_block(
    tokens: &[String],
    cursor: &mut usize,
) -> Result<Vec<TriplePattern>, ParseError> {
    expect(tokens, cursor, "{")?;
    let mut patterns = Vec::new();

    while let Some(tok) = tokens.get(*cursor) {
        if tok == "}" {
            break;
        }
        if tok == "." {
            *cursor += 1;
            continue;
        }

        let s = parse_pattern_term(tokens, cursor)?;
        let p = parse_pattern_term(tokens, cursor)?;
        let o = parse_pattern_term(tokens, cursor)?;
        patterns.push(TriplePattern::new(s, p, o));

        if matches!(peek(tokens, *cursor), Some(".")) {
            *cursor += 1;
        }
    }

    expect(tokens, cursor, "}")?;
    Ok(patterns)
}

/// Parse a single term as a plain string for use in concrete triples.
fn parse_term_str(tokens: &[String], cursor: &mut usize) -> Result<String, ParseError> {
    let tok = consume(tokens, cursor)?;
    // Unwrap IRI angles.
    if tok.starts_with('<') && tok.ends_with('>') {
        return Ok(tok[1..tok.len() - 1].to_string());
    }
    Ok(tok)
}

/// Parse a single term into a `PatternTerm`.
fn parse_pattern_term(tokens: &[String], cursor: &mut usize) -> Result<PatternTerm, ParseError> {
    let tok = consume(tokens, cursor)?;

    // Variable: ?name
    if let Some(stripped) = tok.strip_prefix('?') {
        return Ok(PatternTerm::Variable(stripped.to_string()));
    }

    // IRI: <…>
    if tok.starts_with('<') && tok.ends_with('>') {
        return Ok(PatternTerm::Iri(tok[1..tok.len() - 1].to_string()));
    }

    // Blank node: _:label
    if let Some(stripped) = tok.strip_prefix("_:") {
        return Ok(PatternTerm::BlankNode(stripped.to_string()));
    }

    // Literal: "…"
    if tok.starts_with('"') || tok.starts_with('\'') {
        // Consume optional @lang or ^^datatype.
        if matches!(peek(tokens, *cursor), Some(t) if t.starts_with('@')) {
            let _lang = consume(tokens, cursor)?;
        } else if matches!(peek(tokens, *cursor), Some("^^")) {
            *cursor += 1; // skip ^^
            let _dt = consume(tokens, cursor)?;
        }
        return Ok(PatternTerm::Literal(tok));
    }

    // Prefixed name or keyword treated as IRI-like.
    Ok(PatternTerm::Iri(tok))
}

// ---------------------------------------------------------------------------
// Top-level operation parser
// ---------------------------------------------------------------------------

fn parse_one_operation(tokens: &[String], cursor: &mut usize) -> Result<SparqlUpdate, ParseError> {
    let keyword = match tokens.get(*cursor) {
        Some(k) => k.to_uppercase(),
        None => return Err(ParseError::at(*cursor, "unexpected end of input")),
    };

    match keyword.as_str() {
        "INSERT" => {
            *cursor += 1;
            if matches!(peek(tokens, *cursor), Some(t) if t.to_uppercase() == "DATA") {
                *cursor += 1;
                let triples = parse_triple_block(tokens, cursor)?;
                Ok(SparqlUpdate::InsertData(triples))
            } else {
                // INSERT { template } WHERE { pattern }
                let template = parse_pattern_block(tokens, cursor)?;
                expect(tokens, cursor, "WHERE")?;
                let where_clause = parse_pattern_block(tokens, cursor)?;
                Ok(SparqlUpdate::InsertWhere {
                    template,
                    where_clause,
                })
            }
        }
        "DELETE" => {
            *cursor += 1;
            if matches!(peek(tokens, *cursor), Some(t) if t.to_uppercase() == "DATA") {
                *cursor += 1;
                let triples = parse_triple_block(tokens, cursor)?;
                Ok(SparqlUpdate::DeleteData(triples))
            } else if matches!(peek(tokens, *cursor), Some(t) if t.to_uppercase() == "WHERE") {
                // DELETE WHERE { pattern }
                *cursor += 1;
                let where_clause = parse_pattern_block(tokens, cursor)?;
                Ok(SparqlUpdate::DeleteWhere {
                    template: vec![],
                    where_clause,
                })
            } else {
                // DELETE { del_template } [INSERT { ins_template }] WHERE { pattern }
                let delete_template = parse_pattern_block(tokens, cursor)?;
                let insert_template = if matches!(peek(tokens, *cursor), Some(t) if t.to_uppercase() == "INSERT")
                {
                    *cursor += 1;
                    parse_pattern_block(tokens, cursor)?
                } else {
                    vec![]
                };
                expect(tokens, cursor, "WHERE")?;
                let where_clause = parse_pattern_block(tokens, cursor)?;
                Ok(SparqlUpdate::Modify {
                    delete: delete_template,
                    insert: insert_template,
                    where_clause,
                })
            }
        }
        "CREATE" => {
            *cursor += 1;
            let silent = parse_silent(tokens, cursor);
            consume_keyword(tokens, cursor, "GRAPH");
            let iri = parse_iri(tokens, cursor)?;
            Ok(SparqlUpdate::CreateGraph { iri, silent })
        }
        "DROP" => {
            *cursor += 1;
            let silent = parse_silent(tokens, cursor);
            parse_graph_target_update(tokens, cursor, |iri, drop_type| SparqlUpdate::DropGraph {
                iri,
                silent,
                drop_type: drop_type.into_drop(),
            })
        }
        "CLEAR" => {
            *cursor += 1;
            let silent = parse_silent(tokens, cursor);
            parse_graph_target_update(tokens, cursor, |iri, clear_type| SparqlUpdate::ClearGraph {
                iri,
                silent,
                clear_type: clear_type.into_clear(),
            })
        }
        "COPY" => {
            *cursor += 1;
            let silent = parse_silent(tokens, cursor);
            let source = parse_iri(tokens, cursor)?;
            expect(tokens, cursor, "TO")?;
            let target = parse_iri(tokens, cursor)?;
            Ok(SparqlUpdate::CopyGraph {
                source,
                target,
                silent,
            })
        }
        "MOVE" => {
            *cursor += 1;
            let silent = parse_silent(tokens, cursor);
            let source = parse_iri(tokens, cursor)?;
            expect(tokens, cursor, "TO")?;
            let target = parse_iri(tokens, cursor)?;
            Ok(SparqlUpdate::MoveGraph {
                source,
                target,
                silent,
            })
        }
        "ADD" => {
            *cursor += 1;
            let silent = parse_silent(tokens, cursor);
            let source = parse_iri(tokens, cursor)?;
            expect(tokens, cursor, "TO")?;
            let target = parse_iri(tokens, cursor)?;
            Ok(SparqlUpdate::AddGraph {
                source,
                target,
                silent,
            })
        }
        "LOAD" => {
            *cursor += 1;
            let silent = parse_silent(tokens, cursor);
            let iri = parse_iri(tokens, cursor)?;
            let into = if consume_keyword(tokens, cursor, "INTO") {
                consume_keyword(tokens, cursor, "GRAPH");
                Some(parse_iri(tokens, cursor)?)
            } else {
                None
            };
            Ok(SparqlUpdate::Load { iri, into, silent })
        }
        other => Err(ParseError::at(
            *cursor,
            format!("unknown update operation keyword: '{other}'"),
        )),
    }
}

// ---------------------------------------------------------------------------
// Graph target parsing helper (DROP / CLEAR share the same grammar)
// ---------------------------------------------------------------------------

/// Temporary enum for the parsed scope before converting to `DropType`/`ClearType`.
enum GraphScope {
    GraphIri,
    Default,
    Named,
    All,
}

impl GraphScope {
    fn into_drop(self) -> DropType {
        match self {
            GraphScope::GraphIri => DropType::Graph,
            GraphScope::Default => DropType::Default,
            GraphScope::Named => DropType::Named,
            GraphScope::All => DropType::All,
        }
    }

    fn into_clear(self) -> ClearType {
        match self {
            GraphScope::GraphIri => ClearType::Graph,
            GraphScope::Default => ClearType::Default,
            GraphScope::Named => ClearType::Named,
            GraphScope::All => ClearType::All,
        }
    }
}

fn parse_graph_target_update<F>(
    tokens: &[String],
    cursor: &mut usize,
    builder: F,
) -> Result<SparqlUpdate, ParseError>
where
    F: FnOnce(Option<String>, GraphScope) -> SparqlUpdate,
{
    let keyword = tokens.get(*cursor).map(|t| t.to_uppercase());
    match keyword.as_deref() {
        Some("DEFAULT") => {
            *cursor += 1;
            let scope = GraphScope::Default;
            Ok(builder(None, scope))
        }
        Some("NAMED") => {
            *cursor += 1;
            let scope = GraphScope::Named;
            Ok(builder(None, scope))
        }
        Some("ALL") => {
            *cursor += 1;
            let scope = GraphScope::All;
            Ok(builder(None, scope))
        }
        Some("GRAPH") => {
            *cursor += 1;
            let iri = parse_iri(tokens, cursor)?;
            let scope = GraphScope::GraphIri;
            Ok(builder(Some(iri), scope))
        }
        // Bare IRI without GRAPH keyword.
        Some(tok) if tok.starts_with('<') => {
            let iri = parse_iri(tokens, cursor)?;
            let scope = GraphScope::GraphIri;
            Ok(builder(Some(iri), scope))
        }
        _ => Err(ParseError::at(
            *cursor,
            "expected graph scope (DEFAULT | NAMED | ALL | GRAPH <iri>)",
        )),
    }
}

// ---------------------------------------------------------------------------
// UpdateResult
// ---------------------------------------------------------------------------

/// Summary of the changes made by a single `UpdateExecutor::execute` call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpdateResult {
    /// Number of triples inserted into the default graph or named graphs.
    pub triples_inserted: usize,
    /// Number of triples deleted from the default graph or named graphs.
    pub triples_deleted: usize,
    /// Number of distinct graphs affected (created, cleared, populated, etc.).
    pub graphs_affected: usize,
}

// ---------------------------------------------------------------------------
// ArqError (thin wrapper)
// ---------------------------------------------------------------------------

/// Error type for the update executor.
#[derive(Debug, Clone, PartialEq)]
pub struct ArqError(pub String);

impl std::fmt::Display for ArqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ARQ error: {}", self.0)
    }
}

impl std::error::Error for ArqError {}

// ---------------------------------------------------------------------------
// In-memory UpdateExecutor
// ---------------------------------------------------------------------------

/// A minimal in-memory dataset executor for SPARQL 1.1 Update.
///
/// It maintains a *default graph* (a `Vec<Triple>`) and an arbitrary number
/// of *named graphs* keyed by IRI string.  Pattern-based operations perform a
/// simple structural match (no unification — variables are left as wildcards
/// that match anything).
pub struct UpdateExecutor {
    /// Triples in the default graph.
    triples: Vec<Triple>,
    /// Named graphs keyed by IRI.
    named_graphs: HashMap<String, Vec<Triple>>,
}

impl UpdateExecutor {
    /// Create an empty executor.
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
            named_graphs: HashMap::new(),
        }
    }

    /// Execute a single update and return a summary.
    pub fn execute(&mut self, update: &SparqlUpdate) -> Result<UpdateResult, ArqError> {
        match update {
            SparqlUpdate::InsertData(triples) => {
                let count = triples.len();
                self.triples.extend(triples.iter().cloned());
                Ok(UpdateResult {
                    triples_inserted: count,
                    triples_deleted: 0,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::DeleteData(triples) => {
                let before = self.triples.len();
                for t in triples {
                    self.triples.retain(|existing| existing != t);
                }
                let deleted = before - self.triples.len();
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: deleted,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::InsertWhere {
                template,
                where_clause,
            } => {
                let bindings = self.match_patterns(where_clause);
                let mut inserted = 0usize;
                for binding in &bindings {
                    if let Some(triple) = instantiate_template_triple(template.first(), binding) {
                        for t in &triple {
                            if !self.triples.contains(t) {
                                self.triples.push(t.clone());
                                inserted += 1;
                            }
                        }
                    }
                }
                Ok(UpdateResult {
                    triples_inserted: inserted,
                    triples_deleted: 0,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::DeleteWhere { where_clause, .. } => {
                let bindings = self.match_patterns(where_clause);
                let to_delete: Vec<Triple> = bindings
                    .into_iter()
                    .filter_map(|b| {
                        let s = b.get("s").cloned()?;
                        let p = b.get("p").cloned()?;
                        let o = b.get("o").cloned()?;
                        Some(Triple::new(s, p, o))
                    })
                    .collect();
                let before = self.triples.len();
                for t in &to_delete {
                    self.triples.retain(|e| e != t);
                }
                let deleted = before - self.triples.len();
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: deleted,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::Modify {
                delete,
                insert,
                where_clause,
            } => {
                let bindings = self.match_patterns(where_clause);
                let mut inserted = 0usize;
                let mut deleted_count = 0usize;
                for binding in &bindings {
                    // Delete first.
                    for tp in delete {
                        if let Some(t) = instantiate_one(tp, binding) {
                            let before = self.triples.len();
                            self.triples.retain(|e| e != &t);
                            deleted_count += before - self.triples.len();
                        }
                    }
                    // Then insert.
                    for tp in insert {
                        if let Some(t) = instantiate_one(tp, binding) {
                            if !self.triples.contains(&t) {
                                self.triples.push(t);
                                inserted += 1;
                            }
                        }
                    }
                }
                Ok(UpdateResult {
                    triples_inserted: inserted,
                    triples_deleted: deleted_count,
                    graphs_affected: 0,
                })
            }
            SparqlUpdate::CreateGraph { iri, silent } => {
                if self.named_graphs.contains_key(iri) && !silent {
                    return Err(ArqError(format!("graph <{iri}> already exists")));
                }
                self.named_graphs.entry(iri.clone()).or_default();
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: 0,
                    graphs_affected: 1,
                })
            }
            SparqlUpdate::DropGraph {
                iri,
                silent,
                drop_type,
            } => {
                let count = match drop_type {
                    DropType::Graph => {
                        let key = iri.as_deref().unwrap_or("");
                        if self.named_graphs.remove(key).is_none() && !silent {
                            return Err(ArqError(format!("graph <{key}> does not exist")));
                        }
                        1
                    }
                    DropType::Default => {
                        self.triples.clear();
                        1
                    }
                    DropType::Named => {
                        let count = self.named_graphs.len();
                        self.named_graphs.clear();
                        count
                    }
                    DropType::All => {
                        let ng = self.named_graphs.len();
                        self.named_graphs.clear();
                        self.triples.clear();
                        ng + 1
                    }
                };
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: 0,
                    graphs_affected: count,
                })
            }
            SparqlUpdate::ClearGraph {
                iri,
                silent,
                clear_type,
            } => {
                let count = match clear_type {
                    ClearType::Graph => {
                        let key = iri.as_deref().unwrap_or("");
                        match self.named_graphs.get_mut(key) {
                            Some(g) => {
                                g.clear();
                                1
                            }
                            None if *silent => 0,
                            None => return Err(ArqError(format!("graph <{key}> does not exist"))),
                        }
                    }
                    ClearType::Default => {
                        self.triples.clear();
                        1
                    }
                    ClearType::Named => {
                        for g in self.named_graphs.values_mut() {
                            g.clear();
                        }
                        self.named_graphs.len()
                    }
                    ClearType::All => {
                        self.triples.clear();
                        for g in self.named_graphs.values_mut() {
                            g.clear();
                        }
                        self.named_graphs.len() + 1
                    }
                };
                Ok(UpdateResult {
                    triples_inserted: 0,
                    triples_deleted: 0,
                    graphs_affected: count,
                })
            }
            SparqlUpdate::CopyGraph {
                source,
                target,
                silent: _,
            } => {
                let src_triples: Vec<Triple> =
                    self.named_graphs.get(source).cloned().unwrap_or_default();
                let count = src_triples.len();
                let tgt = self.named_graphs.entry(target.clone()).or_default();
                tgt.clear();
                tgt.extend(src_triples);
                Ok(UpdateResult {
                    triples_inserted: count,
                    triples_deleted: 0,
                    graphs_affected: 1,
                })
            }
            SparqlUpdate::MoveGraph {
                source,
                target,
                silent: _,
            } => {
                let src_triples = self.named_graphs.remove(source).unwrap_or_default();
                let count = src_triples.len();
                let tgt = self.named_graphs.entry(target.clone()).or_default();
                tgt.clear();
                tgt.extend(src_triples);
                Ok(UpdateResult {
                    triples_inserted: count,
                    triples_deleted: 0,
                    graphs_affected: 2,
                })
            }
            SparqlUpdate::AddGraph {
                source,
                target,
                silent: _,
            } => {
                let src_triples: Vec<Triple> =
                    self.named_graphs.get(source).cloned().unwrap_or_default();
                let count = src_triples.len();
                let tgt = self.named_graphs.entry(target.clone()).or_default();
                tgt.extend(src_triples);
                Ok(UpdateResult {
                    triples_inserted: count,
                    triples_deleted: 0,
                    graphs_affected: 1,
                })
            }
            SparqlUpdate::Load { iri, into, silent } => {
                // Actual HTTP loading is not implemented in this in-memory executor.
                // Return success (silent) or error (non-silent).
                if *silent {
                    Ok(UpdateResult::default())
                } else {
                    Err(ArqError(format!(
                        "LOAD is not supported in the in-memory executor (iri=<{iri}>, into={into:?})"
                    )))
                }
            }
        }
    }

    /// Execute a sequence of update operations and collect their results.
    pub fn execute_all(&mut self, updates: &[SparqlUpdate]) -> Result<Vec<UpdateResult>, ArqError> {
        updates.iter().map(|u| self.execute(u)).collect()
    }

    /// Number of triples in the default graph.
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    /// Number of named graphs (not counting the default graph).
    pub fn graph_count(&self) -> usize {
        self.named_graphs.len()
    }

    /// Return the triples in a named graph, or `None` if it does not exist.
    pub fn get_graph(&self, iri: &str) -> Option<&Vec<Triple>> {
        self.named_graphs.get(iri)
    }

    /// Return a reference to the default graph's triple set.
    pub fn default_graph(&self) -> &Vec<Triple> {
        &self.triples
    }
}

impl Default for UpdateExecutor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pattern matching helpers
// ---------------------------------------------------------------------------

type Binding = HashMap<String, String>;

/// Match a slice of triple patterns against the default graph, returning all
/// consistent variable bindings.  Each pattern is matched independently and
/// bindings from consecutive patterns are intersected by joining on shared
/// variable names.
fn match_patterns(triples: &[Triple], patterns: &[TriplePattern]) -> Vec<Binding> {
    let mut results: Vec<Binding> = vec![HashMap::new()];

    for pattern in patterns {
        let mut next: Vec<Binding> = Vec::new();
        for binding in &results {
            for triple in triples {
                if let Some(new_binding) = match_pattern(triple, pattern, binding) {
                    next.push(new_binding);
                }
            }
        }
        results = next;
    }

    results
}

/// Try to extend `existing_binding` with the variable bindings produced by
/// matching `triple` against `pattern`.  Returns `None` on conflict.
fn match_pattern(triple: &Triple, pattern: &TriplePattern, existing: &Binding) -> Option<Binding> {
    let mut binding = existing.clone();
    bind_term(&triple.s, &pattern.s, &mut binding)?;
    bind_term(&triple.p, &pattern.p, &mut binding)?;
    bind_term(&triple.o, &pattern.o, &mut binding)?;
    Some(binding)
}

/// Attempt to bind `value` against `term`, extending `binding` if `term` is a
/// variable.  Returns `None` when an existing binding is inconsistent.
fn bind_term(value: &str, term: &PatternTerm, binding: &mut Binding) -> Option<()> {
    match term {
        PatternTerm::Variable(var) => {
            if let Some(existing) = binding.get(var.as_str()) {
                if existing != value {
                    return None;
                }
            } else {
                binding.insert(var.clone(), value.to_string());
            }
            Some(())
        }
        PatternTerm::Iri(iri) => {
            if iri == value {
                Some(())
            } else {
                None
            }
        }
        PatternTerm::Literal(lit) => {
            // Compare the content without surrounding quotes.
            let inner = lit.trim_matches('"').trim_matches('\'');
            if inner == value || lit == value {
                Some(())
            } else {
                None
            }
        }
        PatternTerm::BlankNode(bn) => {
            if bn == value {
                Some(())
            } else {
                None
            }
        }
    }
}

impl UpdateExecutor {
    /// Match patterns against the default graph's triple set.
    fn match_patterns(&self, patterns: &[TriplePattern]) -> Vec<Binding> {
        match_patterns(&self.triples, patterns)
    }
}

/// Try to instantiate a single `TriplePattern` against a `Binding`, producing
/// a `Triple` when all positions resolve to concrete terms.
fn instantiate_one(pattern: &TriplePattern, binding: &Binding) -> Option<Triple> {
    let s = resolve_term(&pattern.s, binding)?;
    let p = resolve_term(&pattern.p, binding)?;
    let o = resolve_term(&pattern.o, binding)?;
    Some(Triple::new(s, p, o))
}

/// Try to instantiate the first `TriplePattern` in `templates`, returning a
/// `Vec<Triple>` (0 or 1 elements).  This helper is used for `InsertWhere`.
fn instantiate_template_triple(
    template: Option<&TriplePattern>,
    binding: &Binding,
) -> Option<Vec<Triple>> {
    let tp = template?;
    Some(instantiate_one(tp, binding).into_iter().collect())
}

/// Resolve a `PatternTerm` to a concrete string using `binding`.  Returns
/// `None` when a variable is unbound.
fn resolve_term(term: &PatternTerm, binding: &Binding) -> Option<String> {
    match term {
        PatternTerm::Variable(var) => binding.get(var.as_str()).cloned(),
        PatternTerm::Iri(iri) => Some(iri.clone()),
        PatternTerm::Literal(lit) => Some(lit.trim_matches('"').trim_matches('\'').to_string()),
        PatternTerm::BlankNode(bn) => Some(bn.clone()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Parser – InsertData
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_insert_data_single_triple() {
        let input = "INSERT DATA { <http://a> <http://b> <http://c> }";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::InsertData(triples) => {
                assert_eq!(triples.len(), 1);
                assert_eq!(triples[0].s, "http://a");
                assert_eq!(triples[0].p, "http://b");
                assert_eq!(triples[0].o, "http://c");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_insert_data_multiple_triples() {
        let input = "INSERT DATA { <s1> <p1> <o1> . <s2> <p2> <o2> }";
        let result = SparqlUpdateParser::parse_one(input).expect("parse failed");
        if let SparqlUpdate::InsertData(triples) = result {
            assert_eq!(triples.len(), 2);
        } else {
            panic!("wrong variant");
        }
    }

    // ------------------------------------------------------------------
    // Parser – DeleteData
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_delete_data() {
        let input = "DELETE DATA { <http://x> <http://y> <http://z> }";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::DeleteData(triples) => {
                assert_eq!(triples.len(), 1);
                assert_eq!(triples[0].s, "http://x");
            }
            _ => panic!("wrong variant"),
        }
    }

    // ------------------------------------------------------------------
    // Parser – CreateGraph
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_create_graph() {
        let input = "CREATE GRAPH <http://example.org/g>";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::CreateGraph { iri, silent } => {
                assert_eq!(iri, "http://example.org/g");
                assert!(!silent);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_create_graph_silent() {
        let input = "CREATE SILENT GRAPH <http://example.org/g>";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::CreateGraph { iri: _, silent } => {
                assert!(silent);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ------------------------------------------------------------------
    // Parser – DropGraph
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_drop_graph_named() {
        let input = "DROP GRAPH <http://g>";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::DropGraph {
                iri,
                silent,
                drop_type,
            } => {
                assert_eq!(iri, Some("http://g".to_string()));
                assert!(!silent);
                assert_eq!(drop_type, DropType::Graph);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_drop_all() {
        let input = "DROP ALL";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::DropGraph { drop_type, .. } => {
                assert_eq!(drop_type, DropType::All);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ------------------------------------------------------------------
    // Parser – ClearGraph
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_clear_default() {
        let input = "CLEAR DEFAULT";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::ClearGraph { clear_type, .. } => {
                assert_eq!(clear_type, ClearType::Default);
            }
            _ => panic!("wrong variant"),
        }
    }

    // ------------------------------------------------------------------
    // Parser – CopyGraph, MoveGraph, AddGraph
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_copy_graph() {
        let input = "COPY <http://src> TO <http://dst>";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::CopyGraph {
                source,
                target,
                silent,
            } => {
                assert_eq!(source, "http://src");
                assert_eq!(target, "http://dst");
                assert!(!silent);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_move_graph() {
        let input = "MOVE <http://old> TO <http://new>";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::MoveGraph { source, target, .. } => {
                assert_eq!(source, "http://old");
                assert_eq!(target, "http://new");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_add_graph() {
        let input = "ADD <http://src> TO <http://dst>";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::AddGraph { .. } => {}
            _ => panic!("wrong variant"),
        }
    }

    // ------------------------------------------------------------------
    // Parser – LOAD
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_load_basic() {
        let input = "LOAD <http://data.example.org/data.ttl>";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::Load { iri, into, silent } => {
                assert_eq!(iri, "http://data.example.org/data.ttl");
                assert!(into.is_none());
                assert!(!silent);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_load_into_graph() {
        let input = "LOAD <http://src.ttl> INTO GRAPH <http://target>";
        let update = SparqlUpdateParser::parse_one(input).expect("parse failed");
        match update {
            SparqlUpdate::Load { into, .. } => {
                assert_eq!(into, Some("http://target".to_string()));
            }
            _ => panic!("wrong variant"),
        }
    }

    // ------------------------------------------------------------------
    // Parser – multiple operations
    // ------------------------------------------------------------------

    #[test]
    fn test_parse_multiple_operations() {
        let input = "INSERT DATA { <s1> <p1> <o1> } ; DELETE DATA { <s2> <p2> <o2> }";
        let updates = SparqlUpdateParser::parse(input).expect("parse failed");
        assert_eq!(updates.len(), 2);
    }

    #[test]
    fn test_parse_empty_input() {
        let updates = SparqlUpdateParser::parse("").expect("parse failed");
        assert!(updates.is_empty());
    }

    #[test]
    fn test_parse_unknown_keyword_returns_error() {
        let result = SparqlUpdateParser::parse_one("FROBULATE { }");
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // UpdateExecutor – InsertData / DeleteData
    // ------------------------------------------------------------------

    #[test]
    fn test_executor_insert_data() {
        let mut exec = UpdateExecutor::new();
        let update = SparqlUpdate::InsertData(vec![Triple::new("s", "p", "o")]);
        let result = exec.execute(&update).expect("execute failed");
        assert_eq!(result.triples_inserted, 1);
        assert_eq!(exec.triple_count(), 1);
    }

    #[test]
    fn test_executor_delete_data_existing() {
        let mut exec = UpdateExecutor::new();
        let t = Triple::new("s", "p", "o");
        exec.execute(&SparqlUpdate::InsertData(vec![t.clone()]))
            .expect("insert failed");
        let result = exec
            .execute(&SparqlUpdate::DeleteData(vec![t]))
            .expect("delete failed");
        assert_eq!(result.triples_deleted, 1);
        assert_eq!(exec.triple_count(), 0);
    }

    #[test]
    fn test_executor_delete_data_nonexistent() {
        let mut exec = UpdateExecutor::new();
        let result = exec
            .execute(&SparqlUpdate::DeleteData(vec![Triple::new("x", "y", "z")]))
            .expect("delete failed");
        assert_eq!(result.triples_deleted, 0);
    }

    // ------------------------------------------------------------------
    // UpdateExecutor – CreateGraph / DropGraph
    // ------------------------------------------------------------------

    #[test]
    fn test_executor_create_graph() {
        let mut exec = UpdateExecutor::new();
        exec.execute(&SparqlUpdate::CreateGraph {
            iri: "http://g".to_string(),
            silent: false,
        })
        .expect("create failed");
        assert_eq!(exec.graph_count(), 1);
        assert!(exec.get_graph("http://g").is_some());
    }

    #[test]
    fn test_executor_create_duplicate_graph_silent() {
        let mut exec = UpdateExecutor::new();
        let update = SparqlUpdate::CreateGraph {
            iri: "http://g".to_string(),
            silent: true,
        };
        exec.execute(&update).expect("first create failed");
        exec.execute(&update)
            .expect("second create (silent) should not error");
    }

    #[test]
    fn test_executor_create_duplicate_graph_non_silent_errors() {
        let mut exec = UpdateExecutor::new();
        let update = SparqlUpdate::CreateGraph {
            iri: "http://g".to_string(),
            silent: false,
        };
        exec.execute(&update).expect("first create failed");
        let result = exec.execute(&update);
        assert!(result.is_err());
    }

    #[test]
    fn test_executor_drop_named_graph() {
        let mut exec = UpdateExecutor::new();
        exec.execute(&SparqlUpdate::CreateGraph {
            iri: "http://g".to_string(),
            silent: false,
        })
        .expect("create failed");
        exec.execute(&SparqlUpdate::DropGraph {
            iri: Some("http://g".to_string()),
            silent: false,
            drop_type: DropType::Graph,
        })
        .expect("drop failed");
        assert_eq!(exec.graph_count(), 0);
    }

    // ------------------------------------------------------------------
    // UpdateExecutor – ClearGraph
    // ------------------------------------------------------------------

    #[test]
    fn test_executor_clear_default_graph() {
        let mut exec = UpdateExecutor::new();
        exec.execute(&SparqlUpdate::InsertData(vec![
            Triple::new("s1", "p1", "o1"),
            Triple::new("s2", "p2", "o2"),
        ]))
        .expect("insert failed");
        exec.execute(&SparqlUpdate::ClearGraph {
            iri: None,
            silent: false,
            clear_type: ClearType::Default,
        })
        .expect("clear failed");
        assert_eq!(exec.triple_count(), 0);
    }

    // ------------------------------------------------------------------
    // UpdateExecutor – execute_all
    // ------------------------------------------------------------------

    #[test]
    fn test_executor_execute_all() {
        let mut exec = UpdateExecutor::new();
        let updates = vec![
            SparqlUpdate::InsertData(vec![Triple::new("a", "b", "c")]),
            SparqlUpdate::InsertData(vec![Triple::new("d", "e", "f")]),
        ];
        let results = exec.execute_all(&updates).expect("execute_all failed");
        assert_eq!(results.len(), 2);
        assert_eq!(exec.triple_count(), 2);
    }

    // ------------------------------------------------------------------
    // Triple / PatternTerm helpers
    // ------------------------------------------------------------------

    #[test]
    fn test_triple_equality() {
        let t1 = Triple::new("s", "p", "o");
        let t2 = Triple::new("s", "p", "o");
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_pattern_term_is_variable() {
        let var = PatternTerm::Variable("x".to_string());
        assert!(var.is_variable());
        let iri = PatternTerm::Iri("http://a".to_string());
        assert!(!iri.is_variable());
    }

    #[test]
    fn test_pattern_term_variable_name() {
        let var = PatternTerm::Variable("myVar".to_string());
        assert_eq!(var.variable_name(), Some("myVar"));
        let iri = PatternTerm::Iri("http://a".to_string());
        assert_eq!(iri.variable_name(), None);
    }

    // ------------------------------------------------------------------
    // UpdateExecutor – CopyGraph / MoveGraph / AddGraph
    // ------------------------------------------------------------------

    #[test]
    fn test_executor_copy_graph() {
        let mut exec = UpdateExecutor::new();
        exec.execute(&SparqlUpdate::CreateGraph {
            iri: "http://src".into(),
            silent: false,
        })
        .expect("create src");
        // Manually add a triple to the named graph via the HashMap.
        exec.named_graphs
            .get_mut("http://src")
            .expect("src exists")
            .push(Triple::new("a", "b", "c"));

        exec.execute(&SparqlUpdate::CopyGraph {
            source: "http://src".into(),
            target: "http://dst".into(),
            silent: false,
        })
        .expect("copy failed");
        assert_eq!(exec.get_graph("http://dst").expect("dst exists").len(), 1);
    }

    #[test]
    fn test_executor_move_graph() {
        let mut exec = UpdateExecutor::new();
        exec.execute(&SparqlUpdate::CreateGraph {
            iri: "http://src".into(),
            silent: false,
        })
        .expect("create src");
        exec.named_graphs
            .get_mut("http://src")
            .expect("src exists")
            .push(Triple::new("x", "y", "z"));

        exec.execute(&SparqlUpdate::MoveGraph {
            source: "http://src".into(),
            target: "http://dst".into(),
            silent: false,
        })
        .expect("move failed");

        assert!(exec.get_graph("http://src").is_none());
        assert_eq!(exec.get_graph("http://dst").expect("dst exists").len(), 1);
    }
}
