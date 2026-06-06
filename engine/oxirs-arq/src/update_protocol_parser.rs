//! Tokeniser and recursive-descent parser for SPARQL 1.1 Update text.
//!
//! The [`SparqlUpdateParser`] consumes a SPARQL Update string and produces a
//! sequence of [`SparqlUpdate`] operations.  The implementation is a
//! hand-written, single-pass tokeniser plus recursive descent — it covers
//! the most common SPARQL 1.1 Update grammar productions but does not
//! perform prefix expansion (callers should expand prefixes themselves).

use crate::update_protocol_types::{
    ClearType, DropType, ParseError, PatternTerm, SparqlUpdate, Triple, TriplePattern,
};

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
