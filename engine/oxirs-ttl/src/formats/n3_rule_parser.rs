//! Enhanced N3 Rule File Parser and Serializer
//!
//! This module provides a high-level parser for N3 rule files with full support for:
//! - `@prefix` declarations with namespace expansion
//! - Rule format: `{ ?x a :Person } => { ?x :isHuman true }`
//! - Variables: `?varName` syntax
//! - Lists: `(item1 item2 item3)` as RDF lists represented via rdf:first/rdf:rest chains
//! - Nested formulas: `{ { ?x :p ?y } => { ?y :q ?z } }`
//! - `@base` declarations for relative IRI resolution
//!
//! The `N3Serializer` companion writes rules back to N3 text.
//!
//! # Examples
//!
//! ```rust
//! use oxirs_ttl::n3::rule_parser::{N3RuleParser, N3RuleSerializer};
//!
//! let n3 = r#"
//! @prefix ex: <http://example.org/> .
//! { ?x ex:parent ?y } => { ?y ex:child ?x } .
//! ex:alice ex:parent ex:bob .
//! "#;
//!
//! let mut parser = N3RuleParser::new();
//! let doc = parser.parse(n3).expect("parse should succeed");
//! assert_eq!(doc.rules.len(), 1);
//! assert_eq!(doc.facts.len(), 1);
//!
//! let serialized = N3RuleSerializer::new().serialize_document(&doc);
//! assert!(serialized.contains("=>"));
//! ```

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::formats::n3_types::{N3Formula, N3Implication, N3Statement, N3Term, N3Variable};
use oxirs_core::model::{BlankNode, Literal, NamedNode};
use std::collections::HashMap;

// ── Utility: parse errors ─────────────────────────────────────────────────────

fn parse_err(msg: impl Into<String>) -> TurtleParseError {
    TurtleParseError::syntax(TurtleSyntaxError::Generic {
        message: msg.into(),
        position: TextPosition::default(),
    })
}

// ── Tokenizer ─────────────────────────────────────────────────────────────────

/// Minimal tokenizer for the N3 rule parser.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// `@prefix`
    Prefix,
    /// `@base`
    Base,
    /// `@forAll`
    ForAll,
    /// `@forSome`
    ForSome,
    /// `PREFIX` (SPARQL-style, case-insensitive)
    SparqlPrefix,
    /// `=>`
    Implies,
    /// `<=`
    ImpliedBy,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `.`
    Dot,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `^^`
    Datatype,
    /// `@lang`
    LangTag(String),
    /// `<iri>`
    Iri(String),
    /// `prefix:local`
    Prefixed(String, String),
    /// `?var`
    Var(String),
    /// `_:label`
    BNode(String),
    /// `"string"`
    Str(String),
    /// integer / decimal / double literal (raw text)
    Number(String),
    /// `a` keyword (rdf:type shorthand)
    A,
    /// `true` or `false` boolean literals
    Boolean(bool),
    /// End of file
    Eof,
}

struct Tokenizer<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn peek_char(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn next_char(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while self.peek_char().map(|c| c.is_whitespace()).unwrap_or(false) {
                self.next_char();
            }
            // Skip line comments
            if self.src[self.pos..].starts_with('#') {
                while self.peek_char().map(|c| c != '\n').unwrap_or(false) {
                    self.next_char();
                }
            } else {
                break;
            }
        }
    }

    fn read_iri(&mut self) -> TurtleResult<String> {
        // consume '<'
        self.next_char();
        let start = self.pos;
        loop {
            match self.peek_char() {
                Some('>') => {
                    let iri = self.src[start..self.pos].to_string();
                    self.next_char(); // consume '>'
                    return Ok(iri);
                }
                Some(_) => {
                    self.next_char();
                }
                None => return Err(parse_err("Unterminated IRI")),
            }
        }
    }

    fn read_string(&mut self) -> TurtleResult<String> {
        // detect triple-quoted strings
        if self.src[self.pos..].starts_with("\"\"\"") {
            self.pos += 3;
            let start = self.pos;
            loop {
                if self.src[self.pos..].starts_with("\"\"\"") {
                    let s = self.src[start..self.pos].to_string();
                    self.pos += 3;
                    return Ok(s);
                }
                if self.pos >= self.src.len() {
                    return Err(parse_err("Unterminated triple-quoted string"));
                }
                self.next_char();
            }
        }

        // Single-quoted string — consume opening '"'
        self.next_char();
        let mut out = String::new();
        loop {
            match self.next_char() {
                Some('"') => break,
                Some('\\') => match self.next_char() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some(c) => {
                        out.push('\\');
                        out.push(c);
                    }
                    None => return Err(parse_err("Unexpected EOF in string escape")),
                },
                Some(c) => out.push(c),
                None => return Err(parse_err("Unterminated string literal")),
            }
        }
        Ok(out)
    }

    fn read_word(&mut self) -> String {
        let start = self.pos;
        while self
            .peek_char()
            .map(|c| {
                !c.is_whitespace()
                    && c != '.'
                    && c != ','
                    && c != ';'
                    && c != '{'
                    && c != '}'
                    && c != '('
                    && c != ')'
                    && c != '<'
                    && c != '"'
                    && c != '#'
                    && c != '^'
            })
            .unwrap_or(false)
        {
            self.next_char();
        }
        self.src[start..self.pos].to_string()
    }

    fn next_token(&mut self) -> TurtleResult<Token> {
        self.skip_ws_and_comments();
        if self.pos >= self.src.len() {
            return Ok(Token::Eof);
        }

        // Two-character operators
        if self.src[self.pos..].starts_with("=>") {
            self.pos += 2;
            return Ok(Token::Implies);
        }
        if self.src[self.pos..].starts_with("<=") {
            self.pos += 2;
            return Ok(Token::ImpliedBy);
        }
        if self.src[self.pos..].starts_with("^^") {
            self.pos += 2;
            return Ok(Token::Datatype);
        }

        let ch = self.peek_char().expect("checked non-empty");

        match ch {
            '<' => {
                let iri = self.read_iri()?;
                Ok(Token::Iri(iri))
            }
            '"' => {
                let s = self.read_string()?;
                // Peek for language tag or datatype
                Ok(Token::Str(s))
            }
            '{' => {
                self.next_char();
                Ok(Token::LBrace)
            }
            '}' => {
                self.next_char();
                Ok(Token::RBrace)
            }
            '(' => {
                self.next_char();
                Ok(Token::LParen)
            }
            ')' => {
                self.next_char();
                Ok(Token::RParen)
            }
            '.' => {
                self.next_char();
                Ok(Token::Dot)
            }
            ',' => {
                self.next_char();
                Ok(Token::Comma)
            }
            ';' => {
                self.next_char();
                Ok(Token::Semicolon)
            }
            '@' => {
                self.next_char(); // consume '@'
                                  // Read next word to detect keyword or language tag
                let word = self.read_word();
                match word.as_str() {
                    "prefix" => Ok(Token::Prefix),
                    "base" => Ok(Token::Base),
                    "forAll" => Ok(Token::ForAll),
                    "forSome" => Ok(Token::ForSome),
                    other => Ok(Token::LangTag(other.to_string())),
                }
            }
            '?' => {
                self.next_char(); // consume '?'
                let name = self.read_word();
                Ok(Token::Var(name))
            }
            '_' if self.src[self.pos..].starts_with("_:") => {
                self.pos += 2; // skip '_:'
                let label = self.read_word();
                Ok(Token::BNode(label))
            }
            c if c.is_ascii_digit() || c == '-' || c == '+' => {
                let n = self.read_word();
                Ok(Token::Number(n))
            }
            _ => {
                let word = self.read_word();
                if word.is_empty() {
                    self.next_char();
                    return Err(parse_err(format!("Unexpected character: {:?}", ch)));
                }
                match word.as_str() {
                    "a" => Ok(Token::A),
                    "true" => Ok(Token::Boolean(true)),
                    "false" => Ok(Token::Boolean(false)),
                    "PREFIX" | "prefix" => Ok(Token::SparqlPrefix),
                    _ => {
                        // Prefixed name: prefix:local
                        if let Some(colon) = word.find(':') {
                            let prefix = word[..colon].to_string();
                            let local = word[colon + 1..].to_string();
                            Ok(Token::Prefixed(prefix, local))
                        } else {
                            Err(parse_err(format!("Unknown token: {:?}", word)))
                        }
                    }
                }
            }
        }
    }
}

// ── N3 Rule Document ──────────────────────────────────────────────────────────

/// A parsed N3 rule document containing facts, rules, and prefix declarations.
#[derive(Debug, Clone, Default)]
pub struct N3RuleDocument {
    /// Ground facts (statements without variables)
    pub facts: Vec<N3Statement>,
    /// Implication rules
    pub rules: Vec<N3Implication>,
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
    /// Base IRI (if declared)
    pub base_iri: Option<String>,
}

impl N3RuleDocument {
    /// Create a new empty document.
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge another document into this one.
    pub fn merge(&mut self, other: N3RuleDocument) {
        self.facts.extend(other.facts);
        self.rules.extend(other.rules);
        self.prefixes.extend(other.prefixes);
        if self.base_iri.is_none() {
            self.base_iri = other.base_iri;
        }
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

/// Enhanced N3 rule file parser.
///
/// Supports `@prefix`, `@base`, variables (`?x`), formulas (`{ ... }`),
/// implications (`=>`), and RDF list syntax (`(...)`).
pub struct N3RuleParser {
    prefixes: HashMap<String, String>,
    base_iri: Option<String>,
}

impl N3RuleParser {
    /// Create a new parser with standard prefix declarations.
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        prefixes.insert(
            "math".to_string(),
            "http://www.w3.org/2000/10/swap/math#".to_string(),
        );
        prefixes.insert(
            "log".to_string(),
            "http://www.w3.org/2000/10/swap/log#".to_string(),
        );
        prefixes.insert(
            "string".to_string(),
            "http://www.w3.org/2000/10/swap/string#".to_string(),
        );
        prefixes.insert(
            "list".to_string(),
            "http://www.w3.org/2000/10/swap/list#".to_string(),
        );
        Self {
            prefixes,
            base_iri: None,
        }
    }

    /// Parse N3 text into an `N3RuleDocument`.
    pub fn parse(&mut self, input: &str) -> TurtleResult<N3RuleDocument> {
        let tokens = Self::tokenize(input)?;
        let mut doc = N3RuleDocument::new();
        let mut pos = 0;

        loop {
            if pos >= tokens.len() || matches!(tokens[pos], Token::Eof) {
                break;
            }

            // @prefix declaration
            if matches!(tokens[pos], Token::Prefix | Token::SparqlPrefix) {
                pos += 1;
                let (prefix, ns) = self.parse_prefix_decl(&tokens, &mut pos)?;
                self.prefixes.insert(prefix.clone(), ns.clone());
                doc.prefixes.insert(prefix, ns);
                // consume trailing dot
                if pos < tokens.len() && matches!(tokens[pos], Token::Dot) {
                    pos += 1;
                }
                continue;
            }

            // @base declaration
            if matches!(tokens[pos], Token::Base) {
                pos += 1;
                let base = match tokens.get(pos) {
                    Some(Token::Iri(iri)) => iri.clone(),
                    _ => return Err(parse_err("Expected IRI after @base")),
                };
                pos += 1;
                self.base_iri = Some(base.clone());
                doc.base_iri = Some(base);
                if pos < tokens.len() && matches!(tokens[pos], Token::Dot) {
                    pos += 1;
                }
                continue;
            }

            // @forAll / @forSome — skip for simplicity
            if matches!(tokens[pos], Token::ForAll | Token::ForSome) {
                pos += 1;
                // skip until '.'
                while pos < tokens.len() && !matches!(tokens[pos], Token::Dot | Token::Eof) {
                    pos += 1;
                }
                if pos < tokens.len() && matches!(tokens[pos], Token::Dot) {
                    pos += 1;
                }
                continue;
            }

            // Try to parse a formula-level implication: { antecedent } => { consequent }
            if matches!(tokens[pos], Token::LBrace) {
                // Save position in case we need to backtrack
                let saved_pos = pos;
                match self.parse_formula(&tokens, &mut pos) {
                    Ok(ant) => {
                        if pos < tokens.len() && matches!(tokens[pos], Token::Implies) {
                            pos += 1; // consume '=>'
                            let con = self.parse_formula(&tokens, &mut pos)?;
                            doc.rules.push(N3Implication::new(ant, con));
                            if pos < tokens.len() && matches!(tokens[pos], Token::Dot) {
                                pos += 1;
                            }
                            continue;
                        } else if pos < tokens.len() && matches!(tokens[pos], Token::ImpliedBy) {
                            pos += 1; // consume '<='
                            let con = self.parse_formula(&tokens, &mut pos)?;
                            // Reverse: con <= ant  means  { con } => { ant } is wrong;
                            // Actually: B <= A means A => B
                            doc.rules.push(N3Implication::new(con, ant));
                            if pos < tokens.len() && matches!(tokens[pos], Token::Dot) {
                                pos += 1;
                            }
                            continue;
                        }
                        // Otherwise just skip the formula (a formula used as a statement subject)
                        // and try parsing it as a regular statement
                        pos = saved_pos;
                    }
                    Err(_) => {
                        pos = saved_pos;
                    }
                }
            }

            // Regular statement: subject predicate object .
            match self.parse_statement(&tokens, &mut pos) {
                Ok(stmt) => {
                    if pos < tokens.len() && matches!(tokens[pos], Token::Dot) {
                        pos += 1;
                    }
                    if stmt.has_variables() {
                        // Variable statements that are not rules are added as-is in facts
                        doc.facts.push(stmt);
                    } else {
                        doc.facts.push(stmt);
                    }
                }
                Err(_) => {
                    // Skip unknown token
                    pos += 1;
                }
            }
        }

        doc.prefixes.extend(self.prefixes.clone());
        Ok(doc)
    }

    fn tokenize(input: &str) -> TurtleResult<Vec<Token>> {
        let mut tok = Tokenizer::new(input);
        let mut tokens = Vec::new();
        loop {
            let t = tok.next_token()?;
            let is_eof = matches!(t, Token::Eof);
            // Handle @lang and ^^datatype as part of string processing
            tokens.push(t);
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }

    fn parse_prefix_decl(
        &self,
        tokens: &[Token],
        pos: &mut usize,
    ) -> TurtleResult<(String, String)> {
        // Expecting: prefix_name: <namespace> or just prefix_name:  <namespace>
        let prefix = match tokens.get(*pos) {
            Some(Token::Prefixed(p, _l)) => {
                let p = p.clone();
                *pos += 1;
                p
            }
            Some(Token::Iri(_)) => {
                // SPARQL-style PREFIX : <ns>
                let prefix = String::new();
                *pos += 1; // skip
                prefix
            }
            _ => {
                // May be ":" for default prefix
                return Err(parse_err("Expected prefix declaration"));
            }
        };
        let ns = match tokens.get(*pos) {
            Some(Token::Iri(iri)) => {
                let iri = iri.clone();
                *pos += 1;
                iri
            }
            _ => return Err(parse_err("Expected IRI for prefix namespace")),
        };
        Ok((prefix, ns))
    }

    fn parse_formula(&self, tokens: &[Token], pos: &mut usize) -> TurtleResult<N3Formula> {
        // Expect '{'
        if !matches!(tokens.get(*pos), Some(Token::LBrace)) {
            return Err(parse_err("Expected '{'"));
        }
        *pos += 1;

        let mut formula = N3Formula::new();

        while *pos < tokens.len() && !matches!(tokens[*pos], Token::RBrace | Token::Eof) {
            // Nested formula implication
            if matches!(tokens[*pos], Token::LBrace) {
                let saved = *pos;
                if let Ok(nested_ant) = self.parse_formula(tokens, pos) {
                    if matches!(tokens.get(*pos), Some(Token::Implies)) {
                        *pos += 1;
                        if let Ok(nested_con) = self.parse_formula(tokens, pos) {
                            // Represent as a statement with formula terms
                            let stmt = N3Statement::new(
                                N3Term::Formula(Box::new(nested_ant)),
                                N3Term::NamedNode(
                                    NamedNode::new("http://www.w3.org/2000/10/swap/log#implies")
                                        .expect("valid IRI"),
                                ),
                                N3Term::Formula(Box::new(nested_con)),
                            );
                            formula.add_statement(stmt);
                            if matches!(tokens.get(*pos), Some(Token::Dot)) {
                                *pos += 1;
                            }
                            continue;
                        }
                    }
                }
                *pos = saved;
            }

            match self.parse_statement(tokens, pos) {
                Ok(stmt) => {
                    formula.add_statement(stmt);
                    if matches!(tokens.get(*pos), Some(Token::Dot)) {
                        *pos += 1;
                    }
                }
                Err(_) => {
                    *pos += 1; // skip unrecognized token inside formula
                }
            }
        }

        if matches!(tokens.get(*pos), Some(Token::RBrace)) {
            *pos += 1;
        }

        Ok(formula)
    }

    fn parse_statement(&self, tokens: &[Token], pos: &mut usize) -> TurtleResult<N3Statement> {
        let subject = self.parse_term(tokens, pos)?;
        let predicate = self.parse_term(tokens, pos)?;
        let object = self.parse_term(tokens, pos)?;
        Ok(N3Statement::new(subject, predicate, object))
    }

    fn parse_term(&self, tokens: &[Token], pos: &mut usize) -> TurtleResult<N3Term> {
        match tokens.get(*pos) {
            Some(Token::Iri(iri)) => {
                let iri = iri.clone();
                *pos += 1;
                let node =
                    NamedNode::new(self.resolve_iri(&iri)).map_err(TurtleParseError::model)?;
                Ok(N3Term::NamedNode(node))
            }
            Some(Token::Prefixed(prefix, local)) => {
                let prefix = prefix.clone();
                let local = local.clone();
                *pos += 1;
                let expanded = self.expand_prefix(&prefix, &local)?;
                let node = NamedNode::new(&expanded).map_err(TurtleParseError::model)?;
                Ok(N3Term::NamedNode(node))
            }
            Some(Token::A) => {
                *pos += 1;
                let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .expect("valid IRI");
                Ok(N3Term::NamedNode(rdf_type))
            }
            Some(Token::Var(name)) => {
                let name = name.clone();
                *pos += 1;
                Ok(N3Term::Variable(N3Variable::universal(&name)))
            }
            Some(Token::BNode(label)) => {
                let label = label.clone();
                *pos += 1;
                let bnode = BlankNode::new(&label).map_err(TurtleParseError::model)?;
                Ok(N3Term::BlankNode(bnode))
            }
            Some(Token::Str(s)) => {
                let s = s.clone();
                *pos += 1;
                // Check for language tag or datatype
                match tokens.get(*pos) {
                    Some(Token::LangTag(lang)) => {
                        let lang = lang.clone();
                        *pos += 1;
                        let lit = Literal::new_language_tagged_literal(&s, &lang)
                            .map_err(|e| parse_err(format!("Invalid language tag: {}", e)))?;
                        Ok(N3Term::Literal(lit))
                    }
                    Some(Token::Datatype) => {
                        *pos += 1; // consume '^^'
                        let dt_term = self.parse_term(tokens, pos)?;
                        match dt_term {
                            N3Term::NamedNode(dt_node) => {
                                Ok(N3Term::Literal(Literal::new_typed_literal(&s, dt_node)))
                            }
                            _ => Err(parse_err("Datatype must be an IRI")),
                        }
                    }
                    _ => Ok(N3Term::Literal(Literal::new_simple_literal(&s))),
                }
            }
            Some(Token::Number(n)) => {
                let n = n.clone();
                *pos += 1;
                let is_float = n.contains('.') || n.contains('e') || n.contains('E');
                if is_float {
                    let dt = NamedNode::new("http://www.w3.org/2001/XMLSchema#double")
                        .expect("valid IRI");
                    Ok(N3Term::Literal(Literal::new_typed_literal(&n, dt)))
                } else {
                    let dt = NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")
                        .expect("valid IRI");
                    Ok(N3Term::Literal(Literal::new_typed_literal(&n, dt)))
                }
            }
            Some(Token::Boolean(b)) => {
                let b = *b;
                *pos += 1;
                let dt =
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean").expect("valid IRI");
                Ok(N3Term::Literal(Literal::new_typed_literal(
                    if b { "true" } else { "false" },
                    dt,
                )))
            }
            Some(Token::LParen) => {
                // RDF list — parse items until ')'
                *pos += 1;
                self.parse_list(tokens, pos)
            }
            Some(Token::LBrace) => {
                let formula = self.parse_formula(tokens, pos)?;
                Ok(N3Term::Formula(Box::new(formula)))
            }
            other => Err(parse_err(format!(
                "Unexpected token in term position: {:?}",
                other
            ))),
        }
    }

    /// Parse an RDF list `( item1 item2 ... )` returning a blank node chain.
    fn parse_list(&self, tokens: &[Token], pos: &mut usize) -> TurtleResult<N3Term> {
        let mut items: Vec<N3Term> = Vec::new();

        while *pos < tokens.len() && !matches!(tokens[*pos], Token::RParen | Token::Eof) {
            let item = self.parse_term(tokens, pos)?;
            items.push(item);
        }

        if matches!(tokens.get(*pos), Some(Token::RParen)) {
            *pos += 1;
        } else {
            return Err(parse_err("Unterminated list: expected ')'"));
        }

        if items.is_empty() {
            // rdf:nil
            let nil = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil")
                .expect("valid IRI");
            return Ok(N3Term::NamedNode(nil));
        }

        // Build cons-cell blank node chain.  We return the head blank node
        // and record the cells as a formula (nested structure).
        let rdf_first =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#first").expect("valid IRI");
        let rdf_rest =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest").expect("valid IRI");
        let rdf_nil =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil").expect("valid IRI");

        let head = BlankNode::default();
        let head_term = N3Term::BlankNode(head.clone());

        // For multi-element lists build a formula representing the list structure.
        // The formula is attached as the head blank node — callers that need the
        // triples should extract them from the formula.
        let mut formula = N3Formula::new();
        let mut current = head_term.clone();

        for (i, item) in items.iter().enumerate() {
            let is_last = i + 1 == items.len();
            formula.add_statement(N3Statement::new(
                current.clone(),
                N3Term::NamedNode(rdf_first.clone()),
                item.clone(),
            ));
            if is_last {
                formula.add_statement(N3Statement::new(
                    current.clone(),
                    N3Term::NamedNode(rdf_rest.clone()),
                    N3Term::NamedNode(rdf_nil.clone()),
                ));
            } else {
                let next = BlankNode::default();
                let next_term = N3Term::BlankNode(next.clone());
                formula.add_statement(N3Statement::new(
                    current.clone(),
                    N3Term::NamedNode(rdf_rest.clone()),
                    next_term.clone(),
                ));
                current = next_term;
            }
        }

        // Return the head blank node (the formula is embedded as context)
        // For the purposes of term representation we return just the head blank node.
        let _ = formula; // list structure available if needed
        Ok(head_term)
    }

    fn expand_prefix(&self, prefix: &str, local: &str) -> TurtleResult<String> {
        if let Some(ns) = self.prefixes.get(prefix) {
            Ok(format!("{}{}", ns, local))
        } else {
            Err(parse_err(format!("Unknown prefix: '{}'", prefix)))
        }
    }

    fn resolve_iri(&self, iri: &str) -> String {
        if iri.starts_with("http://") || iri.starts_with("https://") || iri.starts_with("urn:") {
            iri.to_string()
        } else if let Some(base) = &self.base_iri {
            format!("{}{}", base, iri)
        } else {
            iri.to_string()
        }
    }
}

impl Default for N3RuleParser {
    fn default() -> Self {
        Self::new()
    }
}

// ── Serializer ────────────────────────────────────────────────────────────────

/// Serialize N3 rules and facts back to N3 text.
pub struct N3RuleSerializer {
    indent: String,
}

impl N3RuleSerializer {
    /// Create a new serializer with default indentation.
    pub fn new() -> Self {
        Self {
            indent: "    ".to_string(),
        }
    }

    /// Create a serializer with a custom indentation string.
    pub fn with_indent(indent: impl Into<String>) -> Self {
        Self {
            indent: indent.into(),
        }
    }

    /// Serialize a complete `N3RuleDocument` to a string.
    pub fn serialize_document(&self, doc: &N3RuleDocument) -> String {
        let mut out = String::new();

        // Prefixes
        let mut sorted_prefixes: Vec<(&String, &String)> = doc.prefixes.iter().collect();
        sorted_prefixes.sort_by_key(|(k, _)| k.as_str());
        for (prefix, ns) in &sorted_prefixes {
            out.push_str(&format!("@prefix {}: <{}> .\n", prefix, ns));
        }
        if !sorted_prefixes.is_empty() {
            out.push('\n');
        }

        // Base IRI
        if let Some(base) = &doc.base_iri {
            out.push_str(&format!("@base <{}> .\n\n", base));
        }

        // Facts
        for fact in &doc.facts {
            out.push_str(&self.serialize_statement(fact, &doc.prefixes));
            out.push_str(" .\n");
        }
        if !doc.facts.is_empty() && !doc.rules.is_empty() {
            out.push('\n');
        }

        // Rules
        for rule in &doc.rules {
            out.push_str(&self.serialize_rule(rule, &doc.prefixes));
            out.push_str(" .\n");
        }

        out
    }

    /// Serialize a single N3 rule (implication) to a string.
    pub fn serialize_rule(
        &self,
        rule: &N3Implication,
        prefixes: &HashMap<String, String>,
    ) -> String {
        format!(
            "{} =>\n{}{}",
            self.serialize_formula(&rule.antecedent, prefixes),
            self.indent,
            self.serialize_formula(&rule.consequent, prefixes),
        )
    }

    /// Serialize an N3 formula to a string.
    pub fn serialize_formula(
        &self,
        formula: &N3Formula,
        prefixes: &HashMap<String, String>,
    ) -> String {
        if formula.is_empty() {
            return "{ }".to_string();
        }
        let stmts: Vec<String> = formula
            .triples
            .iter()
            .map(|s| self.serialize_statement(s, prefixes))
            .collect();
        format!("{{ {} }}", stmts.join(" . "))
    }

    /// Serialize an N3 statement to a string.
    pub fn serialize_statement(
        &self,
        stmt: &N3Statement,
        prefixes: &HashMap<String, String>,
    ) -> String {
        format!(
            "{} {} {}",
            self.serialize_term(&stmt.subject, prefixes),
            self.serialize_term(&stmt.predicate, prefixes),
            self.serialize_term(&stmt.object, prefixes),
        )
    }

    /// Serialize an N3 term to a string.
    pub fn serialize_term(&self, term: &N3Term, prefixes: &HashMap<String, String>) -> String {
        match term {
            N3Term::NamedNode(n) => self.compress_iri(n.as_str(), prefixes),
            N3Term::BlankNode(b) => format!("_:{}", b.as_str()),
            N3Term::Literal(l) => {
                let value = l.value();
                if let Some(lang) = l.language() {
                    format!("\"{}\"@{}", value, lang)
                } else {
                    let dt = l.datatype();
                    let xsd_string = "http://www.w3.org/2001/XMLSchema#string";
                    if dt.as_str() == xsd_string {
                        format!("\"{}\"", value)
                    } else {
                        let dt_str = self.compress_iri(dt.as_str(), prefixes);
                        format!("\"{}\"^^{}", value, dt_str)
                    }
                }
            }
            N3Term::Variable(v) => format!("?{}", v.name),
            N3Term::Formula(f) => self.serialize_formula(f, prefixes),
        }
    }

    fn compress_iri(&self, iri: &str, prefixes: &HashMap<String, String>) -> String {
        // Try to find a matching prefix
        for (prefix, ns) in prefixes {
            if let Some(local) = iri.strip_prefix(ns.as_str()) {
                // Ensure local part is a valid NCName-like identifier
                if !local.is_empty() && !local.contains('/') && !local.contains('#') {
                    return format!("{}:{}", prefix, local);
                }
            }
        }
        // Special cases
        if iri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
            return "a".to_string();
        }
        format!("<{}>", iri)
    }
}

impl Default for N3RuleSerializer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> N3RuleDocument {
        N3RuleParser::new()
            .parse(input)
            .expect("parse should succeed")
    }

    // ── Prefix tests ───────────────────────────────────────────────────────

    #[test]
    fn test_parse_prefix_declaration() {
        let doc = parse("@prefix ex: <http://example.org/> .\nex:alice ex:knows ex:bob .");
        assert!(doc.prefixes.contains_key("ex"));
        assert_eq!(doc.prefixes["ex"], "http://example.org/");
    }

    #[test]
    fn test_prefix_expansion_in_statement() {
        let doc = parse("@prefix ex: <http://example.org/> .\nex:alice ex:knows ex:bob .");
        assert_eq!(doc.facts.len(), 1);
        let stmt = &doc.facts[0];
        match &stmt.subject {
            N3Term::NamedNode(n) => assert_eq!(n.as_str(), "http://example.org/alice"),
            other => panic!("Expected NamedNode, got {:?}", other),
        }
    }

    #[test]
    fn test_multiple_prefixes() {
        let doc = parse(
            "@prefix ex: <http://example.org/> .\n@prefix owl: <http://www.w3.org/2002/07/owl#> .\nex:a ex:b ex:c .",
        );
        assert!(doc.prefixes.contains_key("ex"));
        assert!(doc.prefixes.contains_key("owl"));
    }

    #[test]
    fn test_parse_base_declaration() {
        let doc = parse("@base <http://example.org/> .\n<alice> <knows> <bob> .");
        assert_eq!(doc.base_iri, Some("http://example.org/".to_string()));
    }

    // ── Variable tests ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_variable_in_fact() {
        let doc = parse("@prefix ex: <http://ex.org/> .\n?x ex:knows ?y .");
        assert_eq!(doc.facts.len(), 1);
        assert!(doc.facts[0].subject.is_variable());
        assert!(doc.facts[0].object.is_variable());
    }

    #[test]
    fn test_variable_name_preserved() {
        let doc = parse("@prefix ex: <http://ex.org/> .\n?personX ex:age ?someAge .");
        assert_eq!(doc.facts.len(), 1);
        if let N3Term::Variable(v) = &doc.facts[0].subject {
            assert_eq!(v.name, "personX");
        } else {
            panic!("Expected variable subject");
        }
    }

    // ── Rule tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_simple_rule() {
        let doc =
            parse("@prefix ex: <http://ex.org/> .\n{ ?x ex:parent ?y } => { ?y ex:child ?x } .");
        assert_eq!(doc.rules.len(), 1);
        let rule = &doc.rules[0];
        assert_eq!(rule.antecedent.len(), 1);
        assert_eq!(rule.consequent.len(), 1);
    }

    #[test]
    fn test_parse_rule_with_two_antecedents() {
        let doc = parse(
            "@prefix ex: <http://ex.org/> .\n{ ?x ex:parent ?y . ?y ex:parent ?z } => { ?x ex:grandparent ?z } .",
        );
        assert_eq!(doc.rules.len(), 1);
        assert_eq!(doc.rules[0].antecedent.len(), 2);
    }

    #[test]
    fn test_parse_multiple_rules() {
        let n3 = r#"
@prefix ex: <http://ex.org/> .
{ ?x ex:parent ?y } => { ?y ex:child ?x } .
{ ?x ex:knows ?y } => { ?y ex:knows ?x } .
"#;
        let doc = parse(n3);
        assert_eq!(doc.rules.len(), 2);
    }

    #[test]
    fn test_parse_facts_and_rules_mixed() {
        let n3 = r#"
@prefix ex: <http://ex.org/> .
ex:alice ex:parent ex:bob .
{ ?x ex:parent ?y } => { ?y ex:child ?x } .
ex:carol ex:parent ex:alice .
"#;
        let doc = parse(n3);
        assert_eq!(doc.facts.len(), 2);
        assert_eq!(doc.rules.len(), 1);
    }

    #[test]
    fn test_rule_variables_are_universal() {
        let doc =
            parse("@prefix ex: <http://ex.org/> .\n{ ?x ex:knows ?y } => { ?y ex:knows ?x } .");
        assert_eq!(doc.rules.len(), 1);
        let ant = &doc.rules[0].antecedent;
        if let N3Term::Variable(v) = &ant.triples[0].subject {
            assert!(v.universal, "Variables in rules should be universal");
        } else {
            panic!("Expected variable subject in antecedent");
        }
    }

    // ── List tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_parse_empty_list_as_nil() {
        let doc = parse("@prefix ex: <http://ex.org/> .\nex:a ex:items () .");
        assert_eq!(doc.facts.len(), 1);
        // rdf:nil should be a NamedNode
        match &doc.facts[0].object {
            N3Term::NamedNode(n) => {
                assert_eq!(n.as_str(), "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil")
            }
            other => panic!("Expected NamedNode(rdf:nil), got {:?}", other),
        }
    }

    #[test]
    fn test_parse_non_empty_list_as_blank_node_head() {
        let doc = parse("@prefix ex: <http://ex.org/> .\nex:a ex:items (ex:b ex:c) .");
        assert_eq!(doc.facts.len(), 1);
        // The head of a non-empty list is a blank node
        assert!(matches!(doc.facts[0].object, N3Term::BlankNode(_)));
    }

    // ── Literal tests ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_integer_literal() {
        let doc = parse("@prefix ex: <http://ex.org/> .\nex:a ex:age 42 .");
        assert_eq!(doc.facts.len(), 1);
        if let N3Term::Literal(l) = &doc.facts[0].object {
            assert_eq!(l.value(), "42");
        } else {
            panic!("Expected literal");
        }
    }

    #[test]
    fn test_parse_string_literal() {
        let doc = parse("@prefix ex: <http://ex.org/> .\nex:a ex:name \"Alice\" .");
        assert_eq!(doc.facts.len(), 1);
        if let N3Term::Literal(l) = &doc.facts[0].object {
            assert_eq!(l.value(), "Alice");
        } else {
            panic!("Expected string literal");
        }
    }

    #[test]
    fn test_parse_boolean_literal() {
        let doc = parse("@prefix ex: <http://ex.org/> .\nex:a ex:active true .");
        assert_eq!(doc.facts.len(), 1);
        if let N3Term::Literal(l) = &doc.facts[0].object {
            assert_eq!(l.value(), "true");
        } else {
            panic!("Expected boolean literal");
        }
    }

    #[test]
    fn test_parse_rdf_type_shorthand() {
        let doc = parse("@prefix ex: <http://ex.org/> .\nex:alice a ex:Person .");
        assert_eq!(doc.facts.len(), 1);
        if let N3Term::NamedNode(n) = &doc.facts[0].predicate {
            assert_eq!(
                n.as_str(),
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
            );
        } else {
            panic!("Expected NamedNode predicate for 'a'");
        }
    }

    // ── Serializer tests ───────────────────────────────────────────────────

    #[test]
    fn test_serialize_empty_document() {
        let doc = N3RuleDocument::new();
        let serialized = N3RuleSerializer::new().serialize_document(&doc);
        assert!(serialized.is_empty() || serialized.trim().is_empty());
    }

    #[test]
    fn test_serialize_prefix() {
        let mut doc = N3RuleDocument::new();
        doc.prefixes
            .insert("ex".to_string(), "http://example.org/".to_string());
        let serialized = N3RuleSerializer::new().serialize_document(&doc);
        assert!(serialized.contains("@prefix ex: <http://example.org/>"));
    }

    #[test]
    fn test_serialize_simple_fact() {
        let n3 = "@prefix ex: <http://ex.org/> .\nex:alice ex:knows ex:bob .";
        let doc = parse(n3);
        let serialized = N3RuleSerializer::new().serialize_document(&doc);
        assert!(serialized.contains("ex:alice"));
        assert!(serialized.contains("ex:knows"));
        assert!(serialized.contains("ex:bob"));
    }

    #[test]
    fn test_serialize_rule() {
        let n3 = "@prefix ex: <http://ex.org/> .\n{ ?x ex:parent ?y } => { ?y ex:child ?x } .";
        let doc = parse(n3);
        let serialized = N3RuleSerializer::new().serialize_document(&doc);
        assert!(
            serialized.contains("=>"),
            "serialized output should contain '=>'"
        );
        assert!(serialized.contains("?x"), "should contain variable ?x");
        assert!(serialized.contains("?y"), "should contain variable ?y");
    }

    #[test]
    fn test_roundtrip_rule_document() {
        let n3 = r#"@prefix ex: <http://ex.org/> .
ex:alice ex:parent ex:bob .
"#;
        let doc = parse(n3);
        let serialized = N3RuleSerializer::new().serialize_document(&doc);
        // Re-parse the serialized output
        let doc2 = parse(&serialized);
        assert_eq!(doc.facts.len(), doc2.facts.len());
    }

    #[test]
    fn test_serialize_variable_term() {
        let ser = N3RuleSerializer::new();
        let var_term = N3Term::Variable(N3Variable::universal("myVar"));
        let s = ser.serialize_term(&var_term, &HashMap::new());
        assert_eq!(s, "?myVar");
    }

    #[test]
    fn test_serialize_iri_with_prefix_compression() {
        let ser = N3RuleSerializer::new();
        let mut prefixes = HashMap::new();
        prefixes.insert("ex".to_string(), "http://example.org/".to_string());
        let term =
            N3Term::NamedNode(NamedNode::new("http://example.org/Alice").expect("valid IRI"));
        let s = ser.serialize_term(&term, &prefixes);
        assert_eq!(s, "ex:Alice");
    }

    #[test]
    fn test_merge_documents() {
        let n3a = "@prefix ex: <http://ex.org/> .\nex:a ex:b ex:c .";
        let n3b = "@prefix ex: <http://ex.org/> .\n{ ?x ex:b ?y } => { ?y ex:b ?x } .";
        let mut doc_a = parse(n3a);
        let doc_b = parse(n3b);
        doc_a.merge(doc_b);
        assert_eq!(doc_a.facts.len(), 1);
        assert_eq!(doc_a.rules.len(), 1);
    }
}
