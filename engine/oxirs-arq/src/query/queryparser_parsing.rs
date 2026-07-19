//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Algebra, PropertyPath, Term, TriplePattern, Variable};
use anyhow::{anyhow, bail, Result};
use oxirs_core::model::NamedNode;
use std::collections::{HashMap, HashSet};

use super::types::{DatasetClause, DescribeTarget, ProjectionItem, Query, QueryType, Token};

use super::queryparser_type::QueryParser;

/// Compose the next group-graph-pattern element onto the accumulated pattern.
///
/// `OPTIONAL`, `MINUS` and a leading `UNION` are each parsed with the unit table
/// (`Algebra::Table`) as their left operand, because a group element is written
/// standalone. At the group level, however, they operate on the pattern that
/// PRECEDES them, so they must be re-rooted on the accumulated `prev` rather
/// than joined as an independent unit:
///
/// * `Join(prev, Minus(Table, r))` makes `MINUS` a silent no-op — a `Table` left
///   shares no variables with `r`, and SPARQL `MINUS` removes nothing when the
///   operands are variable-disjoint. Re-rooting to `Minus(prev, r)` restores the
///   difference.
/// * `Join(prev, LeftJoin(Table, r))` collapses `OPTIONAL` into an inner join
///   (dropping `prev` rows without an `r` match). Re-rooting to
///   `LeftJoin(prev, r)` restores proper optional semantics.
fn compose_group_element(prev: Algebra, element: Algebra) -> Algebra {
    match element {
        Algebra::LeftJoin {
            left,
            right,
            filter,
        } if matches!(*left, Algebra::Table) => Algebra::LeftJoin {
            left: Box::new(prev),
            right,
            filter,
        },
        Algebra::Minus { left, right } if matches!(*left, Algebra::Table) => Algebra::Minus {
            left: Box::new(prev),
            right,
        },
        other => Algebra::join(prev, other),
    }
}

impl QueryParser {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            position: 0,
            prefixes: HashMap::new(),
            base_iri: None,
            variables: HashSet::new(),
            blank_node_counter: 0,
        }
    }
    /// Tokenize SPARQL query string
    pub(super) fn tokenize(&mut self, input: &str) -> Result<()> {
        let mut chars = input.chars().peekable();
        let mut tokens = Vec::new();
        while let Some(&ch) = chars.peek() {
            match ch {
                ' ' | '\t' | '\r' => {
                    chars.next();
                }
                '\n' => {
                    chars.next();
                    tokens.push(Token::Newline);
                }
                '#' => {
                    while let Some(&ch) = chars.peek() {
                        chars.next();
                        if ch == '\n' {
                            tokens.push(Token::Newline);
                            break;
                        }
                    }
                }
                '(' => {
                    chars.next();
                    tokens.push(Token::LeftParen);
                }
                ')' => {
                    chars.next();
                    tokens.push(Token::RightParen);
                }
                '{' => {
                    chars.next();
                    tokens.push(Token::LeftBrace);
                }
                '}' => {
                    chars.next();
                    tokens.push(Token::RightBrace);
                }
                '[' => {
                    chars.next();
                    tokens.push(Token::LeftBracket);
                }
                ']' => {
                    chars.next();
                    tokens.push(Token::RightBracket);
                }
                '.' => {
                    chars.next();
                    tokens.push(Token::Dot);
                }
                ';' => {
                    chars.next();
                    tokens.push(Token::Semicolon);
                }
                ',' => {
                    chars.next();
                    tokens.push(Token::Comma);
                }
                ':' => {
                    chars.next();
                    if chars
                        .peek()
                        .map_or(true, |c| !c.is_ascii_alphanumeric() && *c != '_')
                    {
                        tokens.push(Token::Colon);
                    } else {
                        let mut id = ":".to_string();
                        id.push_str(&self.parse_identifier(&mut chars));
                        tokens.push(self.classify_identifier(&id));
                    }
                }
                '=' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Equal);
                    } else {
                        tokens.push(Token::Equal);
                    }
                }
                '<' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::LessEqual);
                    } else if chars.peek() == Some(&'h') || chars.peek() == Some(&'/') {
                        let mut iri = String::new();
                        while let Some(&ch) = chars.peek() {
                            if ch == '>' {
                                chars.next();
                                break;
                            }
                            iri.push(ch);
                            chars.next();
                        }
                        tokens.push(Token::Iri(iri));
                    } else {
                        tokens.push(Token::Less);
                    }
                }
                '>' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::GreaterEqual);
                    } else {
                        tokens.push(Token::Greater);
                    }
                }
                '+' => {
                    chars.next();
                    tokens.push(Token::Plus);
                }
                '-' => {
                    chars.next();
                    tokens.push(Token::Minus_);
                }
                '/' => {
                    chars.next();
                    tokens.push(Token::Divide);
                }
                '|' => {
                    chars.next();
                    if chars.peek() == Some(&'|') {
                        chars.next();
                        tokens.push(Token::Or);
                    } else {
                        tokens.push(Token::Pipe);
                    }
                }
                '^' => {
                    chars.next();
                    tokens.push(Token::Caret);
                }
                '?' => {
                    chars.next();
                    if chars.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                        let var = self.parse_identifier(&mut chars);
                        tokens.push(Token::Variable(var));
                    } else {
                        tokens.push(Token::Question);
                    }
                    continue;
                }
                '*' => {
                    chars.next();
                    tokens.push(Token::Star);
                }
                '!' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::NotEqual);
                    } else {
                        tokens.push(Token::Bang);
                    }
                    continue;
                }
                '&' => {
                    chars.next();
                    if chars.peek() == Some(&'&') {
                        chars.next();
                        tokens.push(Token::And);
                    } else {
                        return Err(anyhow!("Unexpected '&' - did you mean '&&'?"));
                    }
                    continue;
                }
                '$' => {
                    chars.next();
                    let var = self.parse_identifier(&mut chars);
                    tokens.push(Token::Variable(var));
                }
                '"' | '\'' => {
                    let quote = ch;
                    chars.next();
                    let mut literal = String::new();
                    while let Some(&ch) = chars.peek() {
                        chars.next();
                        if ch == quote {
                            break;
                        }
                        if ch == '\\' {
                            if let Some(&escaped) = chars.peek() {
                                chars.next();
                                match escaped {
                                    'n' => literal.push('\n'),
                                    't' => literal.push('\t'),
                                    'r' => literal.push('\r'),
                                    '\\' => literal.push('\\'),
                                    '\'' => literal.push('\''),
                                    '"' => literal.push('"'),
                                    _ => {
                                        literal.push('\\');
                                        literal.push(escaped);
                                    }
                                }
                            }
                        } else {
                            literal.push(ch);
                        }
                    }
                    // Optional RDF literal suffix: a language tag (`@ja`, with
                    // subtags such as `@ja-JP`) or an explicit datatype
                    // (`^^<iri>` / `^^prefix:local`). A plain literal keeps the
                    // simple `StringLiteral` token so existing call sites are
                    // unaffected.
                    if chars.peek() == Some(&'@') {
                        chars.next();
                        let mut lang = String::new();
                        while let Some(&c) = chars.peek() {
                            if c.is_ascii_alphanumeric() || c == '-' {
                                lang.push(c);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        tokens.push(Token::RdfLiteral {
                            value: literal,
                            language: Some(lang),
                            datatype: None,
                        });
                    } else if chars.peek() == Some(&'^') {
                        chars.next();
                        if chars.peek() == Some(&'^') {
                            chars.next();
                            let datatype = self.parse_datatype_iri(&mut chars);
                            tokens.push(Token::RdfLiteral {
                                value: literal,
                                language: None,
                                datatype: Some(datatype),
                            });
                        } else {
                            // A lone `^` after a string is not a datatype marker;
                            // keep the literal and emit the caret separately.
                            tokens.push(Token::StringLiteral(literal));
                            tokens.push(Token::Caret);
                        }
                    } else {
                        tokens.push(Token::StringLiteral(literal));
                    }
                }
                '_' => {
                    chars.next();
                    if chars.peek() == Some(&':') {
                        chars.next();
                        let id = self.parse_identifier(&mut chars);
                        tokens.push(Token::BlankNode(id));
                    } else {
                        let mut id = "_".to_string();
                        id.push_str(&self.parse_identifier(&mut chars));
                        tokens.push(self.classify_identifier(&id));
                    }
                }
                _ if ch.is_ascii_alphabetic() || ch == '_' => {
                    let identifier = self.parse_identifier(&mut chars);
                    tokens.push(self.classify_identifier(&identifier));
                }
                _ if ch.is_ascii_digit() => {
                    let number = self.parse_number(&mut chars);
                    tokens.push(Token::NumericLiteral(number));
                }
                _ => {
                    chars.next();
                }
            }
        }
        tokens.push(Token::Eof);
        self.tokens = tokens;
        self.position = 0;
        Ok(())
    }
    pub(super) fn parse_identifier(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> String {
        let mut identifier = String::new();
        let mut found_colon = false;
        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                identifier.push(ch);
                chars.next();
            } else if ch == ':' && !found_colon {
                identifier.push(ch);
                chars.next();
                found_colon = true;
            } else {
                break;
            }
        }
        identifier
    }
    /// Read the datatype that follows a `^^` marker: either an absolute IRI in
    /// angle brackets (`<iri>`, returned without the brackets) or a
    /// `prefix:local` name (returned verbatim for parse-time resolution).
    pub(super) fn parse_datatype_iri(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
    ) -> String {
        if chars.peek() == Some(&'<') {
            chars.next();
            let mut iri = String::new();
            while let Some(&c) = chars.peek() {
                chars.next();
                if c == '>' {
                    break;
                }
                iri.push(c);
            }
            iri
        } else {
            self.parse_identifier(chars)
        }
    }
    pub(super) fn parse_number(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
        let mut number = String::new();
        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == 'E' || ch == '+' || ch == '-'
            {
                number.push(ch);
                chars.next();
            } else {
                break;
            }
        }
        number
    }
    pub(super) fn parse_query(&mut self) -> Result<Query> {
        let mut query = Query {
            query_type: QueryType::Select,
            select_variables: Vec::new(),
            where_clause: Algebra::Zero,
            order_by: Vec::new(),
            group_by: Vec::new(),
            having: None,
            limit: None,
            offset: None,
            distinct: false,
            reduced: false,
            construct_template: Vec::new(),
            prefixes: HashMap::new(),
            base_iri: None,
            dataset: DatasetClause::default(),
            projection_items: Vec::new(),
            describe_targets: Vec::new(),
            describe_all: false,
        };
        self.skip_whitespace();
        self.parse_prologue(&mut query)?;
        self.skip_whitespace();
        match self.peek() {
            Some(Token::Select) => {
                query.query_type = QueryType::Select;
                self.parse_select_query(&mut query)?;
            }
            Some(Token::Construct) => {
                query.query_type = QueryType::Construct;
                self.parse_construct_query(&mut query)?;
            }
            Some(Token::Ask) => {
                query.query_type = QueryType::Ask;
                self.parse_ask_query(&mut query)?;
            }
            Some(Token::Describe) => {
                query.query_type = QueryType::Describe;
                self.parse_describe_query(&mut query)?;
            }
            _ => bail!("Expected query type (SELECT, CONSTRUCT, ASK, DESCRIBE)"),
        }
        Ok(query)
    }
    pub(super) fn parse_prologue(&mut self, query: &mut Query) -> Result<()> {
        while let Some(token) = self.peek() {
            match token {
                Token::Prefix => {
                    self.advance();
                    let prefix = match self.peek() {
                        Some(Token::PrefixedName(prefix, local)) => {
                            if prefix.is_empty() && local.is_empty() {
                                self.advance();
                                String::new()
                            } else {
                                let p = prefix.clone();
                                self.advance();
                                p
                            }
                        }
                        Some(Token::Colon) => {
                            self.advance();
                            String::new()
                        }
                        _ => {
                            eprintln!("Debug: Got token: {:?}", self.peek());
                            bail!("Expected prefix name or colon after PREFIX")
                        }
                    };
                    let iri = self.expect_iri()?;
                    query.prefixes.insert(prefix.clone(), iri.clone());
                    self.prefixes.insert(prefix, iri);
                }
                Token::Base => {
                    self.advance();
                    let iri = self.expect_iri()?;
                    query.base_iri = Some(iri.clone());
                    self.base_iri = Some(iri);
                }
                Token::Newline => {
                    self.advance();
                }
                _ => break,
            }
        }
        Ok(())
    }
    pub(super) fn parse_select_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Select)?;
        if self.match_token(&Token::Distinct) {
            query.distinct = true;
        } else if self.match_token(&Token::Reduced) {
            query.reduced = true;
        }
        // `SELECT *`: the tokenizer emits `Token::Star` for `*`, so the legacy
        // `Token::Multiply` check never matched and `SELECT *` failed to parse.
        // Accept either. An empty `select_variables` (and empty
        // `projection_items`) means "project all in-scope variables".
        if self.match_token(&Token::Star) || self.match_token(&Token::Multiply) {
        } else {
            while !self.is_at_end()
                && !matches!(self.peek(), Some(Token::Where) | Some(Token::From))
            {
                match self.peek() {
                    Some(Token::Variable(var)) => {
                        let variable = Variable::new(var.clone())?;
                        self.advance();
                        query.select_variables.push(variable.clone());
                        query
                            .projection_items
                            .push(ProjectionItem::Variable(variable));
                    }
                    // A parenthesized projection: `( Expression AS ?var )`,
                    // including SPARQL aggregates such as `(COUNT(*) AS ?n)`.
                    Some(Token::LeftParen) => {
                        let item = self.parse_projection_paren_item()?;
                        // The projected output variable is the alias; record it
                        // in `select_variables` too so the output column set is
                        // complete regardless of which field a consumer reads.
                        let alias = match &item {
                            ProjectionItem::Variable(v) => v.clone(),
                            ProjectionItem::Expression { alias, .. }
                            | ProjectionItem::Aggregate { alias, .. } => alias.clone(),
                        };
                        query.select_variables.push(alias);
                        query.projection_items.push(item);
                    }
                    _ => break,
                }
            }
        }
        self.parse_dataset_clause(&mut query.dataset)?;
        // The `WHERE` keyword is optional in a SPARQL 1.1 SELECT
        // (`SELECT ?s { … }` and `SELECT * { … }` are both valid); only the
        // group-graph-pattern braces are required. Mirror the ASK handling
        // (`match_token`) rather than demanding the keyword. The projection loop
        // above already stops on `{` (`Token::LeftBrace` hits its `_ => break`),
        // so the brace that opens the pattern is still available here.
        self.match_token(&Token::Where);
        self.expect_token(Token::LeftBrace)?;
        query.where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        self.parse_solution_modifiers(query)?;
        Ok(())
    }
    pub(super) fn parse_describe_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Describe)?;
        self.skip_whitespace_and_newlines();
        // `DESCRIBE *` describes every in-scope variable binding; it takes no
        // explicit target list.
        if self.match_token(&Token::Star) || self.match_token(&Token::Multiply) {
            query.describe_all = true;
        } else {
            // `DESCRIBE VarOrIri+`: one or more IRIs and/or variables. Prefixed
            // names are expanded against the prologue exactly like other term
            // positions; the raw targets are RETAINED in `describe_targets`.
            loop {
                self.skip_whitespace_and_newlines();
                match self.peek() {
                    Some(Token::Variable(var)) => {
                        let variable = Variable::new(var.clone())?;
                        self.advance();
                        query
                            .describe_targets
                            .push(DescribeTarget::Variable(variable.clone()));
                        query.select_variables.push(variable);
                    }
                    Some(Token::Iri(iri)) => {
                        let iri = iri.clone();
                        self.advance();
                        query
                            .describe_targets
                            .push(DescribeTarget::Iri(NamedNode::new_unchecked(iri)));
                    }
                    Some(Token::PrefixedName(prefix, local)) => {
                        let prefix = prefix.clone();
                        let local = local.clone();
                        self.advance();
                        let full_iri = self.resolve_prefixed_name(&prefix, &local)?;
                        query
                            .describe_targets
                            .push(DescribeTarget::Iri(NamedNode::new_unchecked(full_iri)));
                    }
                    _ => break,
                }
            }
            if query.describe_targets.is_empty() {
                bail!("DESCRIBE requires at least one IRI or variable target, or '*'");
            }
        }
        self.parse_dataset_clause(&mut query.dataset)?;
        // `WhereClause ::= 'WHERE'? GroupGraphPattern` — the WHERE keyword is
        // optional, so accept `DESCRIBE ?x { … }` as well as
        // `DESCRIBE ?x WHERE { … }`.
        if self.match_token(&Token::Where) {
            self.expect_token(Token::LeftBrace)?;
            query.where_clause = self.parse_group_graph_pattern()?;
            self.expect_token(Token::RightBrace)?;
        } else if self.match_token(&Token::LeftBrace) {
            query.where_clause = self.parse_group_graph_pattern()?;
            self.expect_token(Token::RightBrace)?;
        }
        self.parse_solution_modifiers(query)?;
        Ok(())
    }
    pub(super) fn parse_group_graph_pattern(&mut self) -> Result<Algebra> {
        // A group graph pattern is a sequence of graph patterns interleaved with
        // `FILTER` / `BIND` modifiers, and the two modifiers scope DIFFERENTLY:
        //
        //   * `FILTER` constrains the WHOLE group regardless of its textual
        //     position, so bare filters are collected and applied once, wrapping
        //     the fully-joined group (SPARQL 1.1 §18.2.2 "collect FILTERs").
        //
        //   * `BIND` is POSITIONAL: `BIND(expr AS ?v)` extends the solution
        //     produced by the elements written BEFORE it, and elements written
        //     AFTER it join against the extended solution. `{ ?s ?p ?o .
        //     BIND(?o AS ?x) . ?x ?q ?r }` therefore means
        //     `Join(Extend(BGP(?s ?p ?o), ?x, ?o), BGP(?x ?q ?r))`. Deferring
        //     the `BIND` to the group end (as `FILTER` is deferred) would join
        //     both BGPs first and only then extend, letting the second BGP bind
        //     `?x` and be silently mis-joined/overwritten. A leading `BIND`
        //     extends the unit table (join identity), so `{ BIND(1 AS ?v) … }`
        //     still works.
        //
        // Modifiers are recognised SYNTACTICALLY, by the leading `FILTER` /
        // `BIND` token of THIS group — never structurally by matching a
        // `Filter/Extend { pattern: Table, .. }` shape. A nested single-element
        // group such as `{ ?s ?p ?o { FILTER(?o > 5) } }` also parses to
        // `Filter { pattern: Table, .. }`, but it is a JOIN operand of the outer
        // group, not an outer-group modifier, and must not be re-scoped.
        //
        // A top-level `UNION` needs no special-casing: it is parsed by
        // `parse_graph_pattern_or_union` as one element of the loop below, so a
        // trailing `FILTER`/`BIND` after a union is still scoped to the group.
        let mut acc: Option<Algebra> = None;
        let mut filters: Vec<Algebra> = Vec::new();
        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RightBrace)) {
            self.skip_whitespace_and_newlines();
            if self.is_at_end() || matches!(self.peek(), Some(Token::RightBrace)) {
                break;
            }
            if matches!(self.peek(), Some(Token::Filter)) {
                // Bare FILTER: whole-group scope, deferred to the group end.
                filters.push(self.parse_filter_pattern()?);
            } else if matches!(self.peek(), Some(Token::Bind)) {
                // Bare BIND: positional Extend over the algebra accumulated so
                // far (the unit table when the BIND leads the group).
                match self.parse_bind_pattern()? {
                    Algebra::Extend { variable, expr, .. } => {
                        let base = acc.take().unwrap_or(Algebra::Table);
                        acc = Some(Algebra::Extend {
                            pattern: Box::new(base),
                            variable,
                            expr,
                        });
                    }
                    other => bail!("BIND parser returned unexpected algebra: {other:?}"),
                }
            } else {
                let pattern = self.parse_graph_pattern_or_union()?;
                acc = Some(match acc.take() {
                    Some(prev) => compose_group_element(prev, pattern),
                    None => pattern,
                });
            }
            self.match_token(&Token::Dot);
            self.skip_whitespace_and_newlines();
        }
        let mut result = acc.unwrap_or(Algebra::Table);
        for modifier in filters {
            match modifier {
                Algebra::Filter { condition, .. } => {
                    result = Algebra::Filter {
                        pattern: Box::new(result),
                        condition,
                    };
                }
                other => bail!("FILTER parser returned unexpected algebra: {other:?}"),
            }
        }
        Ok(result)
    }
    pub(super) fn parse_graph_pattern_or_union(&mut self) -> Result<Algebra> {
        let left = self.parse_graph_pattern()?;
        self.skip_whitespace_and_newlines();
        if self.match_token(&Token::Union) {
            self.skip_whitespace_and_newlines();
            let right = self.parse_graph_pattern_or_union()?;
            return Ok(Algebra::Union {
                left: Box::new(left),
                right: Box::new(right),
            });
        }
        Ok(left)
    }
    pub(super) fn parse_basic_graph_pattern(&mut self) -> Result<Algebra> {
        let mut triples = Vec::new();
        while !self.is_at_end() {
            self.skip_whitespace_and_newlines();
            if self.is_pattern_end() {
                break;
            }
            if matches!(self.peek(), Some(Token::Newline)) {
                self.advance();
                continue;
            }
            triples.extend(self.parse_triples_same_subject()?);
            if !self.match_token(&Token::Dot) {
                break;
            }
        }
        Ok(Algebra::Bgp(triples))
    }
    /// Parse a `TriplesSameSubjectPath`: one subject followed by a
    /// predicate-object list, expanding the SPARQL 1.1 abbreviations `;`
    /// (predicate-object list) and `,` (object list) into every triple that
    /// shares the subject. `{ ?s :p ?o ; :q ?r , ?t }` therefore yields the
    /// three triples `(?s :p ?o)`, `(?s :q ?r)`, `(?s :q ?t)`.
    pub(super) fn parse_triples_same_subject(&mut self) -> Result<Vec<TriplePattern>> {
        self.skip_whitespace_and_newlines();
        let subject = self.parse_term()?;
        let mut triples = Vec::new();
        self.parse_predicate_object_list(&subject, &mut triples)?;
        Ok(triples)
    }
    /// Parse a `PropertyListPathNotEmpty`:
    /// `verb objectList ( ';' ( verb objectList )? )*`. A trailing `;` (and a
    /// repeated `;;`) with no following verb is valid SPARQL and tolerated.
    pub(super) fn parse_predicate_object_list(
        &mut self,
        subject: &Term,
        out: &mut Vec<TriplePattern>,
    ) -> Result<()> {
        loop {
            self.skip_whitespace_and_newlines();
            let predicate = self.parse_verb()?;
            self.parse_object_list(subject, &predicate, out)?;
            self.skip_whitespace_and_newlines();
            if !self.match_token(&Token::Semicolon) {
                break;
            }
            // After ';', a further `verb objectList` is optional. Skip any run of
            // extra `;` and stop when no verb follows (a trailing semicolon).
            self.skip_whitespace_and_newlines();
            while self.match_token(&Token::Semicolon) {
                self.skip_whitespace_and_newlines();
            }
            if !self.is_verb_start() {
                break;
            }
        }
        Ok(())
    }
    /// Parse an `ObjectListPath`: one or more objects separated by `,`, emitting
    /// one triple per object with the shared subject and predicate.
    pub(super) fn parse_object_list(
        &mut self,
        subject: &Term,
        predicate: &Term,
        out: &mut Vec<TriplePattern>,
    ) -> Result<()> {
        loop {
            self.skip_whitespace_and_newlines();
            let object = self.parse_term()?;
            out.push(TriplePattern::new(
                subject.clone(),
                predicate.clone(),
                object,
            ));
            self.skip_whitespace_and_newlines();
            if !self.match_token(&Token::Comma) {
                break;
            }
        }
        Ok(())
    }
    /// Parse a verb (predicate): a property path (`IRI`, `a`, `p1/p2`, `p+`, …)
    /// or a plain variable predicate.
    pub(super) fn parse_verb(&mut self) -> Result<Term> {
        if self.is_property_path_start() {
            Ok(Term::PropertyPath(self.parse_property_path()?))
        } else {
            self.parse_term()
        }
    }
    /// Whether the current token can begin a verb (predicate): a property-path
    /// start (`IRI` / prefixed name / `a` / `^` / `(` / `!`) or a variable.
    pub(super) fn is_verb_start(&self) -> bool {
        self.is_property_path_start() || matches!(self.peek(), Some(Token::Variable(_)))
    }
    /// Parse primary property path expressions
    pub(super) fn parse_property_path_primary(&mut self) -> Result<PropertyPath> {
        match self.peek() {
            Some(Token::Caret) => {
                self.advance();
                let path = self.parse_property_path_primary()?;
                Ok(PropertyPath::inverse(path))
            }
            Some(Token::Iri(iri)) => {
                let iri = iri.clone();
                self.advance();
                Ok(PropertyPath::iri(NamedNode::new_unchecked(iri)))
            }
            Some(Token::PrefixedName(prefix, local)) => {
                let prefix = prefix.clone();
                let local = local.clone();
                self.advance();
                let full_iri = self.resolve_prefixed_name(&prefix, &local)?;
                Ok(PropertyPath::iri(NamedNode::new_unchecked(full_iri)))
            }
            Some(Token::A) => {
                // `a` is the SPARQL shorthand for the rdf:type predicate.
                self.advance();
                Ok(PropertyPath::iri(NamedNode::new_unchecked(
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                )))
            }
            Some(Token::Variable(var)) => {
                let var = var.clone();
                self.advance();
                Ok(PropertyPath::Variable(Variable::new(var)?))
            }
            Some(Token::LeftParen) => {
                self.advance();
                let path = self.parse_property_path()?;
                self.expect_token(Token::RightParen)?;
                Ok(path)
            }
            Some(Token::Bang) => {
                self.advance();
                self.expect_token(Token::LeftParen)?;
                let mut negated_paths = Vec::new();
                loop {
                    negated_paths.push(self.parse_property_path_primary()?);
                    if !self.match_token(&Token::Pipe) {
                        break;
                    }
                }
                self.expect_token(Token::RightParen)?;
                Ok(PropertyPath::NegatedPropertySet(negated_paths))
            }
            _ => bail!("Expected property path expression"),
        }
    }
}
