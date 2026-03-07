//! # QueryParser - parsing Methods
//!
//! This module contains method implementations for `QueryParser`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::algebra::{Algebra, PropertyPath, PropertyPathPattern, Term, TriplePattern, Variable};
use anyhow::{anyhow, bail, Result};
use oxirs_core::model::NamedNode;
use std::collections::{HashMap, HashSet};

use super::types::{DatasetClause, Query, QueryType, Token};

use super::queryparser_type::QueryParser;

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
                    tokens.push(Token::StringLiteral(literal));
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
        if self.match_token(&Token::Multiply) {
        } else {
            while !self.is_at_end()
                && !matches!(self.peek(), Some(Token::Where) | Some(Token::From))
            {
                if let Some(Token::Variable(var)) = self.peek() {
                    query.select_variables.push(Variable::new(var.clone())?);
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.parse_dataset_clause(&mut query.dataset)?;
        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        query.where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;
        self.parse_solution_modifiers(query)?;
        Ok(())
    }
    pub(super) fn parse_describe_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Describe)?;
        while !self.is_at_end() && !matches!(self.peek(), Some(Token::Where) | Some(Token::From)) {
            if let Some(Token::Variable(var)) = self.peek() {
                query.select_variables.push(Variable::new(var.clone())?);
                self.advance();
            } else if matches!(
                self.peek(),
                Some(Token::Iri(_)) | Some(Token::PrefixedName(_, _))
            ) {
                self.advance();
            } else {
                break;
            }
        }
        self.parse_dataset_clause(&mut query.dataset)?;
        if self.match_token(&Token::Where) {
            self.expect_token(Token::LeftBrace)?;
            query.where_clause = self.parse_group_graph_pattern()?;
            self.expect_token(Token::RightBrace)?;
        }
        Ok(())
    }
    pub(super) fn parse_group_graph_pattern(&mut self) -> Result<Algebra> {
        if self.has_union_pattern() {
            return self.parse_graph_pattern_or_union();
        }
        let mut patterns = Vec::new();
        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RightBrace)) {
            self.skip_whitespace_and_newlines();
            if self.is_at_end() || matches!(self.peek(), Some(Token::RightBrace)) {
                break;
            }
            let pattern = self.parse_graph_pattern_or_union()?;
            patterns.push(pattern);
            self.match_token(&Token::Dot);
            self.skip_whitespace_and_newlines();
        }
        if patterns.is_empty() {
            Ok(Algebra::Table)
        } else if patterns.len() == 1 {
            Ok(patterns
                .into_iter()
                .next()
                .expect("collection validated to be non-empty"))
        } else {
            let mut patterns_iter = patterns.into_iter();
            let mut result = patterns_iter
                .next()
                .expect("collection validated to be non-empty");
            for pattern in patterns_iter {
                result = Algebra::join(result, pattern);
            }
            Ok(result)
        }
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
            let triple = self.parse_triple_pattern()?;
            triples.push(triple);
            if !self.match_token(&Token::Dot) {
                break;
            }
        }
        Ok(Algebra::Bgp(triples))
    }
    pub(super) fn parse_triple_pattern(&mut self) -> Result<TriplePattern> {
        self.skip_whitespace_and_newlines();
        let subject = self.parse_term()?;
        self.skip_whitespace_and_newlines();
        if self.is_property_path_start() {
            let path = self.parse_property_path()?;
            self.skip_whitespace_and_newlines();
            let object = self.parse_term()?;
            let path_pattern = PropertyPathPattern::new(subject, path, object);
            return Ok(TriplePattern::new(
                path_pattern.subject.clone(),
                Term::PropertyPath(path_pattern.path.clone()),
                path_pattern.object.clone(),
            ));
        }
        let predicate = self.parse_term()?;
        self.skip_whitespace_and_newlines();
        let object = self.parse_term()?;
        Ok(TriplePattern::new(subject, predicate, object))
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
