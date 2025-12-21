//! SPARQL Query Parser and AST
//!
//! This module provides comprehensive SPARQL 1.1/1.2 query parsing capabilities,
//! converting SPARQL query strings into algebraic expressions for execution.
//!
//! ## File Organization (2383 lines)
//!
//! This file contains a large `impl QueryParser` block (1958 lines) that implements
//! a hand-written recursive descent parser for SPARQL 1.1/1.2. The impl block is
//! intentionally large due to:
//!
//! 1. **Parser Cohesion**: All parsing logic is tightly coupled and shares state
//! 2. **Recursive Grammar**: SPARQL grammar requires many mutually recursive methods
//! 3. **Lookahead Logic**: Complex lookahead and backtracking require shared position tracking
//! 4. **Performance**: Inlining and method locality provide better parsing performance
//!
//! ### Future Refactoring Considerations:
//!
//! Manual refactoring (not SplitRS) would be required due to complex method interdependencies.
//! Potential structure:
//! - `query/parser/core.rs` - Core parser infrastructure and state
//! - `query/parser/expressions.rs` - Expression parsing (logical, relational, arithmetic)
//! - `query/parser/patterns.rs` - Graph pattern parsing (BGP, OPTIONAL, UNION)
//! - `query/parser/property_paths.rs` - Property path parsing
//! - `query/parser/aggregates.rs` - Aggregation and grouping
//! - `query/parser/update.rs` - UPDATE operation parsing

use crate::algebra::{
    Algebra, BinaryOperator, Expression, GroupCondition, Iri, Literal, OrderCondition,
    PropertyPath, PropertyPathPattern, Term, TriplePattern, UnaryOperator, Variable,
};
use crate::update::{GraphReference, GraphTarget, QuadPattern, UpdateOperation};
use anyhow::{anyhow, bail, Result};
use oxirs_core::model::NamedNode;
use std::collections::{HashMap, HashSet};
use std::fmt;

/// SPARQL query types
#[derive(Debug, Clone, PartialEq)]
pub enum QueryType {
    Select,
    Construct,
    Ask,
    Describe,
}

/// SPARQL query representation
#[derive(Debug, Clone)]
pub struct Query {
    pub query_type: QueryType,
    pub select_variables: Vec<Variable>,
    pub where_clause: Algebra,
    pub order_by: Vec<OrderCondition>,
    pub group_by: Vec<GroupCondition>,
    pub having: Option<Expression>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub distinct: bool,
    pub reduced: bool,
    pub construct_template: Vec<TriplePattern>,
    pub prefixes: HashMap<String, String>,
    pub base_iri: Option<String>,
    pub dataset: DatasetClause,
}

impl fmt::Display for Query {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Basic SPARQL query serialization
        match self.query_type {
            QueryType::Select => {
                write!(f, "SELECT ")?;
                if self.distinct {
                    write!(f, "DISTINCT ")?;
                } else if self.reduced {
                    write!(f, "REDUCED ")?;
                }

                if self.select_variables.is_empty() {
                    write!(f, "* ")?;
                } else {
                    for (i, var) in self.select_variables.iter().enumerate() {
                        if i > 0 {
                            write!(f, " ")?;
                        }
                        write!(f, "?{var}", var = var.as_str())?;
                    }
                    write!(f, " ")?;
                }

                write!(f, "WHERE {{ {:?} }}", self.where_clause)?;
            }
            QueryType::Construct => {
                write!(f, "CONSTRUCT {{ ")?;
                for (i, pattern) in self.construct_template.iter().enumerate() {
                    if i > 0 {
                        write!(f, " . ")?;
                    }
                    write!(f, "{pattern:?}")?;
                }
                write!(f, " }} WHERE {{ {:?} }}", self.where_clause)?;
            }
            QueryType::Ask => {
                write!(f, "ASK WHERE {{ {:?} }}", self.where_clause)?;
            }
            QueryType::Describe => {
                write!(f, "DESCRIBE ")?;
                if self.select_variables.is_empty() {
                    write!(f, "* ")?;
                } else {
                    for (i, var) in self.select_variables.iter().enumerate() {
                        if i > 0 {
                            write!(f, " ")?;
                        }
                        write!(f, "?{var}", var = var.as_str())?;
                    }
                    write!(f, " ")?;
                }
                write!(f, "WHERE {{ {:?} }}", self.where_clause)?;
            }
        }

        if let Some(limit) = self.limit {
            write!(f, " LIMIT {limit}")?;
        }
        if let Some(offset) = self.offset {
            write!(f, " OFFSET {offset}")?;
        }

        Ok(())
    }
}

/// Dataset clause for FROM and FROM NAMED
#[derive(Debug, Clone, Default)]
pub struct DatasetClause {
    pub default_graphs: Vec<Iri>,
    pub named_graphs: Vec<Iri>,
}

/// SPARQL UPDATE request representation
#[derive(Debug, Clone)]
pub struct UpdateRequest {
    pub operations: Vec<UpdateOperation>,
    pub prefixes: HashMap<String, String>,
    pub base_iri: Option<String>,
}

/// Query parser implementation
pub struct QueryParser {
    tokens: Vec<Token>,
    position: usize,
    prefixes: HashMap<String, String>,
    base_iri: Option<String>,
    variables: HashSet<Variable>,
    #[allow(dead_code)]
    blank_node_counter: usize,
}

/// Token types for SPARQL parsing
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Select,
    Construct,
    Ask,
    Describe,
    Where,
    Optional,
    Union,
    Minus,
    Filter,
    Bind,
    Service,
    Graph,
    From,
    Named,
    Prefix,
    Base,
    Distinct,
    Reduced,
    OrderBy,
    GroupBy,
    Having,
    Limit,
    Offset,
    Asc,
    Desc,
    As,
    Values,
    Exists,
    NotExists,

    // UPDATE Keywords
    Insert,
    Delete,
    Update,
    Create,
    Drop,
    Clear,
    Load,
    Copy,
    Move,
    Add,
    Data,
    With,
    Using,
    Silent,
    All,
    Default,
    To,

    // Operators
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Not,
    Plus,
    Minus_,
    Multiply,
    Divide,

    // Property Path Operators
    Pipe,     // |
    Caret,    // ^
    Slash,    // /
    Question, // ?
    Star,     // *
    Bang,     // !

    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Dot,
    Semicolon,
    Comma,
    Colon,

    // Literals
    Iri(String),
    PrefixedName(String, String),
    Variable(String),
    StringLiteral(String),
    NumericLiteral(String),
    BooleanLiteral(bool),
    BlankNode(String),

    // Special
    Eof,
    Newline,
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

    /// Parse a SPARQL query string into a Query AST
    pub fn parse(&mut self, query_str: &str) -> Result<Query> {
        self.tokenize(query_str)?;
        self.parse_query()
    }

    /// Tokenize SPARQL query string
    fn tokenize(&mut self, input: &str) -> Result<()> {
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
                    // Skip comments
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
                    // Check if this is a standalone colon or part of a prefixed name
                    if chars
                        .peek()
                        .map_or(true, |c| !c.is_ascii_alphanumeric() && *c != '_')
                    {
                        // Standalone colon (like in default namespace declarations)
                        tokens.push(Token::Colon);
                    } else {
                        // This colon is part of a prefixed name, handle it differently
                        // Put back the colon and parse as identifier
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
                        // Parse IRI
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
                    // Check if this is start of property path syntax or division
                    // For now, treat as divide - property path parsing will be context-sensitive
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
                    // Check if this is a variable or property path operator
                    if chars.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
                        let var = self.parse_identifier(&mut chars);
                        tokens.push(Token::Variable(var));
                    } else {
                        tokens.push(Token::Question);
                    }
                    continue; // Skip the normal variable parsing
                }
                '*' => {
                    chars.next();
                    // Check context - could be multiplication or Kleene star
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
                    continue; // Skip the normal ! parsing
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
                            // Handle escape sequences
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
                        // Put back the underscore and parse as identifier
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
                    // Skip unknown characters for now
                }
            }
        }

        tokens.push(Token::Eof);
        self.tokens = tokens;
        self.position = 0;
        Ok(())
    }

    fn parse_identifier(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
        let mut identifier = String::new();
        let mut found_colon = false;

        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                identifier.push(ch);
                chars.next();
            } else if ch == ':' && !found_colon {
                // Include one colon for prefixed names
                identifier.push(ch);
                chars.next();
                found_colon = true;
            } else {
                break;
            }
        }
        identifier
    }

    fn parse_number(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
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

    fn classify_identifier(&self, identifier: &str) -> Token {
        match identifier.to_uppercase().as_str() {
            "SELECT" => Token::Select,
            "CONSTRUCT" => Token::Construct,
            "ASK" => Token::Ask,
            "DESCRIBE" => Token::Describe,
            "WHERE" => Token::Where,
            "OPTIONAL" => Token::Optional,
            "UNION" => Token::Union,
            "MINUS" => Token::Minus,
            "FILTER" => Token::Filter,
            "BIND" => Token::Bind,
            "SERVICE" => Token::Service,
            "GRAPH" => Token::Graph,
            "FROM" => Token::From,
            "NAMED" => Token::Named,
            "PREFIX" => Token::Prefix,
            "BASE" => Token::Base,
            "DISTINCT" => Token::Distinct,
            "REDUCED" => Token::Reduced,
            "ORDER" => Token::OrderBy,
            "BY" => Token::OrderBy, // Will be handled in context
            "GROUP" => Token::GroupBy,
            "HAVING" => Token::Having,
            "LIMIT" => Token::Limit,
            "OFFSET" => Token::Offset,
            "ASC" => Token::Asc,
            "DESC" => Token::Desc,
            "AS" => Token::As,
            "VALUES" => Token::Values,
            "EXISTS" => Token::Exists,
            "NOT" => Token::Not,
            "AND" => Token::And,
            "OR" => Token::Or,
            "TRUE" => Token::BooleanLiteral(true),
            "FALSE" => Token::BooleanLiteral(false),
            // UPDATE Keywords
            "INSERT" => Token::Insert,
            "DELETE" => Token::Delete,
            "UPDATE" => Token::Update,
            "CREATE" => Token::Create,
            "DROP" => Token::Drop,
            "CLEAR" => Token::Clear,
            "LOAD" => Token::Load,
            "COPY" => Token::Copy,
            "MOVE" => Token::Move,
            "ADD" => Token::Add,
            "DATA" => Token::Data,
            "WITH" => Token::With,
            "USING" => Token::Using,
            "SILENT" => Token::Silent,
            "ALL" => Token::All,
            "DEFAULT" => Token::Default,
            "TO" => Token::To,
            _ => {
                // Check for prefixed name
                if let Some(colon_pos) = identifier.find(':') {
                    let prefix = identifier[..colon_pos].to_string();
                    let local = identifier[colon_pos + 1..].to_string();
                    Token::PrefixedName(prefix, local)
                } else if let Some(stripped) = identifier.strip_prefix(':') {
                    // Default namespace (starts with colon)
                    let local = stripped.to_string();
                    Token::PrefixedName("".to_string(), local)
                } else {
                    // Assume it's an identifier that could be a function name
                    Token::PrefixedName("".to_string(), identifier.to_string())
                }
            }
        }
    }

    fn parse_query(&mut self) -> Result<Query> {
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

        // Skip initial whitespace/newlines
        self.skip_whitespace();

        // Parse prologue (PREFIX and BASE declarations)
        self.parse_prologue(&mut query)?;

        // Skip whitespace after prologue
        self.skip_whitespace();

        // Parse main query
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

    fn parse_prologue(&mut self, query: &mut Query) -> Result<()> {
        while let Some(token) = self.peek() {
            match token {
                Token::Prefix => {
                    self.advance(); // consume PREFIX

                    // Handle both default namespace (:) and named prefixes (prefix:)
                    let prefix = match self.peek() {
                        Some(Token::PrefixedName(prefix, local)) => {
                            if prefix.is_empty() && local.is_empty() {
                                // This is just ":" which represents the default namespace
                                self.advance();
                                String::new()
                            } else if local.is_empty() {
                                // This is a named prefix like "foaf:" where local part is empty
                                let p = prefix.clone();
                                self.advance();
                                p
                            } else {
                                // This is a named prefix like "ex:"
                                let p = prefix.clone();
                                self.advance();
                                p
                            }
                        }
                        Some(Token::Colon) => {
                            // Handle standalone colon for default namespace
                            self.advance();
                            String::new()
                        }
                        _ => {
                            // Debug: print what token we actually got
                            eprintln!("Debug: Got token: {:?}", self.peek());
                            bail!("Expected prefix name or colon after PREFIX")
                        }
                    };

                    let iri = self.expect_iri()?;
                    query.prefixes.insert(prefix.clone(), iri.clone());
                    self.prefixes.insert(prefix, iri);
                }
                Token::Base => {
                    self.advance(); // consume BASE
                    let iri = self.expect_iri()?;
                    query.base_iri = Some(iri.clone());
                    self.base_iri = Some(iri);
                }
                Token::Newline => {
                    self.advance(); // skip newlines
                }
                _ => break,
            }
        }
        Ok(())
    }

    fn parse_select_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Select)?;

        // Parse DISTINCT/REDUCED
        if self.match_token(&Token::Distinct) {
            query.distinct = true;
        } else if self.match_token(&Token::Reduced) {
            query.reduced = true;
        }

        // Parse selection variables or *
        if self.match_token(&Token::Multiply) {
            // SELECT * - will be resolved later to all variables in WHERE clause
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

        // Parse dataset clause
        self.parse_dataset_clause(&mut query.dataset)?;

        // Parse WHERE clause
        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        query.where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;

        // Parse solution modifiers
        self.parse_solution_modifiers(query)?;

        Ok(())
    }

    fn parse_construct_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Construct)?;

        // Parse construct template
        if self.match_token(&Token::LeftBrace) {
            query.construct_template = self.parse_construct_template()?;
            self.expect_token(Token::RightBrace)?;
        }

        // Parse dataset clause
        self.parse_dataset_clause(&mut query.dataset)?;

        // Parse WHERE clause
        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        query.where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;

        // Parse solution modifiers
        self.parse_solution_modifiers(query)?;

        Ok(())
    }

    fn parse_ask_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Ask)?;

        // Parse dataset clause
        self.parse_dataset_clause(&mut query.dataset)?;

        // Parse WHERE clause
        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        query.where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;

        Ok(())
    }

    fn parse_describe_query(&mut self, query: &mut Query) -> Result<()> {
        self.expect_token(Token::Describe)?;

        // Parse variables or IRIs to describe
        while !self.is_at_end() && !matches!(self.peek(), Some(Token::Where) | Some(Token::From)) {
            if let Some(Token::Variable(var)) = self.peek() {
                query.select_variables.push(Variable::new(var.clone())?);
                self.advance();
            } else if matches!(
                self.peek(),
                Some(Token::Iri(_)) | Some(Token::PrefixedName(_, _))
            ) {
                // Skip IRIs for now
                self.advance();
            } else {
                break;
            }
        }

        // Parse dataset clause
        self.parse_dataset_clause(&mut query.dataset)?;

        // Parse optional WHERE clause
        if self.match_token(&Token::Where) {
            self.expect_token(Token::LeftBrace)?;
            query.where_clause = self.parse_group_graph_pattern()?;
            self.expect_token(Token::RightBrace)?;
        }

        Ok(())
    }

    fn parse_dataset_clause(&mut self, dataset: &mut DatasetClause) -> Result<()> {
        while self.match_token(&Token::From) {
            if self.match_token(&Token::Named) {
                let iri = self.expect_iri()?;
                dataset.named_graphs.push(NamedNode::new_unchecked(iri));
            } else {
                let iri = self.expect_iri()?;
                dataset.default_graphs.push(NamedNode::new_unchecked(iri));
            }
        }
        Ok(())
    }

    /// Check if the current pattern contains UNION by looking ahead
    fn has_union_pattern(&self) -> bool {
        let mut pos = self.position;
        let mut brace_depth = 0;

        while pos < self.tokens.len() {
            match &self.tokens[pos] {
                Token::LeftBrace => brace_depth += 1,
                Token::RightBrace => {
                    if brace_depth == 0 {
                        break;
                    }
                    brace_depth -= 1;
                }
                Token::Union if brace_depth == 0 => return true,
                _ => {}
            }
            pos += 1;
        }
        false
    }

    fn parse_group_graph_pattern(&mut self) -> Result<Algebra> {
        // For union patterns, we want to parse the entire union sequence as one unit
        // Check if we have a simple union pattern by looking ahead
        if self.has_union_pattern() {
            return self.parse_graph_pattern_or_union();
        }

        let mut patterns = Vec::new();

        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RightBrace)) {
            // Skip whitespace and newlines before parsing each pattern
            self.skip_whitespace_and_newlines();

            // Check again if we've reached the end after skipping whitespace
            if self.is_at_end() || matches!(self.peek(), Some(Token::RightBrace)) {
                break;
            }

            let pattern = self.parse_graph_pattern_or_union()?;
            patterns.push(pattern);

            // Skip optional dots and whitespace
            self.match_token(&Token::Dot);
            self.skip_whitespace_and_newlines();
        }

        if patterns.is_empty() {
            Ok(Algebra::Table)
        } else if patterns.len() == 1 {
            Ok(patterns.into_iter().next().unwrap())
        } else {
            // Join all patterns
            let mut patterns_iter = patterns.into_iter();
            let mut result = patterns_iter.next().unwrap();
            for pattern in patterns_iter {
                result = Algebra::join(result, pattern);
            }
            Ok(result)
        }
    }

    fn parse_graph_pattern_or_union(&mut self) -> Result<Algebra> {
        let left = self.parse_graph_pattern()?;

        // Check for UNION after the first pattern
        self.skip_whitespace_and_newlines();
        if self.match_token(&Token::Union) {
            // Skip whitespace and newlines after UNION token
            self.skip_whitespace_and_newlines();
            // Parse the rest recursively to get right-associativity
            let right = self.parse_graph_pattern_or_union()?;
            return Ok(Algebra::Union {
                left: Box::new(left),
                right: Box::new(right),
            });
        }

        Ok(left)
    }

    fn parse_graph_pattern(&mut self) -> Result<Algebra> {
        // Skip whitespace and newlines before determining pattern type
        self.skip_whitespace_and_newlines();

        match self.peek() {
            Some(Token::Optional) => self.parse_optional_pattern(),
            Some(Token::Union) => self.parse_union_pattern(),
            Some(Token::Minus) => self.parse_minus_pattern(),
            Some(Token::Filter) => self.parse_filter_pattern(),
            Some(Token::Bind) => self.parse_bind_pattern(),
            Some(Token::Service) => self.parse_service_pattern(),
            Some(Token::Graph) => self.parse_graph_pattern_named(),
            Some(Token::Values) => self.parse_values_pattern(),
            Some(Token::LeftBrace) => {
                self.advance(); // consume {
                let pattern = self.parse_group_graph_pattern()?;
                // Skip whitespace and newlines before expecting the closing brace
                self.skip_whitespace_and_newlines();
                self.expect_token(Token::RightBrace)?;
                Ok(pattern)
            }
            _ => self.parse_basic_graph_pattern(),
        }
    }

    fn parse_basic_graph_pattern(&mut self) -> Result<Algebra> {
        let mut triples = Vec::new();

        while !self.is_at_end() {
            // Skip whitespace/newlines before checking for pattern end
            self.skip_whitespace_and_newlines();

            // Check if we've reached the end of the pattern
            if self.is_pattern_end() {
                break;
            }

            // Skip any additional newlines that might appear
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

    fn parse_triple_pattern(&mut self) -> Result<TriplePattern> {
        // Skip whitespace and newlines before parsing
        self.skip_whitespace_and_newlines();

        let subject = self.parse_term()?;

        // Skip whitespace between subject and predicate
        self.skip_whitespace_and_newlines();

        // Check if we have a property path instead of a simple predicate
        if self.is_property_path_start() {
            let path = self.parse_property_path()?;

            // Skip whitespace between path and object
            self.skip_whitespace_and_newlines();
            let object = self.parse_term()?;

            // Convert property path pattern to proper property path algebra
            // Create a PropertyPathPattern which will be handled by the algebra
            let path_pattern = PropertyPathPattern::new(subject, path, object);

            // For now, return a special triple pattern that indicates property path processing
            // The actual property path evaluation would be handled in the algebra/executor
            return Ok(TriplePattern::new(
                path_pattern.subject.clone(),
                Term::PropertyPath(path_pattern.path.clone()),
                path_pattern.object.clone(),
            ));
        }

        let predicate = self.parse_term()?;

        // Skip whitespace between predicate and object
        self.skip_whitespace_and_newlines();
        let object = self.parse_term()?;

        Ok(TriplePattern::new(subject, predicate, object))
    }

    /// Check if current position starts a property path
    fn is_property_path_start(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Caret)
                | Some(Token::Iri(_))
                | Some(Token::PrefixedName(_, _))
                | Some(Token::LeftParen)
                | Some(Token::Bang)
        )
    }

    /// Parse property path expression
    fn parse_property_path(&mut self) -> Result<PropertyPath> {
        self.parse_property_path_alternative()
    }

    /// Parse property path alternatives (highest precedence)
    fn parse_property_path_alternative(&mut self) -> Result<PropertyPath> {
        let mut left = self.parse_property_path_sequence()?;

        while self.match_token(&Token::Pipe) {
            let right = self.parse_property_path_sequence()?;
            left = PropertyPath::alternative(left, right);
        }

        Ok(left)
    }

    /// Parse property path sequences
    fn parse_property_path_sequence(&mut self) -> Result<PropertyPath> {
        let mut left = self.parse_property_path_postfix()?;

        while self.match_token(&Token::Slash) {
            let right = self.parse_property_path_postfix()?;
            left = PropertyPath::sequence(left, right);
        }

        Ok(left)
    }

    /// Parse property path with postfix operators (*, +, ?)
    fn parse_property_path_postfix(&mut self) -> Result<PropertyPath> {
        let mut path = self.parse_property_path_primary()?;

        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.advance();
                    path = PropertyPath::zero_or_more(path);
                }
                Some(Token::Plus) => {
                    self.advance();
                    path = PropertyPath::one_or_more(path);
                }
                Some(Token::Question) => {
                    self.advance();
                    path = PropertyPath::zero_or_one(path);
                }
                _ => break,
            }
        }

        Ok(path)
    }

    /// Parse primary property path expressions
    fn parse_property_path_primary(&mut self) -> Result<PropertyPath> {
        match self.peek() {
            Some(Token::Caret) => {
                self.advance(); // consume ^
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
                self.advance(); // consume (
                let path = self.parse_property_path()?;
                self.expect_token(Token::RightParen)?;
                Ok(path)
            }
            Some(Token::Bang) => {
                self.advance(); // consume !
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

    fn parse_term(&mut self) -> Result<Term> {
        // Skip whitespace/newlines
        self.skip_whitespace_and_newlines();

        match self.peek() {
            Some(Token::Variable(var)) => {
                let var = var.clone();
                self.advance();
                let variable = Variable::new(&var)?;
                self.variables.insert(variable.clone());
                Ok(Term::Variable(variable))
            }
            Some(Token::Iri(iri)) => {
                let iri = iri.clone();
                self.advance();
                Ok(Term::Iri(NamedNode::new_unchecked(iri)))
            }
            Some(Token::PrefixedName(prefix, local)) => {
                let prefix = prefix.clone();
                let local = local.clone();
                self.advance();

                let full_iri = self.resolve_prefixed_name(&prefix, &local)?;
                Ok(Term::Iri(NamedNode::new_unchecked(full_iri)))
            }
            Some(Token::StringLiteral(value)) => {
                let value = value.clone();
                self.advance();
                Ok(Term::Literal(Literal {
                    value,
                    language: None,
                    datatype: None,
                }))
            }
            Some(Token::NumericLiteral(value)) => {
                let value = value.clone();
                self.advance();
                Ok(Term::Literal(Literal {
                    value,
                    language: None,
                    datatype: Some(NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#decimal",
                    )),
                }))
            }
            Some(Token::BooleanLiteral(value)) => {
                let value = *value;
                self.advance();
                Ok(Term::Literal(Literal {
                    value: value.to_string(),
                    language: None,
                    datatype: Some(NamedNode::new_unchecked(
                        "http://www.w3.org/2001/XMLSchema#boolean",
                    )),
                }))
            }
            Some(Token::BlankNode(id)) => {
                let id = id.clone();
                self.advance();
                Ok(Term::BlankNode(id))
            }
            _ => bail!("Expected term"),
        }
    }

    /// Parse a variable or term (used for quad parsing)
    fn parse_var_or_term(&mut self) -> Result<Term> {
        // This is the same as parse_term since parse_term already handles variables
        self.parse_term()
    }

    fn parse_optional_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Optional)?;
        self.expect_token(Token::LeftBrace)?;
        let pattern = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;

        // Return as left join with no left pattern (will be filled by join operation)
        Ok(Algebra::LeftJoin {
            left: Box::new(Algebra::Table),
            right: Box::new(pattern),
            filter: None,
        })
    }

    fn parse_union_pattern(&mut self) -> Result<Algebra> {
        // This handles the edge case where UNION appears at the start of a pattern
        // In standard SPARQL, UNION should appear between patterns, not at the start
        // But we'll handle it gracefully by treating it as an empty pattern UNION { pattern }
        self.expect_token(Token::Union)?;

        // Parse the pattern after UNION
        let pattern = if self.match_token(&Token::LeftBrace) {
            let p = self.parse_group_graph_pattern()?;
            self.expect_token(Token::RightBrace)?;
            p
        } else {
            self.parse_graph_pattern()?
        };

        Ok(Algebra::Union {
            left: Box::new(Algebra::Table), // Empty pattern on the left
            right: Box::new(pattern),
        })
    }

    fn parse_minus_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Minus)?;
        self.expect_token(Token::LeftBrace)?;
        let pattern = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;

        Ok(Algebra::Minus {
            left: Box::new(Algebra::Table),
            right: Box::new(pattern),
        })
    }

    fn parse_filter_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Filter)?;
        let condition = self.parse_expression()?;

        Ok(Algebra::Filter {
            pattern: Box::new(Algebra::Table),
            condition,
        })
    }

    fn parse_bind_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Bind)?;
        self.expect_token(Token::LeftParen)?;
        let expr = self.parse_expression()?;
        self.expect_token(Token::As)?;
        let var = self.expect_variable()?;
        self.expect_token(Token::RightParen)?;

        Ok(Algebra::Extend {
            pattern: Box::new(Algebra::Table),
            variable: var,
            expr,
        })
    }

    fn parse_service_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Service)?;
        let silent = self.match_token(&Token::Not); // SILENT
        let endpoint = self.parse_term()?;
        self.expect_token(Token::LeftBrace)?;
        let pattern = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;

        Ok(Algebra::Service {
            endpoint,
            pattern: Box::new(pattern),
            silent,
        })
    }

    fn parse_graph_pattern_named(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Graph)?;
        let graph = self.parse_term()?;
        self.expect_token(Token::LeftBrace)?;
        let pattern = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;

        Ok(Algebra::Graph {
            graph,
            pattern: Box::new(pattern),
        })
    }

    fn parse_values_pattern(&mut self) -> Result<Algebra> {
        self.expect_token(Token::Values)?;
        let mut variables = Vec::new();
        let mut bindings = Vec::new();

        // Parse variables
        if self.match_token(&Token::LeftParen) {
            while !self.match_token(&Token::RightParen) {
                variables.push(self.expect_variable()?);
            }
        } else {
            variables.push(self.expect_variable()?);
        }

        // Parse values
        self.expect_token(Token::LeftBrace)?;
        while !self.match_token(&Token::RightBrace) {
            let mut binding = HashMap::new();

            if self.match_token(&Token::LeftParen) {
                for var in &variables {
                    let term = self.parse_term()?;
                    binding.insert(var.clone(), term);
                }
                self.expect_token(Token::RightParen)?;
            } else if !variables.is_empty() {
                let term = self.parse_term()?;
                binding.insert(variables[0].clone(), term);
            }

            bindings.push(binding);
        }

        Ok(Algebra::Values {
            variables,
            bindings,
        })
    }

    fn parse_expression(&mut self) -> Result<Expression> {
        self.parse_or_expression()
    }

    fn parse_or_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_and_expression()?;

        while self.match_token(&Token::Or) {
            let right = self.parse_and_expression()?;
            expr = Expression::Binary {
                op: BinaryOperator::Or,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_and_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_equality_expression()?;

        while self.match_token(&Token::And) {
            let right = self.parse_equality_expression()?;
            expr = Expression::Binary {
                op: BinaryOperator::And,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_equality_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_relational_expression()?;

        while let Some(op) = self.match_equality_operator() {
            let right = self.parse_relational_expression()?;
            expr = Expression::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_relational_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_additive_expression()?;

        while let Some(op) = self.match_relational_operator() {
            let right = self.parse_additive_expression()?;
            expr = Expression::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_additive_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_multiplicative_expression()?;

        while let Some(op) = self.match_additive_operator() {
            let right = self.parse_multiplicative_expression()?;
            expr = Expression::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_multiplicative_expression(&mut self) -> Result<Expression> {
        let mut expr = self.parse_unary_expression()?;

        while let Some(op) = self.match_multiplicative_operator() {
            let right = self.parse_unary_expression()?;
            expr = Expression::Binary {
                op,
                left: Box::new(expr),
                right: Box::new(right),
            };
        }

        Ok(expr)
    }

    fn parse_unary_expression(&mut self) -> Result<Expression> {
        if let Some(op) = self.match_unary_operator() {
            let expr = self.parse_unary_expression()?;
            Ok(Expression::Unary {
                op,
                operand: Box::new(expr),
            })
        } else {
            self.parse_primary_expression()
        }
    }

    fn parse_primary_expression(&mut self) -> Result<Expression> {
        match self.peek() {
            Some(Token::Variable(var)) => {
                let var = var.clone();
                self.advance();
                Ok(Expression::Variable(Variable::new(var)?))
            }
            Some(Token::Iri(iri)) => {
                let iri = iri.clone();
                self.advance();
                Ok(Expression::Iri(NamedNode::new_unchecked(iri)))
            }
            Some(Token::StringLiteral(value)) | Some(Token::NumericLiteral(value)) => {
                let value = value.clone();
                self.advance();
                Ok(Expression::Literal(Literal {
                    value,
                    language: None,
                    datatype: None,
                }))
            }
            Some(Token::BooleanLiteral(value)) => {
                let value = *value;
                self.advance();
                Ok(Expression::Literal(Literal {
                    value: value.to_string(),
                    language: None,
                    datatype: None,
                }))
            }
            Some(Token::LeftParen) => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect_token(Token::RightParen)?;
                Ok(expr)
            }
            Some(Token::PrefixedName(prefix, local)) => {
                // Function call
                let prefix = prefix.clone();
                let local = local.clone();
                let name = format!("{prefix}:{local}");
                self.advance();

                if self.match_token(&Token::LeftParen) {
                    let mut args = Vec::new();
                    while !self.match_token(&Token::RightParen) {
                        args.push(self.parse_expression()?);
                        if !self.match_token(&Token::Comma) {
                            self.expect_token(Token::RightParen)?;
                            break;
                        }
                    }
                    Ok(Expression::Function { name, args })
                } else {
                    // It's an IRI
                    let full_iri = if let Some(base) = self.prefixes.get(&prefix) {
                        format!("{base}{local}")
                    } else {
                        name
                    };
                    Ok(Expression::Iri(NamedNode::new_unchecked(full_iri)))
                }
            }
            _ => bail!("Expected primary expression"),
        }
    }

    fn parse_solution_modifiers(&mut self, query: &mut Query) -> Result<()> {
        // Parse GROUP BY
        if self.match_token(&Token::GroupBy) {
            while !self.is_at_end() && !self.is_solution_modifier_end() {
                let expr = self.parse_expression()?;
                let alias = if self.match_token(&Token::As) {
                    Some(self.expect_variable()?)
                } else {
                    None
                };
                query.group_by.push(GroupCondition { expr, alias });
            }
        }

        // Parse HAVING
        if self.match_token(&Token::Having) {
            query.having = Some(self.parse_expression()?);
        }

        // Parse ORDER BY
        if self.match_token(&Token::OrderBy) {
            while !self.is_at_end() && !self.is_solution_modifier_end() {
                let ascending = if self.match_token(&Token::Desc) {
                    false
                } else {
                    self.match_token(&Token::Asc);
                    true
                };

                let expr = self.parse_expression()?;
                query.order_by.push(OrderCondition { expr, ascending });
            }
        }

        // Parse LIMIT
        if self.match_token(&Token::Limit) {
            if let Some(Token::NumericLiteral(num)) = self.peek() {
                query.limit = num.parse().ok();
                self.advance();
            }
        }

        // Parse OFFSET
        if self.match_token(&Token::Offset) {
            if let Some(Token::NumericLiteral(num)) = self.peek() {
                query.offset = num.parse().ok();
                self.advance();
            }
        }

        Ok(())
    }

    fn parse_construct_template(&mut self) -> Result<Vec<TriplePattern>> {
        let mut triples = Vec::new();

        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RightBrace)) {
            let triple = self.parse_triple_pattern()?;
            triples.push(triple);

            if !self.match_token(&Token::Dot) {
                break;
            }
        }

        Ok(triples)
    }

    // Helper methods
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    fn skip_whitespace(&mut self) {
        while let Some(token) = self.peek() {
            match token {
                Token::Newline => {
                    self.advance();
                }
                _ => break,
            }
        }
    }

    /// Skip whitespace and newlines more comprehensively
    fn skip_whitespace_and_newlines(&mut self) {
        while let Some(token) = self.peek() {
            match token {
                Token::Newline => {
                    self.advance();
                }
                _ => break,
            }
        }
    }

    /// Resolve a prefixed name to its full IRI
    fn resolve_prefixed_name(&self, prefix: &str, local: &str) -> Result<String> {
        if let Some(base) = self.prefixes.get(prefix) {
            Ok(format!("{base}{local}"))
        } else {
            bail!("Undefined prefix '{prefix}' in prefixed name '{prefix}:{local}'")
        }
    }

    fn advance(&mut self) -> Option<&Token> {
        if !self.is_at_end() {
            self.position += 1;
        }
        self.tokens.get(self.position - 1)
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Some(Token::Eof) | None)
    }

    fn match_token(&mut self, token: &Token) -> bool {
        if self.check(token) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check(&self, token: &Token) -> bool {
        if let Some(current) = self.peek() {
            std::mem::discriminant(current) == std::mem::discriminant(token)
        } else {
            false
        }
    }

    fn expect_token(&mut self, token: Token) -> Result<()> {
        if self.check(&token) {
            self.advance();
            Ok(())
        } else {
            bail!("Expected {token:?}, found {:?}", self.peek())
        }
    }

    fn expect_variable(&mut self) -> Result<Variable> {
        if let Some(Token::Variable(var)) = self.peek() {
            let var = var.clone();
            self.advance();
            Ok(Variable::new(var)?)
        } else {
            bail!("Expected variable")
        }
    }

    fn expect_iri(&mut self) -> Result<String> {
        match self.peek() {
            Some(Token::Iri(iri)) => {
                let iri = iri.clone();
                self.advance();
                Ok(iri)
            }
            Some(Token::PrefixedName(prefix, local)) => {
                let prefix = prefix.clone();
                let local = local.clone();
                self.advance();

                self.resolve_prefixed_name(&prefix, &local)
            }
            _ => bail!("Expected IRI"),
        }
    }

    fn expect_prefixed_name(&mut self) -> Result<(String, String)> {
        if let Some(Token::PrefixedName(prefix, local)) = self.peek() {
            let result = (prefix.clone(), local.clone());
            self.advance();
            Ok(result)
        } else {
            bail!("Expected prefixed name")
        }
    }

    fn is_pattern_end(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::RightBrace)
                | Some(Token::Optional)
                | Some(Token::Union)
                | Some(Token::Minus)
                | Some(Token::Filter)
                | Some(Token::Bind)
                | Some(Token::Service)
                | Some(Token::Graph)
                | Some(Token::Values)
                | Some(Token::Eof)
        )
    }

    fn is_solution_modifier_end(&self) -> bool {
        matches!(
            self.peek(),
            Some(Token::Limit)
                | Some(Token::Offset)
                | Some(Token::OrderBy)
                | Some(Token::GroupBy)
                | Some(Token::Having)
                | Some(Token::Eof)
        )
    }

    fn match_equality_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek() {
            Some(Token::Equal) => {
                self.advance();
                Some(BinaryOperator::Equal)
            }
            Some(Token::NotEqual) => {
                self.advance();
                Some(BinaryOperator::NotEqual)
            }
            _ => None,
        }
    }

    fn match_relational_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek() {
            Some(Token::Less) => {
                self.advance();
                Some(BinaryOperator::Less)
            }
            Some(Token::LessEqual) => {
                self.advance();
                Some(BinaryOperator::LessEqual)
            }
            Some(Token::Greater) => {
                self.advance();
                Some(BinaryOperator::Greater)
            }
            Some(Token::GreaterEqual) => {
                self.advance();
                Some(BinaryOperator::GreaterEqual)
            }
            _ => None,
        }
    }

    fn match_additive_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek() {
            Some(Token::Plus) => {
                self.advance();
                Some(BinaryOperator::Add)
            }
            Some(Token::Minus_) => {
                self.advance();
                Some(BinaryOperator::Subtract)
            }
            _ => None,
        }
    }

    fn match_multiplicative_operator(&mut self) -> Option<BinaryOperator> {
        match self.peek() {
            Some(Token::Star) => {
                self.advance();
                Some(BinaryOperator::Multiply)
            }
            Some(Token::Divide) => {
                self.advance();
                Some(BinaryOperator::Divide)
            }
            _ => None,
        }
    }

    fn match_unary_operator(&mut self) -> Option<UnaryOperator> {
        match self.peek() {
            Some(Token::Not) => {
                self.advance();
                Some(UnaryOperator::Not)
            }
            Some(Token::Plus) => {
                self.advance();
                Some(UnaryOperator::Plus)
            }
            Some(Token::Minus_) => {
                self.advance();
                Some(UnaryOperator::Minus)
            }
            _ => None,
        }
    }

    /// Parse a SPARQL UPDATE request string into an UpdateRequest AST
    pub fn parse_update(&mut self, update_str: &str) -> Result<UpdateRequest> {
        self.tokenize(update_str)?;
        self.parse_update_request()
    }

    /// Parse UPDATE request with multiple operations
    fn parse_update_request(&mut self) -> Result<UpdateRequest> {
        let mut update_request = UpdateRequest {
            operations: Vec::new(),
            prefixes: HashMap::new(),
            base_iri: None,
        };

        // Skip initial whitespace/newlines
        self.skip_whitespace();

        // Parse prologue (PREFIX and BASE declarations)
        while let Some(token) = self.peek() {
            match token {
                Token::Prefix => {
                    self.advance(); // consume PREFIX
                    let prefix = self.expect_prefixed_name()?.0;
                    let iri = self.expect_iri()?;
                    update_request.prefixes.insert(prefix.clone(), iri.clone());
                    self.prefixes.insert(prefix, iri);
                }
                Token::Base => {
                    self.advance(); // consume BASE
                    let iri = self.expect_iri()?;
                    update_request.base_iri = Some(iri.clone());
                    self.base_iri = Some(iri);
                }
                _ => break,
            }
        }

        // Parse UPDATE operations
        while !self.is_at_end() {
            self.skip_whitespace();

            let operation = match self.peek() {
                Some(Token::Insert) => self.parse_insert_operation()?,
                Some(Token::Delete) => self.parse_delete_operation()?,
                Some(Token::Clear) => self.parse_clear_operation()?,
                Some(Token::Drop) => self.parse_drop_operation()?,
                Some(Token::Create) => self.parse_create_operation()?,
                Some(Token::Load) => self.parse_load_operation()?,
                Some(Token::Copy) => self.parse_copy_operation()?,
                Some(Token::Move) => self.parse_move_operation()?,
                Some(Token::Add) => self.parse_add_operation()?,
                Some(Token::With) => {
                    // WITH clause followed by UPDATE operation
                    self.advance(); // consume WITH
                    let graph_iri = self.expect_iri()?;
                    let graph_ref = GraphReference::Iri(graph_iri);

                    // Parse the operation and set the WITH graph as the default
                    let mut operation = match self.peek() {
                        Some(Token::Insert) => self.parse_insert_operation()?,
                        Some(Token::Delete) => self.parse_delete_operation()?,
                        _ => bail!("Expected INSERT or DELETE after WITH clause"),
                    };

                    // Apply the WITH graph to the operation
                    match &mut operation {
                        UpdateOperation::DeleteInsertWhere { using, .. } => {
                            if using.is_none() {
                                *using = Some(vec![graph_ref]);
                            }
                        }
                        UpdateOperation::InsertWhere { template, .. } => {
                            // Set default graph for all quads in template
                            for quad in template {
                                if quad.graph.is_none() {
                                    quad.graph = Some(graph_ref.clone());
                                }
                            }
                        }
                        UpdateOperation::DeleteWhere { .. } => {
                            // WITH clause affects the evaluation context
                            // This would be handled at execution time
                        }
                        _ => {
                            // Other operations don't support WITH clause in the same way
                        }
                    }

                    operation
                }
                Some(Token::Eof) => break,
                _ => bail!("Expected UPDATE operation"),
            };

            update_request.operations.push(operation);

            // Skip semicolons between operations
            self.match_token(&Token::Semicolon);
            self.skip_whitespace();
        }

        Ok(update_request)
    }

    /// Parse INSERT operation (INSERT DATA or INSERT WHERE)
    fn parse_insert_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Insert)?;

        if self.match_token(&Token::Data) {
            // INSERT DATA { ... }
            self.parse_insert_data()
        } else {
            // INSERT { ... } WHERE { ... }
            self.parse_insert_where()
        }
    }

    /// Parse DELETE operation (DELETE DATA or DELETE WHERE)
    fn parse_delete_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Delete)?;

        if self.match_token(&Token::Data) {
            // DELETE DATA { ... }
            self.parse_delete_data()
        } else if self.peek() == Some(&Token::Where) {
            // DELETE WHERE { ... }
            self.parse_delete_where()
        } else {
            // DELETE { ... } WHERE { ... } or DELETE { ... } INSERT { ... } WHERE { ... }
            self.parse_delete_insert_where()
        }
    }

    /// Parse INSERT DATA operation
    fn parse_insert_data(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::LeftBrace)?;
        let quads = self.parse_quad_data()?;
        self.expect_token(Token::RightBrace)?;

        Ok(UpdateOperation::InsertData { data: quads })
    }

    /// Parse DELETE DATA operation
    fn parse_delete_data(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::LeftBrace)?;
        let quads = self.parse_quad_data()?;
        self.expect_token(Token::RightBrace)?;

        Ok(UpdateOperation::DeleteData { data: quads })
    }

    /// Parse INSERT WHERE operation
    fn parse_insert_where(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::LeftBrace)?;
        let template = self.parse_quad_pattern_data()?;
        self.expect_token(Token::RightBrace)?;

        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        let where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;

        Ok(UpdateOperation::InsertWhere {
            pattern: Box::new(where_clause),
            template,
        })
    }

    /// Parse DELETE WHERE operation
    fn parse_delete_where(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        let patterns = self.parse_quad_pattern_data()?;
        self.expect_token(Token::RightBrace)?;

        // Convert QuadPattern to TriplePattern for now - full implementation would handle quads properly
        let triple_patterns: Vec<TriplePattern> = patterns
            .into_iter()
            .map(|qp| TriplePattern::new(qp.subject, qp.predicate, qp.object))
            .collect();

        Ok(UpdateOperation::DeleteWhere {
            pattern: Box::new(Algebra::Bgp(triple_patterns)),
        })
    }

    /// Parse DELETE ... INSERT ... WHERE operation
    fn parse_delete_insert_where(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::LeftBrace)?;
        let delete_patterns = self.parse_quad_pattern_data()?;
        self.expect_token(Token::RightBrace)?;

        let insert_patterns = if self.match_token(&Token::Insert) {
            self.expect_token(Token::LeftBrace)?;
            let patterns = self.parse_quad_pattern_data()?;
            self.expect_token(Token::RightBrace)?;
            Some(patterns)
        } else {
            None
        };

        self.expect_token(Token::Where)?;
        self.expect_token(Token::LeftBrace)?;
        let where_clause = self.parse_group_graph_pattern()?;
        self.expect_token(Token::RightBrace)?;

        if let Some(insert_patterns) = insert_patterns {
            Ok(UpdateOperation::DeleteInsertWhere {
                delete_template: delete_patterns,
                insert_template: insert_patterns,
                pattern: Box::new(where_clause),
                using: None,
            })
        } else {
            // Just DELETE ... WHERE - convert to TriplePattern
            let triple_patterns: Vec<TriplePattern> = delete_patterns
                .into_iter()
                .map(|qp| TriplePattern::new(qp.subject, qp.predicate, qp.object))
                .collect();

            Ok(UpdateOperation::DeleteWhere {
                pattern: Box::new(Algebra::Bgp(triple_patterns)),
            })
        }
    }

    /// Parse CLEAR operation
    fn parse_clear_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Clear)?;
        let silent = self.match_token(&Token::Silent);
        let target = self.parse_graph_ref_all()?;

        Ok(UpdateOperation::Clear { target, silent })
    }

    /// Parse DROP operation
    fn parse_drop_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Drop)?;
        let silent = self.match_token(&Token::Silent);
        let target = self.parse_graph_ref_all()?;

        Ok(UpdateOperation::Drop { target, silent })
    }

    /// Parse CREATE operation
    fn parse_create_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Create)?;
        let silent = self.match_token(&Token::Silent);
        let graph = self.expect_iri()?;

        Ok(UpdateOperation::Create {
            graph: GraphReference::Iri(graph),
            silent,
        })
    }

    /// Parse LOAD operation
    fn parse_load_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Load)?;
        let silent = self.match_token(&Token::Silent);
        let source = self.expect_iri()?;

        let graph = if matches!(self.peek(), Some(Token::Iri(_)))
            || matches!(self.peek(), Some(Token::PrefixedName(_, _)))
        {
            Some(GraphReference::Iri(self.expect_iri()?))
        } else {
            None
        };

        Ok(UpdateOperation::Load {
            source,
            graph,
            silent,
        })
    }

    /// Parse COPY operation
    fn parse_copy_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Copy)?;
        let silent = self.match_token(&Token::Silent);
        let from = self.parse_graph_ref()?;
        self.expect_token(Token::To)?;
        let to = self.parse_graph_ref()?;

        Ok(UpdateOperation::Copy { from, to, silent })
    }

    /// Parse MOVE operation
    fn parse_move_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Move)?;
        let silent = self.match_token(&Token::Silent);
        let from = self.parse_graph_ref()?;
        self.expect_token(Token::To)?;
        let to = self.parse_graph_ref()?;

        Ok(UpdateOperation::Move { from, to, silent })
    }

    /// Parse ADD operation
    fn parse_add_operation(&mut self) -> Result<UpdateOperation> {
        self.expect_token(Token::Add)?;
        let silent = self.match_token(&Token::Silent);
        let from = self.parse_graph_ref()?;
        self.expect_token(Token::To)?;
        let to = self.parse_graph_ref()?;

        Ok(UpdateOperation::Add { from, to, silent })
    }

    /// Parse graph reference (DEFAULT, NAMED, or IRI)
    fn parse_graph_ref(&mut self) -> Result<GraphTarget> {
        match self.peek() {
            Some(Token::Default) => {
                self.advance();
                Ok(GraphTarget::Default)
            }
            Some(Token::Named) => {
                self.advance();
                Ok(GraphTarget::Named)
            }
            Some(Token::Iri(_)) | Some(Token::PrefixedName(_, _)) => {
                let iri = self.expect_iri()?;
                Ok(GraphTarget::Graph(GraphReference::Iri(iri)))
            }
            _ => bail!("Expected graph reference"),
        }
    }

    /// Parse graph reference with ALL option
    fn parse_graph_ref_all(&mut self) -> Result<GraphTarget> {
        match self.peek() {
            Some(Token::All) => {
                self.advance();
                Ok(GraphTarget::All)
            }
            _ => self.parse_graph_ref(),
        }
    }

    /// Parse quad data for INSERT/DELETE DATA
    fn parse_quad_data(&mut self) -> Result<Vec<QuadPattern>> {
        let mut quads = Vec::new();

        while !self.is_at_end() && !matches!(self.peek(), Some(Token::RightBrace)) {
            let quad = self.parse_quad()?;
            quads.push(quad);

            // Skip optional dots
            self.match_token(&Token::Dot);
        }

        Ok(quads)
    }

    /// Parse quad pattern data for INSERT/DELETE templates
    fn parse_quad_pattern_data(&mut self) -> Result<Vec<QuadPattern>> {
        // For now, use the same logic as quad_data
        // In a full implementation, this would handle variables in templates
        self.parse_quad_data()
    }

    /// Parse a single quad
    fn parse_quad(&mut self) -> Result<QuadPattern> {
        let subject = self.parse_var_or_term()?;
        let predicate = self.parse_var_or_term()?;
        let object = self.parse_var_or_term()?;

        // Check for optional graph context
        let graph = if matches!(
            self.peek(),
            Some(Token::Iri(_)) | Some(Token::PrefixedName(_, _))
        ) && !matches!(self.peek(), Some(Token::Dot) | Some(Token::RightBrace))
        {
            let iri = self.expect_iri()?;
            Some(GraphReference::Iri(iri))
        } else {
            None
        };

        Ok(QuadPattern {
            subject,
            predicate,
            object,
            graph,
        })
    }
}

impl Default for QueryParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to parse a SPARQL query
pub fn parse_query(query_str: &str) -> Result<Query> {
    let mut parser = QueryParser::new();
    parser.parse(query_str)
}

/// Convenience function to parse a SPARQL UPDATE request
pub fn parse_update(update_str: &str) -> Result<UpdateRequest> {
    let mut parser = QueryParser::new();
    parser.parse_update(update_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Variable;

    #[test]
    fn test_simple_select_query() {
        let query_str = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            SELECT ?person ?name WHERE {
                ?person foaf:name ?name .
            }
        "#;

        let query = parse_query(query_str).unwrap();
        assert_eq!(query.query_type, QueryType::Select);
        assert_eq!(
            query.select_variables,
            vec![
                Variable::new("person").unwrap(),
                Variable::new("name").unwrap()
            ]
        );
        assert!(!query.prefixes.is_empty());
    }

    #[test]
    fn test_construct_query() {
        let query_str = r#"
            CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }
        "#;

        let query = parse_query(query_str).unwrap();
        assert_eq!(query.query_type, QueryType::Construct);
        assert_eq!(query.construct_template.len(), 1);
    }

    #[test]
    fn test_ask_query() {
        let query_str = r#"
            ASK WHERE { ?s ?p ?o }
        "#;

        let query = parse_query(query_str).unwrap();
        assert_eq!(query.query_type, QueryType::Ask);
    }

    #[test]
    fn test_tokenization() {
        let mut parser = QueryParser::new();
        parser.tokenize("SELECT ?x WHERE { ?x ?y ?z }").unwrap();

        // Debug print tokens
        println!("Tokens: {:?}", parser.tokens);

        // Also test prefixed name tokenization specifically
        let mut parser2 = QueryParser::new();
        parser2.tokenize("foaf:name").unwrap();
        println!("Prefixed name tokens: {:?}", parser2.tokens);

        assert!(matches!(parser.tokens[0], Token::Select));
        assert!(matches!(parser.tokens[1], Token::Variable(_)));
        assert!(matches!(parser.tokens[2], Token::Where));
    }

    #[test]
    fn test_union_query() {
        let query_str = r#"
            PREFIX foaf: <http://xmlns.com/foaf/0.1/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?name WHERE {
                { ?person foaf:name ?name }
                UNION
                { ?person rdfs:label ?name }
            }
        "#;

        // Debug tokenization first
        let mut parser = QueryParser::new();
        parser.tokenize(query_str).unwrap();
        println!("Union query tokens: {:?}", parser.tokens);

        let query = parse_query(query_str)
            .map_err(|e| {
                eprintln!("Parse error: {e}");
                e
            })
            .unwrap();
        assert_eq!(query.query_type, QueryType::Select);
        assert_eq!(query.select_variables, vec![Variable::new("name").unwrap()]);

        // Check that the where clause is a Union
        match &query.where_clause {
            Algebra::Union { left, right } => {
                // Check left side is a BGP with one pattern
                if let Algebra::Bgp(patterns) = left.as_ref() {
                    assert_eq!(patterns.len(), 1);
                } else {
                    panic!("Expected BGP on left side of union");
                }

                // Check right side is a BGP with one pattern
                if let Algebra::Bgp(patterns) = right.as_ref() {
                    assert_eq!(patterns.len(), 1);
                } else {
                    panic!("Expected BGP on right side of union");
                }
            }
            _ => panic!("Expected Union algebra"),
        }
    }

    #[test]
    fn test_update_parsing() {
        // Test simple INSERT DATA first
        let update_str = r#"INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }"#;

        let update_request = parse_update(update_str).unwrap();
        assert_eq!(update_request.operations.len(), 1);
        match &update_request.operations[0] {
            UpdateOperation::InsertData { data } => {
                assert_eq!(data.len(), 1);
            }
            _ => panic!("Expected InsertData operation"),
        }
    }

    #[test]
    fn test_update_tokenization() {
        let mut parser = QueryParser::new();
        parser.tokenize("INSERT DATA").unwrap();

        println!("UPDATE tokens: {:?}", parser.tokens);
        assert!(matches!(parser.tokens[0], Token::Insert));
        assert!(matches!(parser.tokens[1], Token::Data));
    }

    #[test]
    fn test_multiple_union_query() {
        let query_str = r#"
            PREFIX : <http://example.org/>
            SELECT ?x WHERE {
                { ?x a :ClassA }
                UNION
                { ?x a :ClassB }
                UNION
                { ?x a :ClassC }
            }
        "#;

        let query = parse_query(query_str).unwrap();

        // Check that we have nested unions (right-associative)
        match &query.where_clause {
            Algebra::Union { left: _, right } => {
                // The right side should also be a Union
                match right.as_ref() {
                    Algebra::Union { .. } => {
                        // Success - we have nested unions
                    }
                    _ => panic!("Expected nested Union on right side"),
                }
            }
            _ => panic!("Expected Union algebra"),
        }
    }
}
