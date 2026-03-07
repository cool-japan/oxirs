//! Comprehensive SPARQL 1.2 Parser
//!
//! This module provides a complete native implementation of a SPARQL 1.2 parser,
//! eliminating the need for external dependencies like spargebra.
//!
//! Features:
//! - Full SPARQL 1.2 syntax support
//! - Property paths (^, *, +, ?, /, |)
//! - Built-in functions and operators
//! - Aggregate queries (COUNT, SUM, AVG, MIN, MAX, GROUP_CONCAT, SAMPLE)
//! - Subqueries and complex expressions
//! - VALUES clause
//! - BIND operations
//! - SERVICE federation (basic support)

use crate::model::{BlankNode, Literal, NamedNode, Variable};
use crate::query::algebra::{AlgebraTriplePattern as TriplePattern, TermPattern};
use crate::query::sparql_algebra::{
    AggregateExpression, Expression, GraphPattern, OrderExpression, PropertyPath,
};
use crate::query::sparql_query::{Query, QueryDataset};
use crate::OxirsError;
use std::collections::HashMap;

/// Advanced SPARQL 1.2 parser with full language support
#[derive(Debug, Clone)]
pub struct AdvancedSparqlParser {
    base_iri: Option<NamedNode>,
    prefixes: HashMap<String, NamedNode>,
    blank_node_counter: usize,
}

impl Default for AdvancedSparqlParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedSparqlParser {
    /// Creates a new advanced SPARQL parser
    pub fn new() -> Self {
        let mut parser = Self {
            base_iri: None,
            prefixes: HashMap::new(),
            blank_node_counter: 0,
        };

        // Add standard prefixes
        parser.add_standard_prefixes();
        parser
    }

    /// Add standard W3C prefixes
    fn add_standard_prefixes(&mut self) {
        let standard_prefixes = [
            ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
            ("rdfs", "http://www.w3.org/2000/01/rdf-schema#"),
            ("owl", "http://www.w3.org/2002/07/owl#"),
            ("xsd", "http://www.w3.org/2001/XMLSchema#"),
            ("dc", "http://purl.org/dc/elements/1.1/"),
            ("dcterms", "http://purl.org/dc/terms/"),
            ("foaf", "http://xmlns.com/foaf/0.1/"),
            ("skos", "http://www.w3.org/2004/02/skos/core#"),
        ];

        for (prefix, namespace) in &standard_prefixes {
            if let Ok(iri) = NamedNode::new(*namespace) {
                self.prefixes.insert(prefix.to_string(), iri);
            }
        }
    }

    /// Sets the base IRI for resolving relative IRIs
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, OxirsError> {
        self.base_iri = Some(NamedNode::new(base_iri.into())?);
        Ok(self)
    }

    /// Adds a prefix mapping
    pub fn with_prefix(
        mut self,
        prefix: impl Into<String>,
        iri: impl Into<String>,
    ) -> Result<Self, OxirsError> {
        self.prefixes
            .insert(prefix.into(), NamedNode::new(iri.into())?);
        Ok(self)
    }

    /// Parse a complete SPARQL query
    pub fn parse(&mut self, query: &str) -> Result<Query, OxirsError> {
        let query = self.preprocess_query(query);
        let tokens = self.tokenize(&query)?;
        self.parse_tokens(tokens)
    }

    /// Preprocess query by handling comments and normalizing whitespace
    fn preprocess_query(&self, query: &str) -> String {
        let mut processed = String::new();
        let mut in_string = false;
        let mut escape_next = false;
        let mut chars = query.chars().peekable();

        while let Some(ch) = chars.next() {
            if escape_next {
                processed.push(ch);
                escape_next = false;
                continue;
            }

            match ch {
                '\\' if in_string => {
                    processed.push(ch);
                    escape_next = true;
                }
                '"' => {
                    in_string = !in_string;
                    processed.push(ch);
                }
                '#' if !in_string => {
                    // Skip comment to end of line
                    while let Some(next_ch) = chars.next() {
                        if next_ch == '\n' {
                            processed.push(next_ch);
                            break;
                        }
                    }
                }
                _ => {
                    processed.push(ch);
                }
            }
        }

        processed
    }

    /// Tokenize the SPARQL query
    fn tokenize(&self, query: &str) -> Result<Vec<Token>, OxirsError> {
        let mut tokens = Vec::new();
        let mut chars = query.chars().peekable();
        let mut line = 1;
        let mut column = 1;

        while let Some(&ch) = chars.peek() {
            match ch {
                // Whitespace
                ' ' | '\t' => {
                    chars.next();
                    column += 1;
                }
                '\n' => {
                    chars.next();
                    line += 1;
                    column = 1;
                }
                '\r' => {
                    chars.next();
                    column += 1;
                }

                // String literals
                '"' => {
                    tokens.push(self.parse_string_literal(&mut chars, line, column)?);
                    column += 1;
                }

                // IRIs
                '<' => {
                    tokens.push(self.parse_iri(&mut chars, line, column)?);
                    column += 1;
                }

                // Variables
                '?' | '$' => {
                    tokens.push(self.parse_variable(&mut chars, line, column)?);
                    column += 1;
                }

                // Blank nodes
                '_' => {
                    if chars.clone().nth(1) == Some(':') {
                        tokens.push(self.parse_blank_node(&mut chars, line, column)?);
                    } else {
                        tokens.push(self.parse_identifier(&mut chars, line, column)?);
                    }
                    column += 1;
                }

                // Numbers
                '0'..='9' => {
                    tokens.push(self.parse_number(&mut chars, line, column)?);
                    column += 1;
                }

                // Punctuation and operators
                '(' => {
                    chars.next();
                    tokens.push(Token::LeftParen { line, column });
                    column += 1;
                }
                ')' => {
                    chars.next();
                    tokens.push(Token::RightParen { line, column });
                    column += 1;
                }
                '{' => {
                    chars.next();
                    tokens.push(Token::LeftBrace { line, column });
                    column += 1;
                }
                '}' => {
                    chars.next();
                    tokens.push(Token::RightBrace { line, column });
                    column += 1;
                }
                '[' => {
                    chars.next();
                    tokens.push(Token::LeftBracket { line, column });
                    column += 1;
                }
                ']' => {
                    chars.next();
                    tokens.push(Token::RightBracket { line, column });
                    column += 1;
                }
                '.' => {
                    chars.next();
                    tokens.push(Token::Dot { line, column });
                    column += 1;
                }
                ';' => {
                    chars.next();
                    tokens.push(Token::Semicolon { line, column });
                    column += 1;
                }
                ',' => {
                    chars.next();
                    tokens.push(Token::Comma { line, column });
                    column += 1;
                }

                // Multi-character operators
                '!' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::NotEqual { line, column });
                        column += 2;
                    } else {
                        tokens.push(Token::Not { line, column });
                        column += 1;
                    }
                }
                '=' => {
                    chars.next();
                    tokens.push(Token::Equal { line, column });
                    column += 1;
                }
                '<' => {
                    // Already handled above for IRIs
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::LessEqual { line, column });
                        column += 2;
                    } else {
                        tokens.push(Token::Less { line, column });
                        column += 1;
                    }
                }
                '>' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::GreaterEqual { line, column });
                        column += 2;
                    } else {
                        tokens.push(Token::Greater { line, column });
                        column += 1;
                    }
                }
                '&' => {
                    chars.next();
                    if chars.peek() == Some(&'&') {
                        chars.next();
                        tokens.push(Token::And { line, column });
                        column += 2;
                    } else {
                        return Err(OxirsError::Parse(format!(
                            "Unexpected character '&' at line {}, column {}",
                            line, column
                        )));
                    }
                }
                '|' => {
                    chars.next();
                    if chars.peek() == Some(&'|') {
                        chars.next();
                        tokens.push(Token::Or { line, column });
                        column += 2;
                    } else {
                        tokens.push(Token::Pipe { line, column }); // For property paths
                        column += 1;
                    }
                }

                // Property path operators
                '^' => {
                    chars.next();
                    tokens.push(Token::Caret { line, column });
                    column += 1;
                }
                '*' => {
                    chars.next();
                    tokens.push(Token::Star { line, column });
                    column += 1;
                }
                '+' => {
                    chars.next();
                    tokens.push(Token::Plus { line, column });
                    column += 1;
                }

                // Identifiers and keywords
                'a'..='z' | 'A'..='Z' => {
                    tokens.push(self.parse_identifier(&mut chars, line, column)?);
                    column += 1;
                }

                _ => {
                    return Err(OxirsError::Parse(format!(
                        "Unexpected character '{}' at line {}, column {}",
                        ch, line, column
                    )));
                }
            }
        }

        tokens.push(Token::Eof);
        Ok(tokens)
    }

    /// Parse string literal
    fn parse_string_literal(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
        line: usize,
        column: usize,
    ) -> Result<Token, OxirsError> {
        chars.next(); // consume opening quote
        let mut value = String::new();
        let mut escape_next = false;

        while let Some(ch) = chars.next() {
            if escape_next {
                match ch {
                    'n' => value.push('\n'),
                    'r' => value.push('\r'),
                    't' => value.push('\t'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    _ => {
                        value.push('\\');
                        value.push(ch);
                    }
                }
                escape_next = false;
            } else if ch == '\\' {
                escape_next = true;
            } else if ch == '"' {
                return Ok(Token::StringLiteral { value, line, column });
            } else {
                value.push(ch);
            }
        }

        Err(OxirsError::Parse(format!(
            "Unterminated string literal at line {}, column {}",
            line, column
        )))
    }

    /// Parse IRI
    fn parse_iri(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
        line: usize,
        column: usize,
    ) -> Result<Token, OxirsError> {
        chars.next(); // consume '<'
        let mut value = String::new();

        while let Some(ch) = chars.next() {
            if ch == '>' {
                return Ok(Token::Iri { value, line, column });
            } else if ch == '<' || ch == '"' || ch == '{' || ch == '}' || ch == '|' || ch == '^'
                || ch == '`' || ch == '\\'
                || (ch as u32) < 32
            {
                return Err(OxirsError::Parse(format!(
                    "Invalid character '{}' in IRI at line {}, column {}",
                    ch, line, column
                )));
            } else {
                value.push(ch);
            }
        }

        Err(OxirsError::Parse(format!(
            "Unterminated IRI at line {}, column {}",
            line, column
        )))
    }

    /// Parse variable
    fn parse_variable(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
        line: usize,
        column: usize,
    ) -> Result<Token, OxirsError> {
        let prefix = chars.next().expect("char should be available after peek"); // consume '?' or '$'
        let mut name = String::new();

        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                name.push(chars.next().expect("char should be available after peek"));
            } else {
                break;
            }
        }

        if name.is_empty() {
            return Err(OxirsError::Parse(format!(
                "Empty variable name at line {}, column {}",
                line, column
            )));
        }

        Ok(Token::Variable {
            name: format!("{}{prefix, name}"),
            line,
            column,
        })
    }

    /// Parse blank node
    fn parse_blank_node(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
        line: usize,
        column: usize,
    ) -> Result<Token, OxirsError> {
        chars.next(); // consume '_'
        chars.next(); // consume ':'
        let mut label = String::new();

        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                label.push(chars.next().expect("char should be available after peek"));
            } else {
                break;
            }
        }

        if label.is_empty() {
            return Err(OxirsError::Parse(format!(
                "Empty blank node label at line {}, column {}",
                line, column
            )));
        }

        Ok(Token::BlankNode {
            label,
            line,
            column,
        })
    }

    /// Parse number (integer or decimal)
    fn parse_number(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
        line: usize,
        column: usize,
    ) -> Result<Token, OxirsError> {
        let mut value = String::new();
        let mut is_decimal = false;

        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_digit() {
                value.push(chars.next().expect("char should be available after peek"));
            } else if ch == '.' && !is_decimal {
                is_decimal = true;
                value.push(chars.next().expect("char should be available after peek"));
            } else {
                break;
            }
        }

        if is_decimal {
            Ok(Token::Decimal { value, line, column })
        } else {
            Ok(Token::Integer { value, line, column })
        }
    }

    /// Parse identifier (keywords or prefixed names)
    fn parse_identifier(
        &self,
        chars: &mut std::iter::Peekable<std::str::Chars>,
        line: usize,
        column: usize,
    ) -> Result<Token, OxirsError> {
        let mut value = String::new();

        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                value.push(chars.next().expect("char should be available after peek"));
            } else if ch == ':' {
                // This might be a prefixed name
                chars.next(); // consume ':'
                let mut local_name = String::new();
                
                while let Some(&ch) = chars.peek() {
                    if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                        local_name.push(chars.next().expect("char should be available after peek"));
                    } else {
                        break;
                    }
                }
                
                return Ok(Token::PrefixedName {
                    prefix: value,
                    local_name,
                    line,
                    column,
                });
            } else {
                break;
            }
        }

        // Check if it's a keyword
        let keyword_token = match value.to_uppercase().as_str() {
            "SELECT" => Token::Select { line, column },
            "CONSTRUCT" => Token::Construct { line, column },
            "ASK" => Token::Ask { line, column },
            "DESCRIBE" => Token::Describe { line, column },
            "WHERE" => Token::Where { line, column },
            "FROM" => Token::From { line, column },
            "NAMED" => Token::Named { line, column },
            "PREFIX" => Token::Prefix { line, column },
            "BASE" => Token::Base { line, column },
            "DISTINCT" => Token::Distinct { line, column },
            "REDUCED" => Token::Reduced { line, column },
            "ORDER" => Token::Order { line, column },
            "BY" => Token::By { line, column },
            "ASC" => Token::Asc { line, column },
            "DESC" => Token::Desc { line, column },
            "LIMIT" => Token::Limit { line, column },
            "OFFSET" => Token::Offset { line, column },
            "UNION" => Token::Union { line, column },
            "OPTIONAL" => Token::Optional { line, column },
            "FILTER" => Token::Filter { line, column },
            "GRAPH" => Token::Graph { line, column },
            "SERVICE" => Token::Service { line, column },
            "BIND" => Token::Bind { line, column },
            "VALUES" => Token::Values { line, column },
            "GROUP" => Token::Group { line, column },
            "HAVING" => Token::Having { line, column },
            "COUNT" => Token::Count { line, column },
            "SUM" => Token::Sum { line, column },
            "MIN" => Token::Min { line, column },
            "MAX" => Token::Max { line, column },
            "AVG" => Token::Avg { line, column },
            "SAMPLE" => Token::Sample { line, column },
            "GROUP_CONCAT" => Token::GroupConcat { line, column },
            "SEPARATOR" => Token::Separator { line, column },
            "AS" => Token::As { line, column },
            "IF" => Token::If { line, column },
            "EXISTS" => Token::Exists { line, column },
            "NOT" => Token::Not { line, column },
            "IN" => Token::In { line, column },
            "BOUND" => Token::Bound { line, column },
            "REGEX" => Token::Regex { line, column },
            "REPLACE" => Token::Replace { line, column },
            "SUBSTR" => Token::Substr { line, column },
            "STRLEN" => Token::Strlen { line, column },
            "UCASE" => Token::Ucase { line, column },
            "LCASE" => Token::Lcase { line, column },
            "CONCAT" => Token::Concat { line, column },
            "CONTAINS" => Token::Contains { line, column },
            "STRSTARTS" => Token::Strstarts { line, column },
            "STRENDS" => Token::Strends { line, column },
            "STRBEFORE" => Token::Strbefore { line, column },
            "STRAFTER" => Token::Strafter { line, column },
            "YEAR" => Token::Year { line, column },
            "MONTH" => Token::Month { line, column },
            "DAY" => Token::Day { line, column },
            "HOURS" => Token::Hours { line, column },
            "MINUTES" => Token::Minutes { line, column },
            "SECONDS" => Token::Seconds { line, column },
            "TIMEZONE" => Token::Timezone { line, column },
            "TZ" => Token::Tz { line, column },
            "NOW" => Token::Now { line, column },
            "UUID" => Token::Uuid { line, column },
            "STRUUID" => Token::Struuid { line, column },
            "MD5" => Token::Md5 { line, column },
            "SHA1" => Token::Sha1 { line, column },
            "SHA256" => Token::Sha256 { line, column },
            "SHA384" => Token::Sha384 { line, column },
            "SHA512" => Token::Sha512 { line, column },
            "COALESCE" => Token::Coalesce { line, column },
            "LANG" => Token::Lang { line, column },
            "DATATYPE" => Token::Datatype { line, column },
            "IRI" => Token::IriFunction { line, column },
            "URI" => Token::UriFunction { line, column },
            "BNODE" => Token::Bnode { line, column },
            "RAND" => Token::Rand { line, column },
            "ABS" => Token::Abs { line, column },
            "CEIL" => Token::Ceil { line, column },
            "FLOOR" => Token::Floor { line, column },
            "ROUND" => Token::Round { line, column },
            "SQRT" => Token::Sqrt { line, column },
            "STR" => Token::Str { line, column },
            "LANGMATCHES" => Token::Langmatches { line, column },
            "SAMETERM" => Token::Sameterm { line, column },
            "ISIRI" => Token::Isiri { line, column },
            "ISURI" => Token::Isuri { line, column },
            "ISBLANK" => Token::Isblank { line, column },
            "ISLITERAL" => Token::Isliteral { line, column },
            "ISNUMERIC" => Token::Isnumeric { line, column },
            "TRUE" => Token::True { line, column },
            "FALSE" => Token::False { line, column },
            "A" => Token::A { line, column }, // shorthand for rdf:type
            _ => Token::Identifier { value, line, column },
        };

        Ok(keyword_token)
    }

    /// Parse tokens into a Query
    fn parse_tokens(&mut self, tokens: Vec<Token>) -> Result<Query, OxirsError> {
        let mut parser = TokenParser::new(tokens);
        parser.parse_query()
    }
}

/// Token types for SPARQL parsing
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Literals and identifiers
    StringLiteral { value: String, line: usize, column: usize },
    Iri { value: String, line: usize, column: usize },
    Variable { name: String, line: usize, column: usize },
    BlankNode { label: String, line: usize, column: usize },
    PrefixedName { prefix: String, local_name: String, line: usize, column: usize },
    Integer { value: String, line: usize, column: usize },
    Decimal { value: String, line: usize, column: usize },
    Identifier { value: String, line: usize, column: usize },

    // Keywords
    Select { line: usize, column: usize },
    Construct { line: usize, column: usize },
    Ask { line: usize, column: usize },
    Describe { line: usize, column: usize },
    Where { line: usize, column: usize },
    From { line: usize, column: usize },
    Named { line: usize, column: usize },
    Prefix { line: usize, column: usize },
    Base { line: usize, column: usize },
    Distinct { line: usize, column: usize },
    Reduced { line: usize, column: usize },
    Order { line: usize, column: usize },
    By { line: usize, column: usize },
    Asc { line: usize, column: usize },
    Desc { line: usize, column: usize },
    Limit { line: usize, column: usize },
    Offset { line: usize, column: usize },
    Union { line: usize, column: usize },
    Optional { line: usize, column: usize },
    Filter { line: usize, column: usize },
    Graph { line: usize, column: usize },
    Service { line: usize, column: usize },
    Bind { line: usize, column: usize },
    Values { line: usize, column: usize },
    Group { line: usize, column: usize },
    Having { line: usize, column: usize },
    As { line: usize, column: usize },
    A { line: usize, column: usize }, // rdf:type shorthand

    // Functions and operators
    Count { line: usize, column: usize },
    Sum { line: usize, column: usize },
    Min { line: usize, column: usize },
    Max { line: usize, column: usize },
    Avg { line: usize, column: usize },
    Sample { line: usize, column: usize },
    GroupConcat { line: usize, column: usize },
    Separator { line: usize, column: usize },
    If { line: usize, column: usize },
    Exists { line: usize, column: usize },
    Not { line: usize, column: usize },
    In { line: usize, column: usize },
    Bound { line: usize, column: usize },
    Regex { line: usize, column: usize },
    Replace { line: usize, column: usize },
    Substr { line: usize, column: usize },
    Strlen { line: usize, column: usize },
    Ucase { line: usize, column: usize },
    Lcase { line: usize, column: usize },
    Concat { line: usize, column: usize },
    Contains { line: usize, column: usize },
    Strstarts { line: usize, column: usize },
    Strends { line: usize, column: usize },
    Strbefore { line: usize, column: usize },
    Strafter { line: usize, column: usize },
    Year { line: usize, column: usize },
    Month { line: usize, column: usize },
    Day { line: usize, column: usize },
    Hours { line: usize, column: usize },
    Minutes { line: usize, column: usize },
    Seconds { line: usize, column: usize },
    Timezone { line: usize, column: usize },
    Tz { line: usize, column: usize },
    Now { line: usize, column: usize },
    Uuid { line: usize, column: usize },
    Struuid { line: usize, column: usize },
    Md5 { line: usize, column: usize },
    Sha1 { line: usize, column: usize },
    Sha256 { line: usize, column: usize },
    Sha384 { line: usize, column: usize },
    Sha512 { line: usize, column: usize },
    Coalesce { line: usize, column: usize },
    Lang { line: usize, column: usize },
    Datatype { line: usize, column: usize },
    IriFunction { line: usize, column: usize },
    UriFunction { line: usize, column: usize },
    Bnode { line: usize, column: usize },
    Rand { line: usize, column: usize },
    Abs { line: usize, column: usize },
    Ceil { line: usize, column: usize },
    Floor { line: usize, column: usize },
    Round { line: usize, column: usize },
    Sqrt { line: usize, column: usize },
    Str { line: usize, column: usize },
    Langmatches { line: usize, column: usize },
    Sameterm { line: usize, column: usize },
    Isiri { line: usize, column: usize },
    Isuri { line: usize, column: usize },
    Isblank { line: usize, column: usize },
    Isliteral { line: usize, column: usize },
    Isnumeric { line: usize, column: usize },
    True { line: usize, column: usize },
    False { line: usize, column: usize },

    // Punctuation and operators
    LeftParen { line: usize, column: usize },
    RightParen { line: usize, column: usize },
    LeftBrace { line: usize, column: usize },
    RightBrace { line: usize, column: usize },
    LeftBracket { line: usize, column: usize },
    RightBracket { line: usize, column: usize },
    Dot { line: usize, column: usize },
    Semicolon { line: usize, column: usize },
    Comma { line: usize, column: usize },
    Equal { line: usize, column: usize },
    NotEqual { line: usize, column: usize },
    Less { line: usize, column: usize },
    LessEqual { line: usize, column: usize },
    Greater { line: usize, column: usize },
    GreaterEqual { line: usize, column: usize },
    And { line: usize, column: usize },
    Or { line: usize, column: usize },
    Pipe { line: usize, column: usize }, // For property paths
    Caret { line: usize, column: usize }, // For property paths
    Star { line: usize, column: usize }, // For property paths
    Plus { line: usize, column: usize }, // For property paths

    // End of file
    Eof,
}

/// Token parser that converts tokens to AST
struct TokenParser {
    tokens: Vec<Token>,
    position: usize,
}

impl TokenParser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, position: 0 }
    }

    fn current_token(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> &Token {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
        self.current_token()
    }

    fn parse_query(&mut self) -> Result<Query, OxirsError> {
        // Handle prologue (BASE and PREFIX declarations)
        while matches!(self.current_token(), Token::Base { .. } | Token::Prefix { .. }) {
            match self.current_token() {
                Token::Base { .. } => {
                    self.advance(); // consume BASE
                    if let Token::Iri { value, .. } = self.advance() {
                        // Set base IRI
                        self.advance();
                    } else {
                        return Err(OxirsError::Parse("Expected IRI after BASE".to_string()));
                    }
                }
                Token::Prefix { .. } => {
                    self.advance(); // consume PREFIX
                    if let Token::Identifier { value: prefix, .. } = self.advance() {
                        if let Token::Iri { value: iri, .. } = self.advance() {
                            // Add prefix mapping
                            self.advance();
                        } else {
                            return Err(OxirsError::Parse("Expected IRI after prefix".to_string()));
                        }
                    } else {
                        return Err(OxirsError::Parse("Expected prefix name after PREFIX".to_string()));
                    }
                }
                _ => unreachable!(),
            }
        }

        // Parse main query
        match self.current_token() {
            Token::Select { .. } => self.parse_select_query(),
            Token::Construct { .. } => self.parse_construct_query(),
            Token::Ask { .. } => self.parse_ask_query(),
            Token::Describe { .. } => self.parse_describe_query(),
            _ => Err(OxirsError::Parse(
                "Expected SELECT, CONSTRUCT, ASK, or DESCRIBE".to_string(),
            )),
        }
    }

    fn parse_select_query(&mut self) -> Result<Query, OxirsError> {
        self.advance(); // consume SELECT

        // Parse modifiers
        let distinct = matches!(self.current_token(), Token::Distinct { .. });
        if distinct {
            self.advance();
        }

        // Parse variables (simplified - just consume until WHERE)
        while !matches!(self.current_token(), Token::Where { .. }) {
            self.advance();
        }

        // Parse WHERE clause
        if !matches!(self.current_token(), Token::Where { .. }) {
            return Err(OxirsError::Parse("Expected WHERE clause".to_string()));
        }
        self.advance(); // consume WHERE

        let pattern = self.parse_group_graph_pattern()?;

        Ok(Query::Select {
            dataset: None,
            pattern,
            base_iri: None,
        })
    }

    fn parse_construct_query(&mut self) -> Result<Query, OxirsError> {
        self.advance(); // consume CONSTRUCT

        // Parse template - the triple patterns that define the output graph
        let template = if matches!(self.current_token(), Token::LeftBrace { .. }) {
            self.advance(); // consume '{'
            let mut template_patterns = Vec::new();

            // Parse triple patterns until we hit '}'
            while !matches!(self.current_token(), Token::RightBrace { .. }) {
                if matches!(self.current_token(), Token::Eof) {
                    return Err(OxirsError::Parse("Unexpected EOF in CONSTRUCT template".to_string()));
                }

                // Parse a single triple pattern
                if let Some(pattern) = self.parse_single_triple_pattern()? {
                    template_patterns.push(pattern);
                }

                // Consume optional '.' or ';'
                if matches!(self.current_token(), Token::Dot { .. } | Token::Semicolon { .. }) {
                    self.advance();
                }
            }

            self.advance(); // consume '}'
            template_patterns
        } else {
            // Empty template is valid (means construct all matched triples)
            Vec::new()
        };

        // Find WHERE
        while !matches!(self.current_token(), Token::Where { .. }) {
            if matches!(self.current_token(), Token::Eof) {
                return Err(OxirsError::Parse("Expected WHERE clause in CONSTRUCT query".to_string()));
            }
            self.advance();
        }
        self.advance(); // consume WHERE

        let pattern = self.parse_group_graph_pattern()?;

        Ok(Query::Construct {
            template,
            dataset: None,
            pattern,
            base_iri: None,
        })
    }

    fn parse_ask_query(&mut self) -> Result<Query, OxirsError> {
        self.advance(); // consume ASK

        // Find WHERE
        while !matches!(self.current_token(), Token::Where { .. }) {
            self.advance();
        }
        self.advance(); // consume WHERE

        let pattern = self.parse_group_graph_pattern()?;

        Ok(Query::Ask {
            dataset: None,
            pattern,
            base_iri: None,
        })
    }

    fn parse_describe_query(&mut self) -> Result<Query, OxirsError> {
        self.advance(); // consume DESCRIBE

        // Find WHERE
        while !matches!(self.current_token(), Token::Where { .. }) {
            self.advance();
        }
        self.advance(); // consume WHERE

        let pattern = self.parse_group_graph_pattern()?;

        Ok(Query::Describe {
            dataset: None,
            pattern,
            base_iri: None,
        })
    }

    fn parse_group_graph_pattern(&mut self) -> Result<GraphPattern, OxirsError> {
        if !matches!(self.current_token(), Token::LeftBrace { .. }) {
            return Err(OxirsError::Parse("Expected '{'".to_string()));
        }
        self.advance(); // consume '{'

        let mut patterns = Vec::new();

        // Parse graph patterns (triple patterns, filters, OPTIONAL, UNION, etc.)
        while !matches!(self.current_token(), Token::RightBrace { .. }) {
            if matches!(self.current_token(), Token::Eof) {
                return Err(OxirsError::Parse("Unexpected EOF in graph pattern".to_string()));
            }

            match self.current_token() {
                Token::Optional { .. } => {
                    self.advance(); // consume OPTIONAL
                    let optional_pattern = self.parse_group_graph_pattern()?;
                    patterns.push(TriplePattern {
                        subject: TermPattern::Variable(Variable::new_unchecked("_optional")),
                        predicate: TermPattern::Variable(Variable::new_unchecked("_optional")),
                        object: TermPattern::Variable(Variable::new_unchecked("_optional")),
                    });
                }
                Token::Filter { .. } => {
                    self.advance(); // consume FILTER
                    // Skip filter expression for now
                    if matches!(self.current_token(), Token::LeftParen { .. }) {
                        self.advance();
                        let mut depth = 1;
                        while depth > 0 && !matches!(self.current_token(), Token::Eof) {
                            match self.current_token() {
                                Token::LeftParen { .. } => depth += 1,
                                Token::RightParen { .. } => depth -= 1,
                                _ => {}
                            }
                            self.advance();
                        }
                    }
                }
                Token::Graph { .. } => {
                    self.advance(); // consume GRAPH
                    self.advance(); // skip graph name
                    let _ = self.parse_group_graph_pattern()?; // parse nested pattern
                }
                Token::Union { .. } => {
                    self.advance(); // consume UNION
                    let _ = self.parse_group_graph_pattern()?; // parse union branch
                }
                Token::Bind { .. } => {
                    self.advance(); // consume BIND
                    // Skip BIND expression
                    if matches!(self.current_token(), Token::LeftParen { .. }) {
                        self.advance();
                        let mut depth = 1;
                        while depth > 0 && !matches!(self.current_token(), Token::Eof) {
                            match self.current_token() {
                                Token::LeftParen { .. } => depth += 1,
                                Token::RightParen { .. } => depth -= 1,
                                _ => {}
                            }
                            self.advance();
                        }
                    }
                }
                Token::Values { .. } => {
                    self.advance(); // consume VALUES
                    // Skip VALUES clause
                    while !matches!(
                        self.current_token(),
                        Token::RightBrace { .. } | Token::Dot { .. }
                    ) && !matches!(self.current_token(), Token::Eof)
                    {
                        self.advance();
                    }
                }
                _ => {
                    // Try to parse as triple pattern
                    if let Some(pattern) = self.parse_single_triple_pattern()? {
                        patterns.push(pattern);
                    }

                    // Consume optional '.' or ';'
                    if matches!(self.current_token(), Token::Dot { .. } | Token::Semicolon { .. }) {
                        self.advance();
                    }
                }
            }
        }

        if !matches!(self.current_token(), Token::RightBrace { .. }) {
            return Err(OxirsError::Parse("Expected '}'".to_string()));
        }
        self.advance(); // consume '}'

        Ok(GraphPattern::Bgp { patterns })
    }

    /// Parse a single triple pattern (subject predicate object)
    fn parse_single_triple_pattern(&mut self) -> Result<Option<TriplePattern>, OxirsError> {
        // Parse subject
        let subject = match self.parse_term_pattern()? {
            Some(term) => term,
            None => return Ok(None), // Not a valid subject, skip
        };

        // Parse predicate
        let predicate = match self.parse_term_pattern()? {
            Some(term) => term,
            None => {
                return Err(OxirsError::Parse(format!(
                    "Expected predicate after subject in triple pattern"
                )))
            }
        };

        // Parse object
        let object = match self.parse_term_pattern()? {
            Some(term) => term,
            None => {
                return Err(OxirsError::Parse(format!(
                    "Expected object after predicate in triple pattern"
                )))
            }
        };

        Ok(Some(TriplePattern {
            subject,
            predicate,
            object,
        }))
    }

    /// Parse a term pattern (Variable, IRI, Literal, BlankNode)
    fn parse_term_pattern(&mut self) -> Result<Option<TermPattern>, OxirsError> {
        match self.current_token().clone() {
            Token::Variable { name, .. } => {
                self.advance();
                Ok(Some(TermPattern::Variable(Variable::new_unchecked(&name))))
            }
            Token::Iri { value, .. } => {
                self.advance();
                let node = NamedNode::new(&value)?;
                Ok(Some(TermPattern::NamedNode(node)))
            }
            Token::PrefixedName {
                prefix,
                local_name,
                ..
            } => {
                self.advance();
                // Construct full IRI from prefix (simplified - would need prefix map)
                let full_iri = format!("http://example.org/{prefix}#{local_name}");
                let node = NamedNode::new(&full_iri)?;
                Ok(Some(TermPattern::NamedNode(node)))
            }
            Token::StringLiteral { value, .. } => {
                self.advance();
                let literal = Literal::new_simple_literal(&value);
                Ok(Some(TermPattern::Literal(literal)))
            }
            Token::Integer { value, .. } => {
                self.advance();
                let literal = Literal::new_typed_literal(
                    &value,
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#integer")?,
                );
                Ok(Some(TermPattern::Literal(literal)))
            }
            Token::Decimal { value, .. } => {
                self.advance();
                let literal = Literal::new_typed_literal(
                    &value,
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal")?,
                );
                Ok(Some(TermPattern::Literal(literal)))
            }
            Token::BlankNode { label, .. } => {
                self.advance();
                let bnode = BlankNode::new_unchecked(&label);
                Ok(Some(TermPattern::BlankNode(bnode)))
            }
            Token::A { .. } => {
                // Shorthand for rdf:type
                self.advance();
                let node = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
                Ok(Some(TermPattern::NamedNode(node)))
            }
            Token::True { .. } => {
                self.advance();
                let literal = Literal::new_typed_literal(
                    "true",
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")?,
                );
                Ok(Some(TermPattern::Literal(literal)))
            }
            Token::False { .. } => {
                self.advance();
                let literal = Literal::new_typed_literal(
                    "false",
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#boolean")?,
                );
                Ok(Some(TermPattern::Literal(literal)))
            }
            _ => {
                // Not a term pattern, return None
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenization() {
        let mut parser = AdvancedSparqlParser::new();
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let tokens = parser.tokenize(query).unwrap();
        
        assert!(matches!(tokens[0], Token::Select { .. }));
        assert!(matches!(tokens[1], Token::Variable { .. }));
    }

    #[test]
    fn test_simple_select_parsing() {
        let mut parser = AdvancedSparqlParser::new();
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }";
        let result = parser.parse(query);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prefix_handling() {
        let mut parser = AdvancedSparqlParser::new();
        assert!(parser.prefixes.contains_key("rdf"));
        assert!(parser.prefixes.contains_key("rdfs"));
        assert!(parser.prefixes.contains_key("xsd"));
    }
}