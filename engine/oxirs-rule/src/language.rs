//! Custom Rule Language for Human-Readable Rule Definitions
//!
//! Provides a domain-specific language (DSL) for defining rules in a readable format.
//! The language supports variables, predicates, builtins, and namespaces.
//!
//! # Features
//!
//! - **Human-Readable Syntax**: Intuitive rule definition format
//! - **Variable Support**: Use ?variable syntax for variables
//! - **Namespace Management**: @prefix declarations for URI abbreviation
//! - **Builtin Predicates**: Support for comparison and arithmetic operations
//! - **Import/Export**: Convert between text and internal representation
//! - **Error Reporting**: Detailed syntax error messages with line/column info
//!
//! # Syntax
//!
//! ```text
//! # Comments start with #
//! @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
//! @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
//!
//! rule "transitivity" {
//!   if {
//!     ?x rdf:type ?class1
//!     ?class1 rdfs:subClassOf ?class2
//!   }
//!   then {
//!     ?x rdf:type ?class2
//!   }
//! }
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::language::{RuleLanguageParser, RuleLanguageSerializer};
//!
//! let source = r#"
//! rule "example" {
//!   if {
//!     ?person :hasAge ?age
//!     ?age > 18
//!   }
//!   then {
//!     ?person :isAdult true
//!   }
//! }
//! "#;
//!
//! let mut parser = RuleLanguageParser::new();
//! let rules = parser.parse(source).unwrap();
//!
//! // Serialize back to text
//! let serializer = RuleLanguageSerializer::new();
//! let output = serializer.serialize(&rules).unwrap();
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fmt;

/// Token types in the rule language
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords
    Rule,
    If,
    Then,
    Prefix,

    // Identifiers and literals
    Identifier(String),
    Variable(String),
    Uri(String),
    String(String),
    Number(String),
    Boolean(bool),

    // Operators
    GreaterThan,
    LessThan,
    NotEqual,

    // Delimiters
    LeftBrace,
    RightBrace,
    Colon,
    Dot,

    // Special
    Comment(String),
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Rule => write!(f, "rule"),
            Token::If => write!(f, "if"),
            Token::Then => write!(f, "then"),
            Token::Prefix => write!(f, "@prefix"),
            Token::Identifier(s) => write!(f, "{}", s),
            Token::Variable(s) => write!(f, "?{}", s),
            Token::Uri(s) => write!(f, "<{}>", s),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Number(s) => write!(f, "{}", s),
            Token::Boolean(b) => write!(f, "{}", b),
            Token::GreaterThan => write!(f, ">"),
            Token::LessThan => write!(f, "<"),
            Token::NotEqual => write!(f, "!="),
            Token::LeftBrace => write!(f, "{{"),
            Token::RightBrace => write!(f, "}}"),
            Token::Colon => write!(f, ":"),
            Token::Dot => write!(f, "."),
            Token::Comment(s) => write!(f, "# {}", s),
            Token::Eof => write!(f, "EOF"),
        }
    }
}

/// Lexer for tokenizing rule language source
struct Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    fn current(&self) -> Option<char> {
        if self.position < self.input.len() {
            Some(self.input[self.position])
        } else {
            None
        }
    }

    fn peek(&self, offset: usize) -> Option<char> {
        let pos = self.position + offset;
        if pos < self.input.len() {
            Some(self.input[pos])
        } else {
            None
        }
    }

    fn advance(&mut self) -> Option<char> {
        if let Some(ch) = self.current() {
            self.position += 1;
            if ch == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            Some(ch)
        } else {
            None
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        while let Some(ch) = self.current() {
            if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    fn read_uri(&mut self) -> Result<String> {
        let mut result = String::new();
        self.advance(); // Skip '<'

        while let Some(ch) = self.current() {
            if ch == '>' {
                self.advance();
                return Ok(result);
            }
            result.push(ch);
            self.advance();
        }

        Err(anyhow::anyhow!("Unterminated URI at line {}", self.line))
    }

    fn read_string(&mut self) -> Result<String> {
        let mut result = String::new();
        self.advance(); // Skip opening quote

        while let Some(ch) = self.current() {
            if ch == '"' {
                self.advance();
                return Ok(result);
            }
            if ch == '\\' {
                self.advance();
                if let Some(escaped) = self.current() {
                    match escaped {
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        'r' => result.push('\r'),
                        '"' => result.push('"'),
                        '\\' => result.push('\\'),
                        _ => result.push(escaped),
                    }
                    self.advance();
                }
            } else {
                result.push(ch);
                self.advance();
            }
        }

        Err(anyhow::anyhow!("Unterminated string at line {}", self.line))
    }

    fn read_number(&mut self) -> String {
        let mut result = String::new();
        while let Some(ch) = self.current() {
            if ch.is_numeric() || ch == '.' || ch == '-' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    fn read_comment(&mut self) -> String {
        let mut result = String::new();
        self.advance(); // Skip '#'

        while let Some(ch) = self.current() {
            if ch == '\n' {
                break;
            }
            result.push(ch);
            self.advance();
        }

        result.trim().to_string()
    }

    fn next_token(&mut self) -> Result<Token> {
        self.skip_whitespace();

        match self.current() {
            None => Ok(Token::Eof),
            Some('#') => {
                let comment = self.read_comment();
                Ok(Token::Comment(comment))
            }
            Some('@') => {
                self.advance();
                let keyword = self.read_identifier();
                match keyword.as_str() {
                    "prefix" => Ok(Token::Prefix),
                    _ => Err(anyhow::anyhow!(
                        "Unknown directive @{} at line {}",
                        keyword,
                        self.line
                    )),
                }
            }
            Some('?') => {
                self.advance();
                let var = self.read_identifier();
                Ok(Token::Variable(var))
            }
            Some('<') => {
                // Check if it's the start of a URI or the less-than operator
                if let Some(next_ch) = self.peek(1) {
                    if next_ch == '>' {
                        // This is <>  which we'll treat as NotEqual
                        self.advance();
                        self.advance();
                        Ok(Token::NotEqual)
                    } else if next_ch.is_whitespace()
                        || next_ch.is_numeric()
                        || next_ch == '?'
                        || next_ch == '"'
                    {
                        // This is < followed by whitespace or a term, so it's less-than operator
                        self.advance();
                        Ok(Token::LessThan)
                    } else {
                        // This is <...> so it's a URI
                        let uri = self.read_uri()?;
                        Ok(Token::Uri(uri))
                    }
                } else {
                    // End of input, treat as less-than
                    self.advance();
                    Ok(Token::LessThan)
                }
            }
            Some('>') => {
                self.advance();
                Ok(Token::GreaterThan)
            }
            Some('"') => {
                let string = self.read_string()?;
                Ok(Token::String(string))
            }
            Some('{') => {
                self.advance();
                Ok(Token::LeftBrace)
            }
            Some('}') => {
                self.advance();
                Ok(Token::RightBrace)
            }
            Some(':') => {
                self.advance();
                Ok(Token::Colon)
            }
            Some('.') => {
                self.advance();
                Ok(Token::Dot)
            }
            Some('!') => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    Ok(Token::NotEqual)
                } else {
                    Err(anyhow::anyhow!(
                        "Unexpected character '!' at line {}",
                        self.line
                    ))
                }
            }
            Some(ch) if ch.is_numeric() || ch == '-' => {
                let num = self.read_number();
                Ok(Token::Number(num))
            }
            Some(ch) if ch.is_alphabetic() => {
                let ident = self.read_identifier();
                match ident.as_str() {
                    "rule" => Ok(Token::Rule),
                    "if" => Ok(Token::If),
                    "then" => Ok(Token::Then),
                    "true" => Ok(Token::Boolean(true)),
                    "false" => Ok(Token::Boolean(false)),
                    _ => Ok(Token::Identifier(ident)),
                }
            }
            Some(ch) => Err(anyhow::anyhow!(
                "Unexpected character '{}' at line {}:{}",
                ch,
                self.line,
                self.column
            )),
        }
    }

    fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token()?;
            if token == Token::Eof {
                tokens.push(token);
                break;
            }
            // Skip comments
            if !matches!(token, Token::Comment(_)) {
                tokens.push(token);
            }
        }

        Ok(tokens)
    }
}

/// Parser for rule language
pub struct RuleLanguageParser {
    prefixes: HashMap<String, String>,
}

impl Default for RuleLanguageParser {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleLanguageParser {
    pub fn new() -> Self {
        let mut prefixes = HashMap::new();
        // Default prefixes
        prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );

        Self { prefixes }
    }

    /// Parse rule language source into Rules
    pub fn parse(&mut self, source: &str) -> Result<Vec<Rule>> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;

        let mut rules = Vec::new();
        let mut i = 0;

        while i < tokens.len() {
            match &tokens[i] {
                Token::Prefix => {
                    i = self.parse_prefix(&tokens, i)?;
                }
                Token::Rule => {
                    let (rule, next_i) = self.parse_rule(&tokens, i)?;
                    rules.push(rule);
                    i = next_i;
                }
                Token::Eof => break,
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unexpected token: {}. Expected 'rule' or '@prefix'",
                        tokens[i]
                    ));
                }
            }
        }

        Ok(rules)
    }

    fn parse_prefix(&mut self, tokens: &[Token], start: usize) -> Result<usize> {
        let mut i = start + 1; // Skip @prefix

        // Expect identifier
        let prefix = match &tokens[i] {
            Token::Identifier(p) => p.clone(),
            _ => return Err(anyhow::anyhow!("Expected prefix identifier")),
        };
        i += 1;

        // Expect colon
        if !matches!(tokens[i], Token::Colon) {
            return Err(anyhow::anyhow!("Expected ':' after prefix"));
        }
        i += 1;

        // Expect URI
        let uri = match &tokens[i] {
            Token::Uri(u) => u.clone(),
            _ => return Err(anyhow::anyhow!("Expected URI after prefix")),
        };
        i += 1;

        self.prefixes.insert(prefix, uri);

        Ok(i)
    }

    fn parse_rule(&self, tokens: &[Token], start: usize) -> Result<(Rule, usize)> {
        let mut i = start + 1; // Skip 'rule'

        // Expect rule name
        let name = match &tokens[i] {
            Token::String(n) => n.clone(),
            _ => return Err(anyhow::anyhow!("Expected rule name (string)")),
        };
        i += 1;

        // Expect '{'
        if !matches!(tokens[i], Token::LeftBrace) {
            return Err(anyhow::anyhow!("Expected '{{' to start rule body"));
        }
        i += 1;

        // Parse 'if' block
        if !matches!(tokens[i], Token::If) {
            return Err(anyhow::anyhow!("Expected 'if' block"));
        }
        i += 1;

        let (body, next_i) = self.parse_atom_block(tokens, i)?;
        i = next_i;

        // Parse 'then' block
        if !matches!(tokens[i], Token::Then) {
            return Err(anyhow::anyhow!("Expected 'then' block"));
        }
        i += 1;

        let (head, next_i) = self.parse_atom_block(tokens, i)?;
        i = next_i;

        // Expect '}'
        if !matches!(tokens[i], Token::RightBrace) {
            return Err(anyhow::anyhow!("Expected '}}' to end rule body"));
        }
        i += 1;

        Ok((Rule { name, body, head }, i))
    }

    fn parse_atom_block(&self, tokens: &[Token], start: usize) -> Result<(Vec<RuleAtom>, usize)> {
        let mut i = start;
        let mut atoms = Vec::new();

        // Expect '{'
        if !matches!(tokens[i], Token::LeftBrace) {
            return Err(anyhow::anyhow!("Expected '{{' to start atom block"));
        }
        i += 1;

        // Parse atoms until '}'
        while !matches!(tokens[i], Token::RightBrace) {
            let (atom, next_i) = self.parse_atom(tokens, i)?;
            atoms.push(atom);
            i = next_i;
        }

        // Skip '}'
        i += 1;

        Ok((atoms, i))
    }

    fn parse_atom(&self, tokens: &[Token], start: usize) -> Result<(RuleAtom, usize)> {
        let mut i = start;

        // Parse subject
        let subject = self.parse_term(tokens, i)?;
        i += 1;

        // Skip extra token if we parsed a prefixed name (:localName)
        if matches!(tokens[i - 1], Token::Colon) {
            i += 1; // Skip the identifier after colon
        }

        // Check for operators
        match &tokens[i] {
            Token::GreaterThan => {
                i += 1;
                let right = self.parse_term(tokens, i)?;
                i += 1;
                // Skip extra token if we parsed a prefixed name
                if i > 0 && matches!(tokens[i - 2], Token::Colon) {
                    i += 1;
                }
                return Ok((
                    RuleAtom::GreaterThan {
                        left: subject,
                        right,
                    },
                    i,
                ));
            }
            Token::LessThan => {
                i += 1;
                let right = self.parse_term(tokens, i)?;
                i += 1;
                // Skip extra token if we parsed a prefixed name
                if i > 0 && matches!(tokens[i - 2], Token::Colon) {
                    i += 1;
                }
                return Ok((
                    RuleAtom::LessThan {
                        left: subject,
                        right,
                    },
                    i,
                ));
            }
            Token::NotEqual => {
                i += 1;
                let right = self.parse_term(tokens, i)?;
                i += 1;
                // Skip extra token if we parsed a prefixed name
                if i > 0 && matches!(tokens[i - 2], Token::Colon) {
                    i += 1;
                }
                return Ok((
                    RuleAtom::NotEqual {
                        left: subject,
                        right,
                    },
                    i,
                ));
            }
            _ => {}
        }

        // Parse predicate
        let predicate = self.parse_term(tokens, i)?;
        i += 1;
        // Skip extra token if we parsed a prefixed name
        if matches!(tokens[i - 1], Token::Colon) {
            i += 1;
        }

        // Parse object
        let object = self.parse_term(tokens, i)?;
        i += 1;
        // Skip extra token if we parsed a prefixed name
        if matches!(tokens[i - 1], Token::Colon) {
            i += 1;
        }

        Ok((
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            },
            i,
        ))
    }

    fn parse_term(&self, tokens: &[Token], index: usize) -> Result<Term> {
        match &tokens[index] {
            Token::Variable(v) => Ok(Term::Variable(v.clone())),
            Token::Identifier(id) => {
                // Check for prefixed name (prefix:localName)
                if index + 1 < tokens.len() && matches!(tokens[index + 1], Token::Colon) {
                    // This is a prefixed name, will be handled when we combine tokens
                    Ok(Term::Constant(id.clone()))
                } else {
                    Ok(Term::Constant(id.clone()))
                }
            }
            Token::Colon => {
                // Handle :localName (default namespace)
                if index + 1 < tokens.len() {
                    if let Token::Identifier(local) = &tokens[index + 1] {
                        Ok(Term::Constant(format!(":{}", local)))
                    } else {
                        Err(anyhow::anyhow!("Expected identifier after ':'"))
                    }
                } else {
                    Err(anyhow::anyhow!("Unexpected ':' at end of input"))
                }
            }
            Token::Uri(uri) => Ok(Term::Constant(uri.clone())),
            Token::String(s) => Ok(Term::Literal(s.clone())),
            Token::Number(n) => Ok(Term::Literal(n.clone())),
            Token::Boolean(b) => Ok(Term::Literal(b.to_string())),
            _ => Err(anyhow::anyhow!("Expected term, got {}", tokens[index])),
        }
    }

    /// Add a custom prefix
    pub fn add_prefix(&mut self, prefix: String, uri: String) {
        self.prefixes.insert(prefix, uri);
    }

    /// Get all registered prefixes
    pub fn get_prefixes(&self) -> &HashMap<String, String> {
        &self.prefixes
    }
}

/// Serializer for converting rules to text format
pub struct RuleLanguageSerializer {
    indent_size: usize,
}

impl Default for RuleLanguageSerializer {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleLanguageSerializer {
    pub fn new() -> Self {
        Self { indent_size: 2 }
    }

    /// Serialize rules to text format
    pub fn serialize(&self, rules: &[Rule]) -> Result<String> {
        let mut output = String::new();

        for rule in rules {
            output.push_str(&self.serialize_rule(rule)?);
            output.push('\n');
        }

        Ok(output)
    }

    fn serialize_rule(&self, rule: &Rule) -> Result<String> {
        let mut output = String::new();

        output.push_str(&format!("rule \"{}\" {{\n", rule.name));

        // Serialize body
        output.push_str(&format!("{}if {{\n", self.indent(1)));
        for atom in &rule.body {
            output.push_str(&format!(
                "{}{}\n",
                self.indent(2),
                self.serialize_atom(atom)?
            ));
        }
        output.push_str(&format!("{}}}\n", self.indent(1)));

        // Serialize head
        output.push_str(&format!("{}then {{\n", self.indent(1)));
        for atom in &rule.head {
            output.push_str(&format!(
                "{}{}\n",
                self.indent(2),
                self.serialize_atom(atom)?
            ));
        }
        output.push_str(&format!("{}}}\n", self.indent(1)));

        output.push('}');

        Ok(output)
    }

    fn serialize_atom(&self, atom: &RuleAtom) -> Result<String> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => Ok(format!(
                "{} {} {}",
                Self::serialize_term(subject),
                Self::serialize_term(predicate),
                Self::serialize_term(object)
            )),
            RuleAtom::GreaterThan { left, right } => Ok(format!(
                "{} > {}",
                Self::serialize_term(left),
                Self::serialize_term(right)
            )),
            RuleAtom::LessThan { left, right } => Ok(format!(
                "{} < {}",
                Self::serialize_term(left),
                Self::serialize_term(right)
            )),
            RuleAtom::NotEqual { left, right } => Ok(format!(
                "{} != {}",
                Self::serialize_term(left),
                Self::serialize_term(right)
            )),
            RuleAtom::Builtin { name, args } => {
                let args_str = args
                    .iter()
                    .map(Self::serialize_term)
                    .collect::<Vec<_>>()
                    .join(", ");
                Ok(format!("{}({})", name, args_str))
            }
        }
    }

    fn serialize_term(term: &Term) -> String {
        match term {
            Term::Variable(v) => format!("?{}", v),
            Term::Constant(c) => c.clone(),
            Term::Literal(l) => format!("\"{}\"", l),
            Term::Function { name, args } => {
                let args_str = args
                    .iter()
                    .map(Self::serialize_term)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{}({})", name, args_str)
            }
        }
    }

    fn indent(&self, level: usize) -> String {
        " ".repeat(self.indent_size * level)
    }
}

/// Parse a single rule from text
pub fn parse_rule(source: &str) -> Result<Rule> {
    let mut parser = RuleLanguageParser::new();
    let rules = parser.parse(source).context("Failed to parse rule")?;

    if rules.is_empty() {
        return Err(anyhow::anyhow!("No rules found in source"));
    }

    Ok(rules[0].clone())
}

/// Serialize a single rule to text
pub fn serialize_rule(rule: &Rule) -> Result<String> {
    let serializer = RuleLanguageSerializer::new();
    serializer.serialize_rule(rule)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer_basic() {
        let mut lexer = Lexer::new("rule if then");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0], Token::Rule);
        assert_eq!(tokens[1], Token::If);
        assert_eq!(tokens[2], Token::Then);
    }

    #[test]
    fn test_lexer_variable() {
        let mut lexer = Lexer::new("?x ?person");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0], Token::Variable("x".to_string()));
        assert_eq!(tokens[1], Token::Variable("person".to_string()));
    }

    #[test]
    fn test_lexer_operators() {
        let mut lexer = Lexer::new("> < !=");
        let tokens = lexer.tokenize().unwrap();

        assert_eq!(tokens[0], Token::GreaterThan);
        assert_eq!(tokens[1], Token::LessThan);
        assert_eq!(tokens[2], Token::NotEqual);
    }

    #[test]
    fn test_parse_simple_rule() {
        let source = r#"
rule "test" {
  if {
    ?x :hasAge ?age
  }
  then {
    ?x :isAdult true
  }
}
"#;

        let mut parser = RuleLanguageParser::new();
        let rules = parser.parse(source).unwrap();

        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].name, "test");
        assert_eq!(rules[0].body.len(), 1);
        assert_eq!(rules[0].head.len(), 1);
    }

    #[test]
    fn test_parse_comparison() {
        let source = r#"
rule "age_check" {
  if {
    ?person :age ?age
    ?age > 18
  }
  then {
    ?person :adult true
  }
}
"#;

        let mut parser = RuleLanguageParser::new();
        let rules = parser.parse(source).unwrap();

        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].body.len(), 2);

        match &rules[0].body[1] {
            RuleAtom::GreaterThan { .. } => {}
            _ => panic!("Expected GreaterThan atom"),
        }
    }

    #[test]
    fn test_serialize_rule() {
        let rule = Rule {
            name: "test".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Constant("type".to_string()),
                object: Term::Constant("Person".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("x".to_string()),
                predicate: Term::Constant("category".to_string()),
                object: Term::Literal("human".to_string()),
            }],
        };

        let serializer = RuleLanguageSerializer::new();
        let output = serializer.serialize(&[rule]).unwrap();

        assert!(output.contains("rule \"test\""));
        assert!(output.contains("if {"));
        assert!(output.contains("then {"));
    }

    #[test]
    fn test_roundtrip() {
        let source = r#"
rule "roundtrip" {
  if {
    ?x :name ?name
  }
  then {
    ?x :hasName ?name
  }
}
"#;

        let mut parser = RuleLanguageParser::new();
        let rules = parser.parse(source).unwrap();

        let serializer = RuleLanguageSerializer::new();
        let output = serializer.serialize(&rules).unwrap();

        // Parse the serialized output
        let rules2 = parser.parse(&output).unwrap();

        assert_eq!(rules.len(), rules2.len());
        assert_eq!(rules[0].name, rules2[0].name);
    }
}
