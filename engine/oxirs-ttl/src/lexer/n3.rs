//! N3/Turtle lexer implementation
//!
//! This lexer handles N3-specific syntax including:
//! - Variables (?var)
//! - Formulas ({ ... })
//! - Implications (=>)
//! - Quantifiers (@forAll, @forSome)
//! - Built-in predicates

use crate::error::{TextPosition, TurtleParseError, TurtleResult, TurtleSyntaxError};

/// N3 token types
#[derive(Debug, Clone, PartialEq)]
pub enum N3Token {
    /// Variable (?name)
    Variable(String),
    /// Left brace for formula
    LeftBrace,
    /// Right brace for formula
    RightBrace,
    /// Implication operator (=>)
    Implies,
    /// Reverse implication operator (<=)
    ImpliedBy,
    /// @forAll quantifier
    ForAll,
    /// @forSome quantifier
    ForSome,
    /// IRI enclosed in angle brackets
    Iri(String),
    /// Prefixed name (prefix:local)
    PrefixedName {
        /// Prefix part before the colon
        prefix: String,
        /// Local part after the colon
        local: String,
    },
    /// Blank node (_:label)
    BlankNode(String),
    /// String literal
    StringLiteral(String),
    /// Integer literal
    IntegerLiteral(String),
    /// Decimal literal
    DecimalLiteral(String),
    /// Language tag (@lang)
    LanguageTag(String),
    /// Datatype marker (^^)
    DatatypeMarker,
    /// Prefix declaration (@prefix)
    PrefixDecl,
    /// Base declaration (@base)
    BaseDecl,
    /// Dot (.)
    Dot,
    /// Semicolon (;)
    Semicolon,
    /// Comma (,)
    Comma,
    /// Left parenthesis
    LeftParen,
    /// Right parenthesis
    RightParen,
    /// Left bracket
    LeftBracket,
    /// Right bracket
    RightBracket,
    /// 'a' shorthand for rdf:type
    RdfType,
    /// End of input
    Eof,
}

/// N3 lexer for tokenizing N3 syntax
pub struct N3Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
}

impl N3Lexer {
    /// Create a new N3 lexer
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    /// Get the next token
    pub fn next_token(&mut self) -> TurtleResult<N3Token> {
        self.skip_whitespace_and_comments();

        if self.is_at_end() {
            return Ok(N3Token::Eof);
        }

        let ch = self.peek();

        match ch {
            '?' => self.read_variable(),
            '{' => {
                self.advance();
                Ok(N3Token::LeftBrace)
            }
            '}' => {
                self.advance();
                Ok(N3Token::RightBrace)
            }
            '=' => {
                self.advance();
                if self.peek() == '>' {
                    self.advance();
                    Ok(N3Token::Implies)
                } else {
                    self.syntax_error("Expected '>' after '='")
                }
            }
            '<' => {
                self.advance();
                if self.peek() == '=' {
                    self.advance();
                    Ok(N3Token::ImpliedBy)
                } else {
                    // It's an IRI
                    self.position -= 1;
                    self.read_iri()
                }
            }
            '.' => {
                self.advance();
                Ok(N3Token::Dot)
            }
            ';' => {
                self.advance();
                Ok(N3Token::Semicolon)
            }
            ',' => {
                self.advance();
                Ok(N3Token::Comma)
            }
            '(' => {
                self.advance();
                Ok(N3Token::LeftParen)
            }
            ')' => {
                self.advance();
                Ok(N3Token::RightParen)
            }
            '[' => {
                self.advance();
                Ok(N3Token::LeftBracket)
            }
            ']' => {
                self.advance();
                Ok(N3Token::RightBracket)
            }
            '"' => self.read_string_literal(),
            '@' => self.read_at_keyword(),
            '_' if self.peek_ahead(1) == ':' => self.read_blank_node(),
            'a' if self.is_separator(self.peek_ahead(1)) => {
                self.advance();
                Ok(N3Token::RdfType)
            }
            '0'..='9' | '+' | '-' => self.read_numeric_literal(),
            _ if ch.is_alphabetic() || ch == '_' => self.read_prefixed_name_or_keyword(),
            '^' if self.peek_ahead(1) == '^' => {
                self.advance();
                self.advance();
                Ok(N3Token::DatatypeMarker)
            }
            _ => self.syntax_error(&format!("Unexpected character: '{}'", ch)),
        }
    }

    /// Read a variable (?name)
    fn read_variable(&mut self) -> TurtleResult<N3Token> {
        self.advance(); // skip '?'
        let mut name = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch.is_alphanumeric() || ch == '_' {
                name.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if name.is_empty() {
            self.syntax_error("Variable name cannot be empty")
        } else {
            Ok(N3Token::Variable(name))
        }
    }

    /// Read an IRI (<...>)
    fn read_iri(&mut self) -> TurtleResult<N3Token> {
        self.advance(); // skip '<'
        let mut iri = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch == '>' {
                self.advance();
                return Ok(N3Token::Iri(iri));
            } else if ch == '\\' {
                self.advance();
                if !self.is_at_end() {
                    iri.push(self.peek());
                    self.advance();
                }
            } else {
                iri.push(ch);
                self.advance();
            }
        }

        self.syntax_error("Unterminated IRI")
    }

    /// Read a string literal
    fn read_string_literal(&mut self) -> TurtleResult<N3Token> {
        self.advance(); // skip opening quote

        // Check for triple-quoted strings
        if self.peek() == '"' && self.peek_ahead(1) == '"' {
            return self.read_triple_quoted_string();
        }

        let mut literal = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch == '"' {
                self.advance();
                return Ok(N3Token::StringLiteral(literal));
            } else if ch == '\\' {
                self.advance();
                if !self.is_at_end() {
                    let escaped = self.read_escape_sequence()?;
                    literal.push_str(&escaped);
                }
            } else {
                literal.push(ch);
                self.advance();
            }
        }

        self.syntax_error("Unterminated string literal")
    }

    /// Read a triple-quoted string ("""...""")
    fn read_triple_quoted_string(&mut self) -> TurtleResult<N3Token> {
        self.advance(); // skip second quote
        self.advance(); // skip third quote

        let mut literal = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch == '"' && self.peek_ahead(1) == '"' && self.peek_ahead(2) == '"' {
                self.advance();
                self.advance();
                self.advance();
                return Ok(N3Token::StringLiteral(literal));
            } else if ch == '\\' {
                self.advance();
                if !self.is_at_end() {
                    let escaped = self.read_escape_sequence()?;
                    literal.push_str(&escaped);
                }
            } else {
                literal.push(ch);
                self.advance();
                if ch == '\n' {
                    self.line += 1;
                    self.column = 1;
                }
            }
        }

        self.syntax_error("Unterminated triple-quoted string")
    }

    /// Read an escape sequence
    fn read_escape_sequence(&mut self) -> TurtleResult<String> {
        if self.is_at_end() {
            return self.syntax_error("Incomplete escape sequence");
        }

        let ch = self.peek();
        self.advance();

        match ch {
            'n' => Ok("\n".to_string()),
            'r' => Ok("\r".to_string()),
            't' => Ok("\t".to_string()),
            '"' => Ok("\"".to_string()),
            '\\' => Ok("\\".to_string()),
            'u' => self.read_unicode_escape(4),
            'U' => self.read_unicode_escape(8),
            _ => Ok(ch.to_string()),
        }
    }

    /// Read a Unicode escape sequence
    fn read_unicode_escape(&mut self, len: usize) -> TurtleResult<String> {
        let mut code = String::new();
        for _ in 0..len {
            if self.is_at_end() {
                return self.syntax_error("Incomplete Unicode escape");
            }
            let ch = self.peek();
            if ch.is_ascii_hexdigit() {
                code.push(ch);
                self.advance();
            } else {
                return self.syntax_error("Invalid hex digit in Unicode escape");
            }
        }

        let code_point = u32::from_str_radix(&code, 16)
            .map_err(|_| self.create_error("Invalid Unicode code point"))?;

        char::from_u32(code_point)
            .ok_or_else(|| self.create_error("Invalid Unicode code point"))
            .map(|c| c.to_string())
    }

    /// Read an @-keyword (@prefix, @base, @forAll, @forSome)
    fn read_at_keyword(&mut self) -> TurtleResult<N3Token> {
        self.advance(); // skip '@'
        let mut keyword = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch.is_alphabetic() || ch == '-' {
                keyword.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        match keyword.as_str() {
            "prefix" => Ok(N3Token::PrefixDecl),
            "base" => Ok(N3Token::BaseDecl),
            "forAll" => Ok(N3Token::ForAll),
            "forSome" => Ok(N3Token::ForSome),
            _ => {
                // It's a language tag
                Ok(N3Token::LanguageTag(keyword))
            }
        }
    }

    /// Read a blank node (_:label)
    fn read_blank_node(&mut self) -> TurtleResult<N3Token> {
        self.advance(); // skip '_'
        self.advance(); // skip ':'

        let mut label = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                label.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        if label.is_empty() {
            self.syntax_error("Blank node label cannot be empty")
        } else {
            Ok(N3Token::BlankNode(label))
        }
    }

    /// Read a prefixed name or keyword
    fn read_prefixed_name_or_keyword(&mut self) -> TurtleResult<N3Token> {
        let mut name = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                name.push(ch);
                self.advance();
            } else if ch == ':' {
                self.advance();
                return self.read_prefixed_name_local(name);
            } else {
                break;
            }
        }

        // No colon found - it's a prefixed name with empty prefix (":local")
        // or a standalone keyword, but we'll treat it as a prefixed name
        Ok(N3Token::PrefixedName {
            prefix: String::new(),
            local: name,
        })
    }

    /// Read the local part of a prefixed name after the colon
    fn read_prefixed_name_local(&mut self, prefix: String) -> TurtleResult<N3Token> {
        let mut local = String::new();

        while !self.is_at_end() {
            let ch = self.peek();
            if ch.is_alphanumeric() || ch == '_' || ch == '-' || ch == '.' {
                local.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        Ok(N3Token::PrefixedName { prefix, local })
    }

    /// Read a numeric literal
    fn read_numeric_literal(&mut self) -> TurtleResult<N3Token> {
        let mut num = String::new();
        let mut is_decimal = false;

        // Handle sign
        if self.peek() == '+' || self.peek() == '-' {
            num.push(self.peek());
            self.advance();
        }

        // Read digits
        while !self.is_at_end() && self.peek().is_ascii_digit() {
            num.push(self.peek());
            self.advance();
        }

        // Check for decimal point
        if !self.is_at_end() && self.peek() == '.' && !self.is_separator(self.peek_ahead(1)) {
            is_decimal = true;
            num.push('.');
            self.advance();

            while !self.is_at_end() && self.peek().is_ascii_digit() {
                num.push(self.peek());
                self.advance();
            }
        }

        // Check for exponent
        if !self.is_at_end() && (self.peek() == 'e' || self.peek() == 'E') {
            is_decimal = true;
            num.push(self.peek());
            self.advance();

            if !self.is_at_end() && (self.peek() == '+' || self.peek() == '-') {
                num.push(self.peek());
                self.advance();
            }

            while !self.is_at_end() && self.peek().is_ascii_digit() {
                num.push(self.peek());
                self.advance();
            }
        }

        if is_decimal {
            Ok(N3Token::DecimalLiteral(num))
        } else {
            Ok(N3Token::IntegerLiteral(num))
        }
    }

    /// Skip whitespace and comments
    fn skip_whitespace_and_comments(&mut self) {
        while !self.is_at_end() {
            let ch = self.peek();
            if ch.is_whitespace() {
                if ch == '\n' {
                    self.line += 1;
                    self.column = 1;
                } else {
                    self.column += 1;
                }
                self.advance();
            } else if ch == '#' {
                // Skip comment until end of line
                while !self.is_at_end() && self.peek() != '\n' {
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    /// Check if character is a separator (whitespace or special character)
    fn is_separator(&self, ch: char) -> bool {
        ch.is_whitespace() || matches!(ch, '.' | ';' | ',' | '(' | ')' | '[' | ']' | '{' | '}')
    }

    /// Peek at the current character
    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.input[self.position]
        }
    }

    /// Peek ahead n characters
    fn peek_ahead(&self, n: usize) -> char {
        let pos = self.position + n;
        if pos >= self.input.len() {
            '\0'
        } else {
            self.input[pos]
        }
    }

    /// Advance to the next character
    fn advance(&mut self) {
        if !self.is_at_end() {
            self.position += 1;
            self.column += 1;
        }
    }

    /// Check if we're at the end of input
    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }

    /// Get the current position
    pub fn current_position(&self) -> TextPosition {
        TextPosition {
            line: self.line,
            column: self.column,
            offset: self.position,
        }
    }

    /// Create a syntax error
    fn syntax_error<T>(&self, message: &str) -> TurtleResult<T> {
        Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: message.to_string(),
            position: self.current_position(),
        }))
    }

    /// Create an error
    fn create_error(&self, message: &str) -> TurtleParseError {
        TurtleParseError::syntax(TurtleSyntaxError::Generic {
            message: message.to_string(),
            position: self.current_position(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_lexing() {
        let mut lexer = N3Lexer::new("?x ?name ?_var");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Variable("x".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Variable("name".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Variable("_var".to_string())
        );
    }

    #[test]
    fn test_formula_braces() {
        let mut lexer = N3Lexer::new("{ }");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::LeftBrace
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::RightBrace
        );
    }

    #[test]
    fn test_implication_operators() {
        let mut lexer = N3Lexer::new("=> <=");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Implies
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::ImpliedBy
        );
    }

    #[test]
    fn test_quantifiers() {
        let mut lexer = N3Lexer::new("@forAll @forSome");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::ForAll
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::ForSome
        );
    }

    #[test]
    fn test_prefixed_name() {
        let mut lexer = N3Lexer::new("ex:name rdf:type");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::PrefixedName {
                prefix: "ex".to_string(),
                local: "name".to_string()
            }
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::PrefixedName {
                prefix: "rdf".to_string(),
                local: "type".to_string()
            }
        );
    }

    #[test]
    fn test_iri() {
        let mut lexer = N3Lexer::new("<http://example.org/resource>");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Iri("http://example.org/resource".to_string())
        );
    }

    #[test]
    fn test_string_literal() {
        let mut lexer = N3Lexer::new(r#""Hello World""#);
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::StringLiteral("Hello World".to_string())
        );
    }

    #[test]
    fn test_numeric_literals() {
        let mut lexer = N3Lexer::new("42 3.14 -5");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::IntegerLiteral("42".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::DecimalLiteral("3.14".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::IntegerLiteral("-5".to_string())
        );
    }

    #[test]
    fn test_blank_node() {
        let mut lexer = N3Lexer::new("_:b1 _:node123");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::BlankNode("b1".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::BlankNode("node123".to_string())
        );
    }

    #[test]
    fn test_punctuation() {
        let mut lexer = N3Lexer::new(". ; , ( ) [ ]");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Dot
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Semicolon
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Comma
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::LeftParen
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::RightParen
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::LeftBracket
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::RightBracket
        );
    }

    #[test]
    fn test_comments() {
        let mut lexer = N3Lexer::new("?x # this is a comment\n?y");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Variable("x".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Variable("y".to_string())
        );
    }

    #[test]
    fn test_complete_n3_statement() {
        let mut lexer = N3Lexer::new("{ ?x ex:knows ?y } => { ?y ex:knows ?x } .");
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::LeftBrace
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Variable("x".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::PrefixedName {
                prefix: "ex".to_string(),
                local: "knows".to_string()
            }
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Variable("y".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::RightBrace
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Implies
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::LeftBrace
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Variable("y".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::PrefixedName {
                prefix: "ex".to_string(),
                local: "knows".to_string()
            }
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Variable("x".to_string())
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::RightBrace
        );
        assert_eq!(
            lexer.next_token().expect("token should be available"),
            N3Token::Dot
        );
    }
}
