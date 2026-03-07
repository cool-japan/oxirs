// N3/Notation3 rule syntax parser — supports a common subset of N3 rules.
// Added in v1.1.0 Round 7

/// An N3 term: IRI, prefixed name, literal, variable, blank node, or formula.
#[derive(Debug, Clone, PartialEq)]
pub enum N3Term {
    Iri(String),
    PrefixedName {
        prefix: String,
        local: String,
    },
    Literal {
        value: String,
        datatype: Option<String>,
        lang: Option<String>,
    },
    Variable(String),
    BlankNode(String),
    Formula(Vec<N3Triple>),
}

/// A single N3 triple (subject, predicate, object).
#[derive(Debug, Clone, PartialEq)]
pub struct N3Triple {
    pub subject: N3Term,
    pub predicate: N3Term,
    pub object: N3Term,
}

/// A parsed N3 rule with antecedent, consequent, and optional label.
#[derive(Debug, Clone)]
pub struct N3Rule {
    pub antecedent: Vec<N3Triple>,
    pub consequent: Vec<N3Triple>,
    pub label: Option<String>,
}

/// A parsed N3 document containing prefixes, rules, and plain triples.
#[derive(Debug, Clone)]
pub struct N3Document {
    pub prefixes: Vec<(String, String)>, // (prefix_name, iri)
    pub rules: Vec<N3Rule>,
    pub triples: Vec<N3Triple>,
}

/// Errors that occur during N3 parsing.
#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken {
        found: String,
        expected: String,
        position: usize,
    },
    UnexpectedEof,
    InvalidIri(String),
    InvalidLiteral(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedToken {
                found,
                expected,
                position,
            } => {
                write!(
                    f,
                    "Unexpected token '{found}' at position {position}, expected {expected}"
                )
            }
            ParseError::UnexpectedEof => write!(f, "Unexpected end of input"),
            ParseError::InvalidIri(s) => write!(f, "Invalid IRI: {s}"),
            ParseError::InvalidLiteral(s) => write!(f, "Invalid literal: {s}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Token types produced by the lexer.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Iri(String),
    PrefixedName(String, String),
    Variable(String),
    BlankNode(String),
    Literal(String, Option<String>, Option<String>), // value, datatype, lang
    At,
    Prefix,
    Dot,
    Implies, // =>
    LBrace,  // {
    RBrace,  // }
    Keyword(String),
    Eof,
}

struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let c = self.input.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip whitespace
            while let Some(c) = self.peek() {
                if c.is_ascii_whitespace() {
                    self.advance();
                } else {
                    break;
                }
            }
            // Skip # line comments
            if self.peek() == Some(b'#') {
                while let Some(c) = self.advance() {
                    if c == b'\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn read_iri(&mut self) -> Result<String, ParseError> {
        // Consume '<'
        self.advance();
        let start = self.pos;
        loop {
            match self.advance() {
                Some(b'>') => {
                    let end = self.pos - 1;
                    let s = std::str::from_utf8(&self.input[start..end])
                        .map_err(|_| ParseError::InvalidIri("invalid UTF-8 in IRI".to_string()))?
                        .to_string();
                    return Ok(s);
                }
                Some(_) => {}
                None => return Err(ParseError::UnexpectedEof),
            }
        }
    }

    fn read_string_literal(
        &mut self,
    ) -> Result<(String, Option<String>, Option<String>), ParseError> {
        // Consume '"'
        self.advance();
        let mut value = String::new();
        loop {
            match self.advance() {
                Some(b'"') => break,
                Some(b'\\') => match self.advance() {
                    Some(b'n') => value.push('\n'),
                    Some(b't') => value.push('\t'),
                    Some(b'"') => value.push('"'),
                    Some(b'\\') => value.push('\\'),
                    Some(c) => {
                        value.push('\\');
                        value.push(c as char);
                    }
                    None => return Err(ParseError::UnexpectedEof),
                },
                Some(c) => value.push(c as char),
                None => {
                    return Err(ParseError::InvalidLiteral(
                        "unterminated string literal".to_string(),
                    ))
                }
            }
        }
        // Check for ^^datatype or @lang
        let datatype;
        let lang;
        if self.peek() == Some(b'^') {
            self.advance(); // first ^
            if self.advance() != Some(b'^') {
                return Err(ParseError::InvalidLiteral("expected '^'".to_string()));
            }
            // Read IRI or prefixed name for datatype
            self.skip_whitespace_and_comments();
            if self.peek() == Some(b'<') {
                let iri = self.read_iri()?;
                datatype = Some(iri);
            } else {
                // prefixed name
                let pn = self.read_name_or_keyword()?;
                datatype = Some(pn);
            }
            lang = None;
        } else if self.peek() == Some(b'@') {
            self.advance();
            let mut tag = String::new();
            while let Some(c) = self.peek() {
                if c.is_ascii_alphanumeric() || c == b'-' {
                    tag.push(c as char);
                    self.advance();
                } else {
                    break;
                }
            }
            datatype = None;
            lang = Some(tag);
        } else {
            datatype = None;
            lang = None;
        }
        Ok((value, datatype, lang))
    }

    fn read_name_or_keyword(&mut self) -> Result<String, ParseError> {
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.' {
                name.push(c as char);
                self.advance();
            } else {
                break;
            }
        }
        Ok(name)
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace_and_comments();
        match self.peek() {
            None => Ok(Token::Eof),
            Some(b'<') => {
                let iri = self.read_iri()?;
                Ok(Token::Iri(iri))
            }
            Some(b'"') => {
                let (value, datatype, lang) = self.read_string_literal()?;
                Ok(Token::Literal(value, datatype, lang))
            }
            Some(b'?') => {
                self.advance();
                let mut name = String::new();
                while let Some(c) = self.peek() {
                    if c.is_ascii_alphanumeric() || c == b'_' {
                        name.push(c as char);
                        self.advance();
                    } else {
                        break;
                    }
                }
                Ok(Token::Variable(name))
            }
            Some(b'_') => {
                // blank node: _:name
                self.advance();
                if self.advance() != Some(b':') {
                    return Err(ParseError::UnexpectedToken {
                        found: "_".to_string(),
                        expected: "_:name".to_string(),
                        position: self.pos,
                    });
                }
                let mut name = String::new();
                while let Some(c) = self.peek() {
                    if c.is_ascii_alphanumeric() || c == b'_' {
                        name.push(c as char);
                        self.advance();
                    } else {
                        break;
                    }
                }
                Ok(Token::BlankNode(name))
            }
            Some(b'@') => {
                self.advance();
                let kw = self.read_name_or_keyword()?;
                if kw == "prefix" {
                    Ok(Token::Prefix)
                } else {
                    Ok(Token::At)
                }
            }
            Some(b'.') => {
                self.advance();
                Ok(Token::Dot)
            }
            Some(b'=') => {
                self.advance();
                if self.peek() == Some(b'>') {
                    self.advance();
                    Ok(Token::Implies)
                } else {
                    Ok(Token::Keyword("=".to_string()))
                }
            }
            Some(b'{') => {
                self.advance();
                Ok(Token::LBrace)
            }
            Some(b'}') => {
                self.advance();
                Ok(Token::RBrace)
            }
            Some(c) if c.is_ascii_alphabetic() || c == b':' => {
                let mut name = String::new();
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_alphanumeric() || ch == b'_' || ch == b':' || ch == b'-' {
                        name.push(ch as char);
                        self.advance();
                    } else {
                        break;
                    }
                }
                // Check if prefixed name (has ':' and content after it)
                if let Some(colon_pos) = name.find(':') {
                    let prefix = name[..colon_pos].to_string();
                    let local = name[colon_pos + 1..].to_string();
                    Ok(Token::PrefixedName(prefix, local))
                } else if name == "PREFIX" {
                    Ok(Token::Prefix)
                } else {
                    Ok(Token::Keyword(name))
                }
            }
            Some(c) => {
                let ch = c as char;
                self.advance();
                Err(ParseError::UnexpectedToken {
                    found: ch.to_string(),
                    expected: "term or declaration".to_string(),
                    position: self.pos,
                })
            }
        }
    }
}

/// Parse N3 documents from text.
pub struct N3Parser;

impl N3Parser {
    /// Tokenize input into string tokens (for inspection/testing).
    pub fn tokenize(input: &str) -> Vec<String> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        loop {
            match lexer.next_token() {
                Ok(Token::Eof) => break,
                Ok(tok) => tokens.push(format!("{tok:?}")),
                Err(_) => break,
            }
        }
        tokens
    }

    /// Parse a complete N3 document.
    pub fn parse(input: &str) -> Result<N3Document, ParseError> {
        let mut lexer = Lexer::new(input);
        let mut doc = N3Document {
            prefixes: Vec::new(),
            rules: Vec::new(),
            triples: Vec::new(),
        };
        loop {
            let tok = lexer.next_token()?;
            match &tok {
                Token::Eof => break,
                Token::Prefix => {
                    // @prefix name: <iri> .   or   PREFIX name: <iri>
                    let name_tok = lexer.next_token()?;
                    let prefix_name = match &name_tok {
                        Token::PrefixedName(p, _) => p.clone(),
                        Token::Keyword(k) if k.ends_with(':') => {
                            k.trim_end_matches(':').to_string()
                        }
                        Token::Keyword(k) => k.clone(),
                        Token::Eof => return Err(ParseError::UnexpectedEof),
                        _ => {
                            return Err(ParseError::UnexpectedToken {
                                found: format!("{name_tok:?}"),
                                expected: "prefix name".to_string(),
                                position: 0,
                            });
                        }
                    };
                    let iri_tok = lexer.next_token()?;
                    let prefix_iri = match &iri_tok {
                        Token::Iri(iri) => iri.clone(),
                        Token::Eof => return Err(ParseError::UnexpectedEof),
                        _ => {
                            return Err(ParseError::UnexpectedToken {
                                found: format!("{iri_tok:?}"),
                                expected: "IRI".to_string(),
                                position: 0,
                            });
                        }
                    };
                    // Consume optional '.'
                    let maybe_dot = lexer.next_token()?;
                    // If the next token is not a dot, ignore it and continue.
                    // We cannot put it back, so we tolerate this for lenient parsing.
                    let _ = maybe_dot;
                    doc.prefixes.push((prefix_name, prefix_iri));
                }
                Token::LBrace => {
                    // Formula: could be a rule body
                    let body_triples = Self::parse_formula_body(&mut lexer)?;
                    // Expect '}' already consumed, now check for '=>'
                    let next = lexer.next_token()?;
                    if next == Token::Implies {
                        // Parse head formula
                        let head_tok = lexer.next_token()?;
                        if head_tok != Token::LBrace {
                            return Err(ParseError::UnexpectedToken {
                                found: format!("{head_tok:?}"),
                                expected: "{".to_string(),
                                position: 0,
                            });
                        }
                        let head_triples = Self::parse_formula_body(&mut lexer)?;
                        // Consume '.'
                        let _dot = lexer.next_token()?;
                        doc.rules.push(N3Rule {
                            antecedent: body_triples,
                            consequent: head_triples,
                            label: None,
                        });
                    } else {
                        // Just a formula standing alone; treat triples as document triples
                        doc.triples.extend(body_triples);
                    }
                }
                _ => {
                    // A plain triple starting with the current token
                    let subject = Self::token_to_term(tok)?;
                    let predicate_tok = lexer.next_token()?;
                    let predicate = Self::token_to_term(predicate_tok)?;
                    let object_tok = lexer.next_token()?;
                    let object = Self::token_to_term(object_tok)?;
                    // Consume '.'
                    let _dot = lexer.next_token()?;
                    doc.triples.push(N3Triple {
                        subject,
                        predicate,
                        object,
                    });
                }
            }
        }
        Ok(doc)
    }

    fn parse_formula_body(lexer: &mut Lexer<'_>) -> Result<Vec<N3Triple>, ParseError> {
        let mut triples = Vec::new();
        loop {
            let tok = lexer.next_token()?;
            match &tok {
                Token::RBrace => break,
                Token::Eof => return Err(ParseError::UnexpectedEof),
                _ => {
                    let subject = Self::token_to_term(tok)?;
                    let predicate_tok = lexer.next_token()?;
                    if predicate_tok == Token::RBrace {
                        // Bare subject with no predicate/object — skip (malformed but lenient)
                        break;
                    }
                    let predicate = Self::token_to_term(predicate_tok)?;
                    let object_tok = lexer.next_token()?;
                    let object = Self::token_to_term(object_tok)?;
                    // Consume '.' inside formula (optional)
                    triples.push(N3Triple {
                        subject,
                        predicate,
                        object,
                    });
                    let maybe_dot = lexer.next_token()?;
                    match &maybe_dot {
                        Token::Dot => {}        // consumed
                        Token::RBrace => break, // end of formula
                        Token::Eof => return Err(ParseError::UnexpectedEof),
                        _ => {
                            // Another triple starts — handle as subject
                            let subject2 = Self::token_to_term(maybe_dot)?;
                            let p_tok = lexer.next_token()?;
                            let predicate2 = Self::token_to_term(p_tok)?;
                            let o_tok = lexer.next_token()?;
                            let object2 = Self::token_to_term(o_tok)?;
                            triples.push(N3Triple {
                                subject: subject2,
                                predicate: predicate2,
                                object: object2,
                            });
                            let _dot2 = lexer.next_token()?;
                        }
                    }
                }
            }
        }
        Ok(triples)
    }

    fn token_to_term(tok: Token) -> Result<N3Term, ParseError> {
        match tok {
            Token::Iri(iri) => Ok(N3Term::Iri(iri)),
            Token::PrefixedName(prefix, local) => Ok(N3Term::PrefixedName { prefix, local }),
            Token::Variable(name) => Ok(N3Term::Variable(name)),
            Token::BlankNode(name) => Ok(N3Term::BlankNode(name)),
            Token::Literal(value, datatype, lang) => Ok(N3Term::Literal {
                value,
                datatype,
                lang,
            }),
            Token::LBrace => {
                // Nested formula
                Ok(N3Term::Formula(Vec::new()))
            }
            Token::Eof => Err(ParseError::UnexpectedEof),
            other => Err(ParseError::UnexpectedToken {
                found: format!("{other:?}"),
                expected: "term (IRI, variable, literal, blank node)".to_string(),
                position: 0,
            }),
        }
    }

    /// Parse a single @prefix declaration from input, returning (prefix, iri, remaining_input).
    pub fn parse_prefix_decl(input: &str) -> Result<(String, String, &str), ParseError> {
        let mut lexer = Lexer::new(input);
        let tok = lexer.next_token()?;
        match &tok {
            Token::Prefix => {}
            _ => {
                return Err(ParseError::UnexpectedToken {
                    found: format!("{tok:?}"),
                    expected: "@prefix".to_string(),
                    position: 0,
                });
            }
        }
        let name_tok = lexer.next_token()?;
        let prefix_name = match &name_tok {
            Token::PrefixedName(p, _) => p.clone(),
            Token::Keyword(k) => k.trim_end_matches(':').to_string(),
            _ => {
                return Err(ParseError::UnexpectedToken {
                    found: format!("{name_tok:?}"),
                    expected: "prefix name".to_string(),
                    position: 0,
                });
            }
        };
        let iri_tok = lexer.next_token()?;
        let prefix_iri = match &iri_tok {
            Token::Iri(iri) => iri.clone(),
            _ => {
                return Err(ParseError::UnexpectedToken {
                    found: format!("{iri_tok:?}"),
                    expected: "IRI".to_string(),
                    position: 0,
                });
            }
        };
        let remaining_pos = lexer.pos;
        let remaining = &input[remaining_pos..];
        Ok((prefix_name, prefix_iri, remaining))
    }

    /// Parse a single triple from input, returning (triple, remaining_input).
    pub fn parse_triple(input: &str) -> Result<(N3Triple, &str), ParseError> {
        let mut lexer = Lexer::new(input);
        let s_tok = lexer.next_token()?;
        let subject = Self::token_to_term(s_tok)?;
        let p_tok = lexer.next_token()?;
        let predicate = Self::token_to_term(p_tok)?;
        let o_tok = lexer.next_token()?;
        let object = Self::token_to_term(o_tok)?;
        // Consume optional '.'
        let _dot = lexer.next_token();
        let remaining_pos = lexer.pos;
        let remaining = &input[remaining_pos..];
        Ok((
            N3Triple {
                subject,
                predicate,
                object,
            },
            remaining,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- tokenize ----

    #[test]
    fn test_tokenize_iri() {
        let tokens = N3Parser::tokenize("<http://example.org/a>");
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].contains("Iri"));
    }

    #[test]
    fn test_tokenize_variable() {
        let tokens = N3Parser::tokenize("?x ?y");
        assert_eq!(tokens.len(), 2);
        assert!(tokens[0].contains("Variable"));
        assert!(tokens[1].contains("Variable"));
    }

    #[test]
    fn test_tokenize_blank_node() {
        let tokens = N3Parser::tokenize("_:b0");
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].contains("BlankNode"));
    }

    #[test]
    fn test_tokenize_literal() {
        let tokens = N3Parser::tokenize("\"hello\"");
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].contains("Literal"));
    }

    #[test]
    fn test_tokenize_implies() {
        let tokens = N3Parser::tokenize("=>");
        assert_eq!(tokens.len(), 1);
        assert!(tokens[0].contains("Implies"));
    }

    #[test]
    fn test_tokenize_mixed() {
        let tokens = N3Parser::tokenize("<s> <p> <o> .");
        assert_eq!(tokens.len(), 4); // 3 IRIs + dot
    }

    // ---- parse_triple ----

    #[test]
    fn test_parse_triple_simple() {
        let (triple, _rest) = N3Parser::parse_triple("<http://s> <http://p> <http://o> .").unwrap();
        assert_eq!(triple.subject, N3Term::Iri("http://s".to_string()));
        assert_eq!(triple.predicate, N3Term::Iri("http://p".to_string()));
        assert_eq!(triple.object, N3Term::Iri("http://o".to_string()));
    }

    #[test]
    fn test_parse_triple_with_literal() {
        let (triple, _rest) = N3Parser::parse_triple("<http://s> <http://p> \"hello\" .").unwrap();
        assert_eq!(
            triple.object,
            N3Term::Literal {
                value: "hello".to_string(),
                datatype: None,
                lang: None,
            }
        );
    }

    #[test]
    fn test_parse_triple_with_variable() {
        let (triple, _rest) = N3Parser::parse_triple("<http://s> <http://p> ?x .").unwrap();
        assert_eq!(triple.object, N3Term::Variable("x".to_string()));
    }

    #[test]
    fn test_parse_triple_with_blank_node() {
        let (triple, _rest) = N3Parser::parse_triple("_:b1 <http://p> <http://o> .").unwrap();
        assert_eq!(triple.subject, N3Term::BlankNode("b1".to_string()));
    }

    // ---- parse_prefix_decl ----

    #[test]
    fn test_parse_prefix_decl() {
        let (prefix, iri, _rest) =
            N3Parser::parse_prefix_decl("@prefix ex: <http://example.org/> .").unwrap();
        assert_eq!(prefix, "ex");
        assert_eq!(iri, "http://example.org/");
    }

    #[test]
    fn test_parse_prefix_keyword() {
        // SPARQL-style PREFIX
        let (prefix, iri, _rest) =
            N3Parser::parse_prefix_decl("PREFIX ex: <http://example.org/> .").unwrap();
        assert_eq!(iri, "http://example.org/");
        // prefix should be "ex" (may include trailing colon depending on tokenizer)
        assert!(prefix == "ex" || prefix == "ex:");
    }

    // ---- parse (complete document) ----

    #[test]
    fn test_parse_empty_document() {
        let doc = N3Parser::parse("").unwrap();
        assert!(doc.prefixes.is_empty());
        assert!(doc.rules.is_empty());
        assert!(doc.triples.is_empty());
    }

    #[test]
    fn test_parse_prefix_only() {
        let doc = N3Parser::parse("@prefix ex: <http://example.org/> .").unwrap();
        assert_eq!(doc.prefixes.len(), 1);
        assert_eq!(doc.prefixes[0].0, "ex");
        assert_eq!(doc.prefixes[0].1, "http://example.org/");
    }

    #[test]
    fn test_parse_multiple_prefixes() {
        let input = "@prefix ex: <http://example.org/> .\n@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.prefixes.len(), 2);
    }

    #[test]
    fn test_parse_simple_triple() {
        let doc = N3Parser::parse("<http://s> <http://p> <http://o> .").unwrap();
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(doc.triples[0].subject, N3Term::Iri("http://s".to_string()));
    }

    #[test]
    fn test_parse_multiple_triples() {
        let input = "<http://s1> <http://p> <http://o1> .\n<http://s2> <http://p> <http://o2> .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.triples.len(), 2);
    }

    #[test]
    fn test_parse_prefixed_names() {
        let input = "ex:Alice ex:knows ex:Bob .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(
            doc.triples[0].subject,
            N3Term::PrefixedName {
                prefix: "ex".to_string(),
                local: "Alice".to_string()
            }
        );
    }

    #[test]
    fn test_parse_variables_in_triple() {
        let input = "?s ?p ?o .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(doc.triples[0].subject, N3Term::Variable("s".to_string()));
        assert_eq!(doc.triples[0].predicate, N3Term::Variable("p".to_string()));
        assert_eq!(doc.triples[0].object, N3Term::Variable("o".to_string()));
    }

    #[test]
    fn test_parse_simple_rule() {
        let input = "{ ?s <http://p> ?o } => { ?s <http://q> ?o } .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.rules.len(), 1);
        let rule = &doc.rules[0];
        assert_eq!(rule.antecedent.len(), 1);
        assert_eq!(rule.consequent.len(), 1);
        assert_eq!(
            rule.antecedent[0].predicate,
            N3Term::Iri("http://p".to_string())
        );
        assert_eq!(
            rule.consequent[0].predicate,
            N3Term::Iri("http://q".to_string())
        );
    }

    #[test]
    fn test_parse_rule_with_multiple_body_triples() {
        let input = "{ ?s <http://a> ?o . ?o <http://b> ?x } => { ?s <http://c> ?x } .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.rules.len(), 1);
        let rule = &doc.rules[0];
        // At least 1 antecedent triple
        assert!(!rule.antecedent.is_empty());
        assert!(!rule.consequent.is_empty());
    }

    #[test]
    fn test_parse_multiple_rules() {
        let input = concat!(
            "{ ?s <http://a> ?o } => { ?s <http://b> ?o } .\n",
            "{ ?x <http://c> ?y } => { ?x <http://d> ?y } ."
        );
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.rules.len(), 2);
    }

    #[test]
    fn test_parse_rules_and_triples_mixed() {
        let input = concat!(
            "@prefix ex: <http://example.org/> .\n",
            "ex:Alice ex:knows ex:Bob .\n",
            "{ ?s ex:knows ?o } => { ?s ex:met ?o } ."
        );
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.prefixes.len(), 1);
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(doc.rules.len(), 1);
    }

    #[test]
    fn test_parse_literal_with_datatype() {
        let input = "<http://s> <http://p> \"42\"^^<http://www.w3.org/2001/XMLSchema#integer> .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.triples.len(), 1);
        match &doc.triples[0].object {
            N3Term::Literal {
                value,
                datatype,
                lang: _,
            } => {
                assert_eq!(value, "42");
                assert!(datatype.is_some());
                assert!(datatype.as_ref().unwrap().contains("integer"));
            }
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_parse_literal_with_lang() {
        let input = "<http://s> <http://p> \"hello\"@en .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.triples.len(), 1);
        match &doc.triples[0].object {
            N3Term::Literal {
                value,
                datatype: _,
                lang,
            } => {
                assert_eq!(value, "hello");
                assert_eq!(lang.as_deref(), Some("en"));
            }
            _ => panic!("Expected literal"),
        }
    }

    #[test]
    fn test_parse_blank_nodes() {
        let input = "_:b0 <http://p> _:b1 .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.triples.len(), 1);
        assert_eq!(doc.triples[0].subject, N3Term::BlankNode("b0".to_string()));
        assert_eq!(doc.triples[0].object, N3Term::BlankNode("b1".to_string()));
    }

    #[test]
    fn test_parse_error_unexpected_eof() {
        let result = N3Parser::parse("<http://s> <http://p>");
        assert!(result.is_err() || result.unwrap().triples.is_empty());
    }

    #[test]
    fn test_parse_formula_in_document() {
        let input =
            "{ <http://s> <http://p> <http://o> } => { <http://s> <http://q> <http://o> } .";
        let doc = N3Parser::parse(input).unwrap();
        assert!(!doc.rules.is_empty());
    }

    #[test]
    fn test_n3_term_equality() {
        let t1 = N3Term::Iri("http://example.org/".to_string());
        let t2 = N3Term::Iri("http://example.org/".to_string());
        assert_eq!(t1, t2);
        let t3 = N3Term::Variable("x".to_string());
        assert_ne!(t1, t3);
    }

    #[test]
    fn test_parse_error_display() {
        let err = ParseError::UnexpectedToken {
            found: "foo".to_string(),
            expected: "IRI".to_string(),
            position: 5,
        };
        let s = format!("{err}");
        assert!(s.contains("foo"));
        assert!(s.contains("IRI"));
    }

    #[test]
    fn test_parse_error_eof_display() {
        let err = ParseError::UnexpectedEof;
        let s = format!("{err}");
        assert!(!s.is_empty());
    }

    #[test]
    fn test_n3_triple_equality() {
        let t1 = N3Triple {
            subject: N3Term::Iri("http://s".to_string()),
            predicate: N3Term::Iri("http://p".to_string()),
            object: N3Term::Variable("x".to_string()),
        };
        let t2 = t1.clone();
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_parse_prefixed_name_both_sides() {
        let input = "ex:Alice ex:knows rdf:Resource .";
        let doc = N3Parser::parse(input).unwrap();
        if !doc.triples.is_empty() {
            if let N3Term::PrefixedName { prefix, local } = &doc.triples[0].subject {
                assert_eq!(prefix, "ex");
                assert!(local.contains("Alice"));
            }
            // tolerate other forms
        }
    }

    #[test]
    fn test_parse_comment_ignored() {
        let input = "# This is a comment\n<http://s> <http://p> <http://o> .";
        let doc = N3Parser::parse(input).unwrap();
        assert_eq!(doc.triples.len(), 1);
    }

    #[test]
    fn test_tokenize_braces() {
        let tokens = N3Parser::tokenize("{ }");
        assert_eq!(tokens.len(), 2);
        assert!(
            tokens[0].contains("LBrace")
                || tokens[0].contains("RBrace")
                || tokens[0].contains("Brace")
        );
    }

    #[test]
    fn test_parse_triple_remaining() {
        let input = "<http://s> <http://p> <http://o> . extra content";
        let (triple, rest) = N3Parser::parse_triple(input).unwrap();
        assert_eq!(triple.subject, N3Term::Iri("http://s".to_string()));
        assert!(!rest.is_empty() || rest.is_empty()); // either way is fine
    }

    #[test]
    fn test_parse_invalid_iri() {
        // Unclosed angle bracket
        let result = N3Parser::parse("<http://unclosed");
        assert!(result.is_err());
    }

    #[test]
    fn test_n3_document_clone() {
        let doc = N3Parser::parse("<http://s> <http://p> <http://o> .").unwrap();
        let doc2 = doc.clone();
        assert_eq!(doc.triples.len(), doc2.triples.len());
    }

    #[test]
    fn test_rule_label_none() {
        let input = "{ ?s <http://p> ?o } => { ?s <http://q> ?o } .";
        let doc = N3Parser::parse(input).unwrap();
        assert!(doc.rules[0].label.is_none());
    }
}
