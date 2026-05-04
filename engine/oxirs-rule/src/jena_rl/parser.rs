// Jena Rule Language recursive-descent parser.
// Consumes a token stream produced by `super::lexer` and builds a `JrlRuleSet`.

use std::collections::HashMap;
use std::fmt;

use super::lexer::{SpannedToken, Token};

// ---------------------------------------------------------------------------
// IR types
// ---------------------------------------------------------------------------

/// A term in a JRL atom (variable, IRI, or literal).
#[derive(Debug, Clone, PartialEq)]
pub enum JrlTerm {
    Variable(String),
    Iri(String),
    Literal(JrlLiteral),
}

/// Literal value kinds.
#[derive(Debug, Clone, PartialEq)]
pub enum JrlLiteral {
    String(String),
    Integer(i64),
    Float(f64),
}

/// One atom in the body or head of a rule.
#[derive(Debug, Clone)]
pub enum JrlAtom {
    /// Triple pattern: (subject predicate object)
    Triple {
        subject: JrlTerm,
        predicate: JrlTerm,
        object: JrlTerm,
    },
    /// Built-in call: (builtinName arg1 arg2 ...)
    Builtin { name: String, args: Vec<JrlTerm> },
}

/// One parsed Jena rule.
#[derive(Debug, Clone)]
pub struct JrlRule {
    /// Optional rule name (synthetic if absent: `jrl_rule_N`)
    pub name: Option<String>,
    /// Body atoms (conditions)
    pub conditions: Vec<JrlAtom>,
    /// Head atoms (consequences)
    pub consequences: Vec<JrlAtom>,
    /// `true` if this is a backward-chaining rule (head before `<-`)
    pub is_backward: bool,
}

/// Complete parsed rule set with a prefix map.
#[derive(Debug, Clone)]
pub struct JrlRuleSet {
    pub prefixes: HashMap<String, String>,
    pub rules: Vec<JrlRule>,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct JrlParseError {
    pub message: String,
    pub token_index: usize,
}

impl fmt::Display for JrlParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JRL parse error at token {}: {}",
            self.token_index, self.message
        )
    }
}

impl std::error::Error for JrlParseError {}

// ---------------------------------------------------------------------------
// Default prefix map (well-known namespaces Jena pre-populates)
// ---------------------------------------------------------------------------

fn default_prefixes() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(
        "rdf".to_string(),
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
    );
    m.insert(
        "rdfs".to_string(),
        "http://www.w3.org/2000/01/rdf-schema#".to_string(),
    );
    m.insert(
        "xsd".to_string(),
        "http://www.w3.org/2001/XMLSchema#".to_string(),
    );
    m.insert(
        "owl".to_string(),
        "http://www.w3.org/2002/07/owl#".to_string(),
    );
    m
}

// ---------------------------------------------------------------------------
// Parser state
// ---------------------------------------------------------------------------

struct Parser<'a> {
    tokens: &'a [SpannedToken],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [SpannedToken]) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens
            .get(self.pos)
            .map_or(&Token::Eof, |st| &st.token)
    }

    fn advance(&mut self) -> Token {
        let tok = self
            .tokens
            .get(self.pos)
            .map_or(Token::Eof, |st| st.token.clone());
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn error(&self, msg: impl Into<String>) -> JrlParseError {
        JrlParseError {
            message: msg.into(),
            token_index: self.pos,
        }
    }

    fn expect_colon(&mut self) -> Result<(), JrlParseError> {
        if *self.peek() == Token::Colon {
            self.advance();
            Ok(())
        } else {
            Err(self.error(format!("expected `:`, found `{}`", self.peek())))
        }
    }

    // ------------------------------------------------------------------
    // Term parsing
    // ------------------------------------------------------------------

    fn parse_term(&mut self) -> Result<JrlTerm, JrlParseError> {
        match self.peek().clone() {
            Token::Variable(name) => {
                self.advance();
                Ok(JrlTerm::Variable(name))
            }
            Token::Iri(iri) => {
                self.advance();
                Ok(JrlTerm::Iri(iri))
            }
            Token::PrefixedName(prefix, local) => {
                self.advance();
                // Resolve at lowering time; keep as Iri for now using a placeholder sentinel
                Ok(JrlTerm::Iri(format!("prefix:{}:{}", prefix, local)))
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(JrlTerm::Literal(JrlLiteral::String(s)))
            }
            Token::IntLit(n) => {
                self.advance();
                Ok(JrlTerm::Literal(JrlLiteral::Integer(n)))
            }
            Token::FloatLit(f) => {
                self.advance();
                Ok(JrlTerm::Literal(JrlLiteral::Float(f)))
            }
            Token::Ident(name) => {
                self.advance();
                // Bare identifier — treat as an IRI constant (no prefix)
                Ok(JrlTerm::Iri(name))
            }
            other => Err(self.error(format!("expected term, found `{}`", other))),
        }
    }

    // ------------------------------------------------------------------
    // Atom parsing: `( ... )`
    // ------------------------------------------------------------------

    fn parse_atom(&mut self) -> Result<JrlAtom, JrlParseError> {
        if *self.peek() != Token::LParen {
            return Err(self.error(format!("expected `(`, found `{}`", self.peek())));
        }
        self.advance(); // consume `(`

        // First token inside the atom is the head (predicate for builtins, or subject for triples)
        let first = self.parse_term()?;

        // Collect remaining terms until `)`
        let mut rest: Vec<JrlTerm> = Vec::new();
        while *self.peek() != Token::RParen && *self.peek() != Token::Eof {
            rest.push(self.parse_term()?);
        }

        if *self.peek() == Token::RParen {
            self.advance(); // consume `)`
        } else {
            return Err(self.error("unterminated atom: expected `)`"));
        }

        // Classify as triple or builtin
        // A triple has exactly two more terms (subject pred obj → 3 total)
        // A builtin can have any arity.
        //
        // Heuristic: if first term is an IRI/constant that looks like a builtin name
        // (no slashes, no colons in the expanded form, or a well-known name), treat as builtin.
        // Otherwise treat as triple if we have exactly 2 more terms.
        match &first {
            JrlTerm::Iri(name) if !name.contains('/') && !name.starts_with("prefix:") => {
                // Could be a builtin name
                let known_builtins = [
                    "equal",
                    "notEqual",
                    "lessThan",
                    "greaterThan",
                    "lessThanOrEqual",
                    "greaterThanOrEqual",
                    "sum",
                    "difference",
                    "product",
                    "quotient",
                    "modulo",
                    "min",
                    "max",
                    "abs",
                    "strConcat",
                    "strLen",
                    "strSubstring",
                    "strContains",
                    "strStartsWith",
                    "strEndsWith",
                    "strLang",
                    "strLangMatches",
                    "str",
                    "print",
                    "isBNode",
                    "isLiteral",
                    "isURI",
                    "bound",
                    "now",
                    "regex",
                    "datatypeURI",
                    "makeTemp",
                    "noValue",
                    "listContains",
                    "listLength",
                    "listAppend",
                    "listEntry",
                    "listsEqual",
                    "listNotContains",
                    "listMapWith",
                    "drop",
                    "addOne",
                    "skolem",
                ];
                if known_builtins.contains(&name.as_str()) {
                    let mut args = vec![first];
                    // The first token was the builtin name, args follow
                    // Actually for builtins the name is the first token; args are rest.
                    // Re-structure: name = first Iri value, args = rest
                    let builtin_name = match &args[0] {
                        JrlTerm::Iri(n) => n.clone(),
                        _ => unreachable!(),
                    };
                    args.remove(0);
                    args.extend(rest);
                    return Ok(JrlAtom::Builtin {
                        name: builtin_name,
                        args,
                    });
                }
                // Not a known builtin; treat as triple if 2 more
                if rest.len() == 2 {
                    Ok(JrlAtom::Triple {
                        subject: first,
                        predicate: rest.remove(0),
                        object: rest.remove(0),
                    })
                } else {
                    // Unknown builtin with unknown arity
                    let name_str = match &first {
                        JrlTerm::Iri(n) => n.clone(),
                        _ => String::new(),
                    };
                    let mut all = vec![first];
                    all.extend(rest);
                    // Remove name from args
                    let bname = name_str;
                    all.remove(0);
                    Ok(JrlAtom::Builtin {
                        name: bname,
                        args: all,
                    })
                }
            }
            _ => {
                // Definitely a triple-like thing (variable/prefixed name/IRI with slashes as subject)
                if rest.len() == 2 {
                    Ok(JrlAtom::Triple {
                        subject: first,
                        predicate: rest.remove(0),
                        object: rest.remove(0),
                    })
                } else if !rest.is_empty() {
                    // Best-effort: treat as triple with first 3
                    Err(self.error(format!(
                        "atom has {} terms (expected 3 for a triple)",
                        rest.len() + 1
                    )))
                } else {
                    Err(self.error("atom has fewer than 3 terms"))
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Atom list: zero or more `(...)` atoms
    // ------------------------------------------------------------------

    fn parse_atom_list(&mut self) -> Result<Vec<JrlAtom>, JrlParseError> {
        let mut atoms = Vec::new();
        while *self.peek() == Token::LParen {
            atoms.push(self.parse_atom()?);
        }
        Ok(atoms)
    }

    // ------------------------------------------------------------------
    // Rule: `[ optName: body -> head ]`  or  `[ optName: head <- body ]`
    // ------------------------------------------------------------------

    fn parse_rule(&mut self, rule_idx: usize) -> Result<JrlRule, JrlParseError> {
        // `[` already consumed by caller
        // Optional name followed by `:`
        let name = match self.peek().clone() {
            Token::Ident(n) => {
                // Peek ahead: if next after ident is `:`, this is the rule name
                let saved_pos = self.pos;
                self.advance(); // consume ident
                if *self.peek() == Token::Colon {
                    self.advance(); // consume `:`
                    Some(n)
                } else {
                    // Not a name — restore
                    self.pos = saved_pos;
                    None
                }
            }
            _ => None,
        };

        // Parse first atom list
        let first_atoms = self.parse_atom_list()?;

        // Determine direction from arrow token
        let arrow = self.advance();
        let (conditions, consequences, is_backward) = match arrow {
            Token::Arrow => {
                // Forward: body -> head
                let head_atoms = self.parse_atom_list()?;
                (first_atoms, head_atoms, false)
            }
            Token::BackArrow => {
                // Backward: head <- body
                let body_atoms = self.parse_atom_list()?;
                (body_atoms, first_atoms, true)
            }
            other => {
                return Err(self.error(format!("expected `->` or `<-` in rule, found `{}`", other)));
            }
        };

        // Expect `]`
        if *self.peek() == Token::RBracket {
            self.advance();
        } else {
            return Err(self.error(format!(
                "expected `]` to close rule, found `{}`",
                self.peek()
            )));
        }

        let rule_name = name.or_else(|| Some(format!("jrl_rule_{}", rule_idx)));
        Ok(JrlRule {
            name: rule_name,
            conditions,
            consequences,
            is_backward,
        })
    }

    // ------------------------------------------------------------------
    // @prefix declaration: `@prefix prefix: <IRI> .`
    // ------------------------------------------------------------------

    fn parse_prefix_decl(
        &mut self,
        prefixes: &mut HashMap<String, String>,
    ) -> Result<(), JrlParseError> {
        // `@prefix` already consumed by caller

        // Accept bare ident followed by `:` as the prefix name, or just PrefixedName with empty local.
        let prefix_name = match self.peek().clone() {
            Token::Ident(n) => {
                self.advance();
                self.expect_colon()?;
                n
            }
            Token::PrefixedName(p, local) if local.is_empty() => {
                self.advance();
                p
            }
            Token::Colon => {
                // default namespace prefix (empty prefix)
                self.advance();
                String::new()
            }
            other => {
                return Err(self.error(format!(
                    "expected prefix name after @prefix, found `{}`",
                    other
                )));
            }
        };

        // Expect full IRI
        let iri = match self.peek().clone() {
            Token::Iri(iri) => {
                self.advance();
                iri
            }
            other => {
                return Err(
                    self.error(format!("expected IRI after prefix name, found `{}`", other))
                );
            }
        };

        // Optional trailing `.`
        if *self.peek() == Token::Dot {
            self.advance();
        }

        prefixes.insert(prefix_name, iri);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Top-level parse
    // ------------------------------------------------------------------

    fn parse_rule_set(&mut self) -> Result<JrlRuleSet, JrlParseError> {
        let mut prefixes = default_prefixes();
        let mut rules = Vec::new();
        let mut rule_idx = 0;

        loop {
            match self.peek().clone() {
                Token::Eof => break,
                Token::AtPrefix => {
                    self.advance();
                    self.parse_prefix_decl(&mut prefixes)?;
                }
                Token::LBracket => {
                    self.advance();
                    let rule = self.parse_rule(rule_idx)?;
                    rule_idx += 1;
                    rules.push(rule);
                }
                other => {
                    // Skip unknown top-level tokens (e.g. stray `.`)
                    return Err(self.error(format!("unexpected top-level token: `{}`", other)));
                }
            }
        }

        Ok(JrlRuleSet { prefixes, rules })
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse a token stream into a `JrlRuleSet`.
pub fn parse(tokens: &[SpannedToken]) -> Result<JrlRuleSet, JrlParseError> {
    let mut parser = Parser::new(tokens);
    parser.parse_rule_set()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jena_rl::lexer::Lexer;

    fn do_parse(input: &str) -> JrlRuleSet {
        let toks = Lexer::tokenize(input).expect("lex should succeed");
        parse(&toks).expect("parse should succeed")
    }

    #[test]
    fn test_parse_simple_rule() {
        let rs = do_parse("[(?a rdf:type ex:Person) -> (?a rdf:type ex:Human)]");
        assert_eq!(rs.rules.len(), 1);
        let rule = &rs.rules[0];
        assert_eq!(rule.conditions.len(), 1);
        assert_eq!(rule.consequences.len(), 1);
        assert!(!rule.is_backward);
    }

    #[test]
    fn test_parse_rule_with_name() {
        let rs = do_parse("[parentRule: (?x ex:parent ?y) -> (?y ex:child ?x)]");
        assert_eq!(rs.rules[0].name, Some("parentRule".to_string()));
    }

    #[test]
    fn test_parse_multiple_conditions() {
        let rs = do_parse("[chain: (?a ex:p ?b) (?b ex:q ?c) -> (?a ex:r ?c)]");
        assert_eq!(rs.rules[0].conditions.len(), 2);
        assert_eq!(rs.rules[0].consequences.len(), 1);
    }

    #[test]
    fn test_parse_prefix_declaration() {
        let rs = do_parse("@prefix ex: <http://example.org/> .\n[r: (?x ex:p ?y) -> (?x ex:q ?y)]");
        assert_eq!(
            rs.prefixes.get("ex"),
            Some(&"http://example.org/".to_string())
        );
    }

    #[test]
    fn test_parse_builtin_equal() {
        let rs = do_parse("[r: (?x rdf:value ?v) (equal ?v 42) -> (?x rdf:type ex:Positive)]");
        let body = &rs.rules[0].conditions;
        let has_equal = body
            .iter()
            .any(|a| matches!(a, JrlAtom::Builtin { name, .. } if name == "equal"));
        assert!(has_equal, "should find `equal` builtin in conditions");
    }

    #[test]
    fn test_parse_string_literal_in_triple() {
        let rs = do_parse(r#"[r: (?x ex:name "Alice") -> (?x rdf:type ex:Person)]"#);
        let cond = &rs.rules[0].conditions[0];
        match cond {
            JrlAtom::Triple { object, .. } => {
                assert_eq!(
                    *object,
                    JrlTerm::Literal(JrlLiteral::String("Alice".to_string()))
                );
            }
            _ => panic!("expected Triple"),
        }
    }

    #[test]
    fn test_parse_backward_rule() {
        let rs = do_parse("[r: (?x ex:ancestor ?z) <- (?x ex:parent ?z)]");
        assert!(rs.rules[0].is_backward);
        assert_eq!(rs.rules[0].consequences.len(), 1);
        assert_eq!(rs.rules[0].conditions.len(), 1);
    }

    #[test]
    fn test_parse_unnamed_rule_gets_synthetic_name() {
        let rs = do_parse("[(?x ex:p ?y) -> (?x ex:q ?y)]");
        assert!(rs.rules[0].name.is_some());
        let name = rs.rules[0].name.as_deref().unwrap();
        assert!(
            name.starts_with("jrl_rule_"),
            "name should be synthetic: {}",
            name
        );
    }

    #[test]
    fn test_parse_multiple_rules() {
        let input = "[r1: (?a ex:p ?b) -> (?a ex:q ?b)]\n[r2: (?x ex:q ?y) -> (?x ex:r ?y)]";
        let rs = do_parse(input);
        assert_eq!(rs.rules.len(), 2);
    }
}
