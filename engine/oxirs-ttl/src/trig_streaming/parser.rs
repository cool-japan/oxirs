//! TriG recursive-descent parser.
//!
//! Parses statements from the token stream produced by [`TriGLexer`] and
//! emits [`StreamedQuad`] values.

use std::collections::HashMap;
use std::io::BufRead;

use crate::trig_streaming::{
    lexer::{TriGLexer, TriGToken},
    StreamedQuad, TriGLiteral, TriGParseError, TriGTerm,
};

/// The W3C RDF type IRI.
const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
/// XSD string datatype IRI.
#[allow(dead_code)]
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
/// XSD boolean IRI.
const XSD_BOOLEAN: &str = "http://www.w3.org/2001/XMLSchema#boolean";
/// XSD integer IRI.
const XSD_INTEGER: &str = "http://www.w3.org/2001/XMLSchema#integer";
/// XSD decimal IRI.
const XSD_DECIMAL: &str = "http://www.w3.org/2001/XMLSchema#decimal";
/// XSD double IRI.
const XSD_DOUBLE: &str = "http://www.w3.org/2001/XMLSchema#double";

// ============================================================================
// TriGParser
// ============================================================================

/// A recursive-descent parser over a [`TriGLexer`] token stream.
pub struct TriGParser<R: BufRead> {
    lexer: TriGLexer<R>,
    /// Active prefix mappings.
    prefix_map: HashMap<String, String>,
    /// Active base IRI.
    base: Option<String>,
    /// Counter for generating unique blank node labels.
    bnode_counter: usize,
    /// Named blank node label mapping (label → generated ID).
    blank_node_map: HashMap<String, usize>,
    /// Current graph name; `None` = default graph.
    current_graph: Option<TriGTerm>,
    /// Line at which the current graph block was opened.
    graph_opened_at: Option<usize>,
}

impl<R: BufRead> TriGParser<R> {
    /// Create a new parser wrapping the given reader.
    pub fn new(reader: R) -> Self {
        Self {
            lexer: TriGLexer::new(reader),
            prefix_map: HashMap::new(),
            base: None,
            bnode_counter: 0,
            blank_node_map: HashMap::new(),
            current_graph: None,
            graph_opened_at: None,
        }
    }

    // -----------------------------------------------------------------------
    // Top-level: parse_statement
    // -----------------------------------------------------------------------

    /// Parse one top-level statement.
    ///
    /// Returns:
    /// - `Ok(None)` — EOF reached.
    /// - `Ok(Some(vec))` — zero or more quads (directives produce 0 quads).
    /// - `Err(_)` — parse error.
    pub fn parse_statement(&mut self) -> Result<Option<Vec<StreamedQuad>>, TriGParseError> {
        let tok = match self.lexer.peek()? {
            None => return Ok(None),
            Some(tok) => tok.clone(),
        };

        match tok {
            // @prefix directive — the label is encoded inside the token.
            TriGToken::Prefix(_) => self.handle_prefix_directive(),

            // @base directive — IRI is already inside the token.
            TriGToken::Base(iri) => {
                let resolved = self.resolve_iri(&iri);
                self.base = Some(resolved);
                self.lexer.next_token()?;
                self.expect_dot()?;
                Ok(Some(vec![]))
            }

            // Closing brace — end of graph block.
            TriGToken::RBrace => {
                self.lexer.next_token()?;
                if self.current_graph.is_none() && self.graph_opened_at.is_none() {
                    return Err(TriGParseError::InvalidGraph {
                        line: self.lexer.line(),
                        name: "Unexpected '}' outside any graph block".to_string(),
                    });
                }
                self.current_graph = None;
                self.graph_opened_at = None;
                Ok(Some(vec![]))
            }

            // Dot — skip stray statement terminators.
            TriGToken::Dot => {
                self.lexer.next_token()?;
                Ok(Some(vec![]))
            }

            // Could be a graph name IRI followed by `{`, or a triple subject.
            TriGToken::IriRef(_) | TriGToken::PrefixedName(_, _) => {
                self.parse_iri_or_graph_or_triple()
            }

            // Blank node subject or property list.
            TriGToken::BlankNodeLabel(_) | TriGToken::AnonBlankNode | TriGToken::LBracket => {
                let quads = self.parse_triples()?;
                Ok(Some(quads))
            }

            // Opening brace — anonymous graph block.
            TriGToken::LBrace => {
                self.lexer.next_token()?;
                let opened = self.lexer.line();
                self.current_graph = None; // anonymous graph (= default graph context)
                self.graph_opened_at = Some(opened);
                Ok(Some(vec![]))
            }

            _ => {
                let line = self.lexer.line();
                Err(TriGParseError::InvalidToken {
                    line,
                    message: format!("Unexpected token at start of statement: {:?}", tok),
                })
            }
        }
    }

    // -----------------------------------------------------------------------
    // Directive handling
    // -----------------------------------------------------------------------

    fn handle_prefix_directive(&mut self) -> Result<Option<Vec<StreamedQuad>>, TriGParseError> {
        // The Prefix token has the label embedded inside it.
        let label = match self.lexer.next_token()? {
            Some(TriGToken::Prefix(label)) => label,
            other => {
                return Err(TriGParseError::InvalidToken {
                    line: self.lexer.line(),
                    message: format!("Expected @prefix token, got {:?}", other),
                });
            }
        };

        // Next token must be the IRI reference.
        let iri = match self.lexer.next_token()? {
            Some(TriGToken::IriRef(iri)) => self.resolve_iri(&iri),
            other => {
                return Err(TriGParseError::InvalidToken {
                    line: self.lexer.line(),
                    message: format!("Expected IRI for @prefix, got {:?}", other),
                });
            }
        };

        self.prefix_map.insert(label, iri);
        self.expect_dot()?;
        Ok(Some(vec![]))
    }

    // -----------------------------------------------------------------------
    // Graph block or triple disambiguation
    // -----------------------------------------------------------------------

    fn parse_iri_or_graph_or_triple(
        &mut self,
    ) -> Result<Option<Vec<StreamedQuad>>, TriGParseError> {
        // Consume the IRI/prefixed-name token.
        let tok = self.lexer.next_token()?.expect("peeked above");
        let term = self.token_to_term(tok)?;

        // Peek ahead: if next token is `{`, this is a named graph block.
        match self.lexer.peek()? {
            Some(TriGToken::LBrace) => {
                self.lexer.next_token()?; // consume '{'
                let opened = self.lexer.line();
                self.current_graph = Some(term);
                self.graph_opened_at = Some(opened);
                Ok(Some(vec![]))
            }
            _ => {
                // Not a graph block — this is a triple with the IRI as subject.
                let quads = self.parse_predicate_object_list(term)?;
                self.expect_dot()?;
                Ok(Some(quads))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Triple parsing
    // -----------------------------------------------------------------------

    /// Parse a complete triple statement (subject + predicate-object list + `.`).
    pub fn parse_triples(&mut self) -> Result<Vec<StreamedQuad>, TriGParseError> {
        let subject = self.parse_term()?;
        let quads = self.parse_predicate_object_list(subject)?;
        self.expect_dot()?;
        Ok(quads)
    }

    /// Parse a predicate-object list starting with the given subject.
    ///
    /// Handles:
    /// - Multiple predicate-object pairs separated by `;`
    /// - Multiple objects separated by `,`
    fn parse_predicate_object_list(
        &mut self,
        subject: TriGTerm,
    ) -> Result<Vec<StreamedQuad>, TriGParseError> {
        let mut quads: Vec<StreamedQuad> = Vec::new();

        'outer: loop {
            // Check for end of predicate-object list.
            match self.lexer.peek()? {
                Some(TriGToken::Dot) | Some(TriGToken::RBrace) | None => break,
                Some(TriGToken::Semicolon) => {
                    self.lexer.next_token()?;
                    // `;` can be followed by more predicates or by `.`/`}`.
                    match self.lexer.peek()? {
                        Some(TriGToken::Dot) | Some(TriGToken::RBrace) | None => break,
                        Some(TriGToken::Semicolon) => continue 'outer, // double ;;
                        _ => {} // more predicates follow
                    }
                }
                _ => {}
            }

            // Parse predicate.
            let predicate = self.parse_predicate()?;

            // Parse object list (comma-separated).
            loop {
                let object = self.parse_object()?;
                quads.push(StreamedQuad {
                    subject: subject.clone(),
                    predicate: predicate.clone(),
                    object,
                    graph_name: self.current_graph.clone(),
                });

                // Check for comma (another object).
                match self.lexer.peek()? {
                    Some(TriGToken::Comma) => {
                        self.lexer.next_token()?;
                    }
                    _ => break,
                }
            }
        }

        Ok(quads)
    }

    // -----------------------------------------------------------------------
    // Term parsing
    // -----------------------------------------------------------------------

    /// Parse any RDF term (subject, predicate, or object).
    pub fn parse_term(&mut self) -> Result<TriGTerm, TriGParseError> {
        let tok = match self.lexer.next_token()? {
            Some(t) => t,
            None => {
                return Err(TriGParseError::InvalidTriple {
                    line: self.lexer.line(),
                    message: "Expected term, got EOF".to_string(),
                });
            }
        };
        self.token_to_term(tok)
    }

    /// Parse a predicate (must be a named node or `a`).
    fn parse_predicate(&mut self) -> Result<TriGTerm, TriGParseError> {
        let tok = match self.lexer.next_token()? {
            Some(t) => t,
            None => {
                return Err(TriGParseError::InvalidTriple {
                    line: self.lexer.line(),
                    message: "Expected predicate, got EOF".to_string(),
                });
            }
        };
        match tok {
            TriGToken::A => Ok(TriGTerm::NamedNode(RDF_TYPE.to_string())),
            TriGToken::IriRef(iri) => Ok(TriGTerm::NamedNode(self.resolve_iri(&iri))),
            TriGToken::PrefixedName(p, l) => {
                let iri = self.expand_prefixed_name(&p, &l)?;
                Ok(TriGTerm::NamedNode(iri))
            }
            other => Err(TriGParseError::InvalidTriple {
                line: self.lexer.line(),
                message: format!("Expected predicate (IRI or 'a'), got {:?}", other),
            }),
        }
    }

    /// Parse an object term.
    fn parse_object(&mut self) -> Result<TriGTerm, TriGParseError> {
        match self.lexer.peek()? {
            Some(TriGToken::LBracket) => {
                // Blank node property list: [ predicate object ; ... ]
                self.lexer.next_token()?; // consume '['
                let bnode = self.new_blank_node();
                // Parse inner property list.
                let _inner_quads = self.parse_predicate_object_list(bnode.clone())?;
                // Consume the closing ']'.
                match self.lexer.next_token()? {
                    Some(TriGToken::RBracket) => {}
                    other => {
                        return Err(TriGParseError::InvalidTriple {
                            line: self.lexer.line(),
                            message: format!("Expected ']', got {:?}", other),
                        });
                    }
                }
                Ok(bnode)
            }
            _ => self.parse_term(),
        }
    }

    /// Convert a [`TriGToken`] to a [`TriGTerm`].
    fn token_to_term(&mut self, tok: TriGToken) -> Result<TriGTerm, TriGParseError> {
        match tok {
            TriGToken::IriRef(iri) => Ok(TriGTerm::NamedNode(self.resolve_iri(&iri))),
            TriGToken::PrefixedName(prefix, local) => {
                let iri = self.expand_prefixed_name(&prefix, &local)?;
                Ok(TriGTerm::NamedNode(iri))
            }
            TriGToken::BlankNodeLabel(label) => {
                let id = self.get_or_create_bnode(&label);
                Ok(TriGTerm::BlankNode(format!("b{}", id)))
            }
            TriGToken::AnonBlankNode => Ok(self.new_blank_node()),
            TriGToken::A => Ok(TriGTerm::NamedNode(RDF_TYPE.to_string())),
            TriGToken::StringLiteral { value, lang, datatype } => {
                let resolved_dt = datatype.map(|dt| self.resolve_datatype(&dt));
                Ok(TriGTerm::Literal(TriGLiteral {
                    value,
                    datatype: resolved_dt,
                    language: lang,
                }))
            }
            TriGToken::Integer(i) => Ok(TriGTerm::Literal(TriGLiteral {
                value: i.to_string(),
                datatype: Some(XSD_INTEGER.to_string()),
                language: None,
            })),
            TriGToken::Decimal(f) => Ok(TriGTerm::Literal(TriGLiteral {
                value: format!("{}", f),
                datatype: Some(XSD_DECIMAL.to_string()),
                language: None,
            })),
            TriGToken::Double(f) => Ok(TriGTerm::Literal(TriGLiteral {
                value: format!("{:E}", f),
                datatype: Some(XSD_DOUBLE.to_string()),
                language: None,
            })),
            TriGToken::True => Ok(TriGTerm::Literal(TriGLiteral {
                value: "true".to_string(),
                datatype: Some(XSD_BOOLEAN.to_string()),
                language: None,
            })),
            TriGToken::False => Ok(TriGTerm::Literal(TriGLiteral {
                value: "false".to_string(),
                datatype: Some(XSD_BOOLEAN.to_string()),
                language: None,
            })),
            other => Err(TriGParseError::InvalidTriple {
                line: self.lexer.line(),
                message: format!("Cannot use {:?} as a term", other),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // IRI resolution and prefix expansion
    // -----------------------------------------------------------------------

    /// Resolve an IRI reference against the active base IRI.
    pub fn resolve_iri(&self, iri: &str) -> String {
        // Absolute IRI — return as-is.
        if iri.contains("://") || iri.starts_with("urn:") {
            return iri.to_string();
        }
        if iri.is_empty() {
            return self.base.clone().unwrap_or_default();
        }
        // Relative reference.
        if let Some(base) = &self.base {
            // Fragment and absolute path refs.
            if iri.starts_with('#') || iri.starts_with('/') {
                return format!("{}{}", base, iri);
            }
            // Strip fragment and query from base, then append relative path.
            let base_no_frag = base.split('#').next().unwrap_or(base);
            let base_path = if base_no_frag.contains('/') {
                let last_slash = base_no_frag.rfind('/').unwrap_or(base_no_frag.len());
                &base_no_frag[..=last_slash]
            } else {
                base_no_frag
            };
            return format!("{}{}", base_path, iri);
        }
        iri.to_string()
    }

    /// Expand a CURIE (`prefix:local`) to an absolute IRI.
    pub fn expand_prefixed_name(
        &self,
        prefix: &str,
        local: &str,
    ) -> Result<String, TriGParseError> {
        match self.prefix_map.get(prefix) {
            Some(iri_prefix) => Ok(format!("{}{}", iri_prefix, local)),
            None => Err(TriGParseError::InvalidToken {
                line: self.lexer.line(),
                message: format!("Unknown prefix: {:?}", prefix),
            }),
        }
    }

    /// Resolve a datatype token (CURIE or IRI) to an absolute IRI.
    fn resolve_datatype(&self, dt: &str) -> String {
        // If it looks like a CURIE `prefix:local`, expand it.
        if let Some(colon_pos) = dt.find(':') {
            let prefix = &dt[..colon_pos];
            let local = &dt[colon_pos + 1..];
            if let Ok(expanded) = self.expand_prefixed_name(prefix, local) {
                return expanded;
            }
        }
        // Already absolute.
        if dt.contains("://") || dt.starts_with("urn:") {
            return dt.to_string();
        }
        // Fall back: treat as relative to base.
        self.resolve_iri(dt)
    }

    // -----------------------------------------------------------------------
    // Blank node helpers
    // -----------------------------------------------------------------------

    fn new_blank_node(&mut self) -> TriGTerm {
        let id = self.bnode_counter;
        self.bnode_counter += 1;
        TriGTerm::BlankNode(format!("b{}", id))
    }

    fn get_or_create_bnode(&mut self, label: &str) -> usize {
        if let Some(&id) = self.blank_node_map.get(label) {
            return id;
        }
        let id = self.bnode_counter;
        self.bnode_counter += 1;
        self.blank_node_map.insert(label.to_string(), id);
        id
    }

    // -----------------------------------------------------------------------
    // Structural helpers
    // -----------------------------------------------------------------------

    /// Expect and consume a `.` token.
    fn expect_dot(&mut self) -> Result<(), TriGParseError> {
        match self.lexer.next_token()? {
            Some(TriGToken::Dot) => Ok(()),
            other => Err(TriGParseError::InvalidTriple {
                line: self.lexer.line(),
                message: format!("Expected '.', got {:?}", other),
            }),
        }
    }
}
