//! Advanced N3 parser implementation using the N3 lexer
//!
//! This parser handles full N3 syntax including:
//! - Variables (?var)
//! - Formulas ({ ... })
//! - Implications (=> and <=)
//! - Quantifiers (@forAll, @forSome)

use crate::error::{TurtleParseError, TurtleResult, TurtleSyntaxError};
use crate::formats::n3_types::{
    N3BuiltinRegistry, N3Formula, N3Implication, N3Statement, N3Term, N3Variable,
};
use crate::lexer::n3::{N3Lexer, N3Token};
use oxirs_core::model::{BlankNode, Literal, NamedNode};
use std::collections::{HashMap, HashSet};

/// Advanced N3 parser with full formula and variable support
pub struct AdvancedN3Parser {
    lexer: N3Lexer,
    current_token: N3Token,
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
    /// Base IRI
    pub base_iri: Option<String>,
    /// Universal variables in scope
    universals: Vec<N3Variable>,
    /// Existential variables in scope
    existentials: Vec<N3Variable>,
    /// Built-in predicate registry
    #[allow(dead_code)]
    builtins: N3BuiltinRegistry,
    /// Whether to continue parsing after errors
    pub lenient: bool,
    /// Extra statements produced while parsing a blank-node property list
    /// (`[ ... ]`) or an RDF collection (`( ... )`) that was encountered as a
    /// subject/predicate/object term. These are flushed into the enclosing
    /// document or formula right after the statement that triggered them,
    /// mirroring `TurtleParsingContext::pending_triples`.
    pending_statements: Vec<N3Statement>,
    /// Counter used to mint fresh auto-generated blank node identifiers.
    blank_node_counter: usize,
    /// Blank node labels explicitly written in the input (e.g. `_:foo`), so that
    /// auto-generated identifiers never collide with them.
    explicit_blank_labels: HashSet<String>,
}

impl AdvancedN3Parser {
    /// Create a new advanced N3 parser
    pub fn new(input: &str) -> TurtleResult<Self> {
        let mut lexer = N3Lexer::new(input);
        let current_token = lexer.next_token()?;

        let mut prefixes = HashMap::new();
        // Add standard prefixes
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

        Ok(Self {
            lexer,
            current_token,
            prefixes,
            base_iri: None,
            universals: Vec::new(),
            existentials: Vec::new(),
            builtins: N3BuiltinRegistry::new(),
            lenient: false,
            pending_statements: Vec::new(),
            blank_node_counter: 0,
            explicit_blank_labels: HashSet::new(),
        })
    }

    /// Generate a fresh blank node identifier guaranteed not to collide with any
    /// explicitly-written label seen so far in the document (see
    /// `explicit_blank_labels`).
    fn generate_blank_node_id(&mut self) -> String {
        loop {
            let candidate = format!("genid-b{}", self.blank_node_counter);
            self.blank_node_counter += 1;
            if !self.explicit_blank_labels.contains(&candidate) {
                return candidate;
            }
        }
    }

    /// Parse the entire N3 document
    pub fn parse_document(&mut self) -> TurtleResult<N3Document> {
        let mut statements = Vec::new();
        let mut implications = Vec::new();

        while !matches!(self.current_token, N3Token::Eof) {
            // Handle prefix declarations
            if matches!(self.current_token, N3Token::PrefixDecl) {
                self.parse_prefix_declaration()?;
                continue;
            }

            // Handle base declarations
            if matches!(self.current_token, N3Token::BaseDecl) {
                self.parse_base_declaration()?;
                continue;
            }

            // Handle quantifiers
            if matches!(self.current_token, N3Token::ForAll) {
                self.parse_forall_declaration()?;
                continue;
            }

            if matches!(self.current_token, N3Token::ForSome) {
                self.parse_forsome_declaration()?;
                continue;
            }

            // Check if this is an implication
            if let Ok(impl_stmt) = self.try_parse_implication() {
                implications.push(impl_stmt);
                self.expect_token(N3Token::Dot)?;
                continue;
            }

            // Otherwise, parse a regular statement
            match self.parse_statement() {
                Ok(mut stmts) => {
                    statements.append(&mut stmts);
                    self.expect_token(N3Token::Dot)?;
                }
                Err(e) => {
                    if !self.lenient {
                        return Err(e);
                    }
                    // Skip to next statement in lenient mode
                    self.skip_to_next_statement();
                }
            }
        }

        Ok(N3Document {
            statements,
            implications,
            prefixes: self.prefixes.clone(),
            base_iri: self.base_iri.clone(),
        })
    }

    /// Try to parse an implication (formula => formula)
    fn try_parse_implication(&mut self) -> TurtleResult<N3Implication> {
        let antecedent = self.parse_formula()?;

        if !matches!(self.current_token, N3Token::Implies | N3Token::ImpliedBy) {
            return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Expected '=>' or '<=' for implication".to_string(),
                position: self.lexer.current_position(),
            }));
        }

        let is_forward = matches!(self.current_token, N3Token::Implies);
        self.advance()?;

        let consequent = self.parse_formula()?;

        if is_forward {
            Ok(N3Implication::new(antecedent, consequent))
        } else {
            // Reverse implication: consequent <= antecedent
            Ok(N3Implication::new(consequent, antecedent))
        }
    }

    /// Parse a formula ({ ... })
    fn parse_formula(&mut self) -> TurtleResult<N3Formula> {
        self.expect_token(N3Token::LeftBrace)?;

        let mut formula = N3Formula::new();

        // Save current quantifiers
        let saved_universals = self.universals.clone();
        let saved_existentials = self.existentials.clone();

        // Parse statements inside the formula
        while !matches!(self.current_token, N3Token::RightBrace | N3Token::Eof) {
            // Handle quantifiers inside formulas
            if matches!(self.current_token, N3Token::ForAll) {
                self.parse_forall_declaration()?;
                continue;
            }

            if matches!(self.current_token, N3Token::ForSome) {
                self.parse_forsome_declaration()?;
                continue;
            }

            // Parse statement
            let mut stmts = self.parse_statement()?;
            formula.triples.append(&mut stmts);

            // Expect dot after each statement (unless we're at the end)
            if !matches!(self.current_token, N3Token::RightBrace) {
                self.expect_token(N3Token::Dot)?;
            }
        }

        self.expect_token(N3Token::RightBrace)?;

        // Copy quantifiers into formula
        formula.universals = self.universals.clone();
        formula.existentials = self.existentials.clone();

        // Restore previous quantifiers
        self.universals = saved_universals;
        self.existentials = saved_existentials;

        Ok(formula)
    }

    /// Parse a statement: `subject predicate-object-list`.
    ///
    /// Supports Turtle/N3's abbreviated syntax for a single subject with
    /// multiple predicate-object pairs separated by `;`, and multiple
    /// objects for the same predicate separated by `,` (e.g.
    /// `:s :p1 :o1 ; :p2 :o2, :o3 .`), which is extremely common in
    /// pretty-printed documents and previously was not supported at the
    /// top level (only inside a `[ ... ]` blank-node property list).
    ///
    /// Returns every statement produced while parsing it, including any
    /// extra statements generated by blank-node property lists (`[ ... ]`)
    /// or RDF collections (`( ... )`) nested in the subject/predicate/object
    /// position.
    fn parse_statement(&mut self) -> TurtleResult<Vec<N3Statement>> {
        let mut statements = Vec::new();

        let subject = self.parse_term(TermPosition::Subject)?;
        statements.append(&mut self.pending_statements);

        loop {
            let predicate = self.parse_term(TermPosition::Predicate)?;
            statements.append(&mut self.pending_statements);

            // Object list, separated by commas.
            loop {
                let object = self.parse_term(TermPosition::Object)?;
                statements.append(&mut self.pending_statements);
                statements.push(N3Statement::new(subject.clone(), predicate.clone(), object));

                if matches!(self.current_token, N3Token::Comma) {
                    self.advance()?;
                    continue;
                }
                break;
            }

            // Predicate-object list, separated by semicolons.
            if matches!(self.current_token, N3Token::Semicolon) {
                self.advance()?;
                // Allow a trailing semicolon right before the statement
                // terminator (Dot) or the end of an enclosing formula.
                if matches!(
                    self.current_token,
                    N3Token::Dot | N3Token::RightBrace | N3Token::Eof
                ) {
                    break;
                }
                continue;
            }
            break;
        }

        Ok(statements)
    }

    /// Parse a term (can be variable, formula, or RDF term)
    fn parse_term(&mut self, position: TermPosition) -> TurtleResult<N3Term> {
        match &self.current_token {
            N3Token::Variable(name) => {
                let var = N3Variable::existential(name); // Default to existential
                self.advance()?;
                Ok(N3Term::Variable(var))
            }
            N3Token::LeftBrace => {
                let formula = self.parse_formula()?;
                Ok(N3Term::Formula(Box::new(formula)))
            }
            N3Token::Iri(iri) => {
                let node = NamedNode::new(iri).map_err(TurtleParseError::model)?;
                self.advance()?;
                Ok(N3Term::NamedNode(node))
            }
            N3Token::PrefixedName { prefix, local } => {
                let expanded = self.expand_prefixed_name(prefix, local)?;
                let node = NamedNode::new(&expanded).map_err(TurtleParseError::model)?;
                self.advance()?;
                Ok(N3Term::NamedNode(node))
            }
            N3Token::RdfType => {
                let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                    .expect("valid IRI");
                self.advance()?;
                Ok(N3Term::NamedNode(rdf_type))
            }
            N3Token::BlankNode(label) => {
                self.explicit_blank_labels.insert(label.clone());
                let bnode = BlankNode::new(label).map_err(TurtleParseError::model)?;
                self.advance()?;
                Ok(N3Term::BlankNode(bnode))
            }
            N3Token::StringLiteral(value) => {
                let lit = self.parse_literal_with_modifiers(value.clone())?;
                Ok(N3Term::Literal(lit))
            }
            N3Token::IntegerLiteral(value) => {
                let xsd_integer =
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").expect("valid IRI");
                let lit = Literal::new_typed_literal(value, xsd_integer);
                self.advance()?;
                Ok(N3Term::Literal(lit))
            }
            N3Token::DecimalLiteral(value) => {
                let xsd_decimal =
                    NamedNode::new("http://www.w3.org/2001/XMLSchema#decimal").expect("valid IRI");
                let lit = Literal::new_typed_literal(value, xsd_decimal);
                self.advance()?;
                Ok(N3Term::Literal(lit))
            }
            N3Token::LeftBracket => {
                // Anonymous blank node with property list
                self.parse_blank_node_property_list()
            }
            N3Token::LeftParen => {
                // Collection (list)
                self.parse_collection()
            }
            _ => Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!(
                    "Unexpected token in {} position: {:?}",
                    position.as_str(),
                    self.current_token
                ),
                position: self.lexer.current_position(),
            })),
        }
    }

    /// Parse a literal with optional language tag or datatype
    fn parse_literal_with_modifiers(&mut self, value: String) -> TurtleResult<Literal> {
        self.advance()?;

        // Check for language tag
        if let N3Token::LanguageTag(lang) = &self.current_token {
            let lit = Literal::new_language_tagged_literal(&value, lang).map_err(|e| {
                TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: format!("Invalid language tag: {}", e),
                    position: self.lexer.current_position(),
                })
            })?;
            self.advance()?;
            return Ok(lit);
        }

        // Check for datatype
        if matches!(self.current_token, N3Token::DatatypeMarker) {
            self.advance()?;

            let datatype = match &self.current_token {
                N3Token::Iri(iri) => NamedNode::new(iri).map_err(TurtleParseError::model)?,
                N3Token::PrefixedName { prefix, local } => {
                    let expanded = self.expand_prefixed_name(prefix, local)?;
                    NamedNode::new(&expanded).map_err(TurtleParseError::model)?
                }
                _ => {
                    return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                        message: "Expected IRI for datatype".to_string(),
                        position: self.lexer.current_position(),
                    }))
                }
            };

            self.advance()?;
            return Ok(Literal::new_typed_literal(&value, datatype));
        }

        // Plain literal
        Ok(Literal::new_simple_literal(&value))
    }

    /// Parse a blank node property list (`[ pred obj ; pred2 obj2 , obj3 ]`).
    ///
    /// Every `predicate object` pair inside the brackets becomes a statement whose
    /// subject is the (possibly anonymous) blank node standing for the whole `[...]`
    /// term; those statements are queued in `pending_statements` and flushed by the
    /// caller (mirrors `TurtleParser::parse_blank_node_property_list`).
    fn parse_blank_node_property_list(&mut self) -> TurtleResult<N3Term> {
        self.expect_token(N3Token::LeftBracket)?;

        let id = self.generate_blank_node_id();
        let bnode = BlankNode::new(&id).map_err(TurtleParseError::model)?;
        let subject = N3Term::BlankNode(bnode.clone());

        // Empty blank node: `[]`
        if matches!(self.current_token, N3Token::RightBracket) {
            self.advance()?;
            return Ok(N3Term::BlankNode(bnode));
        }

        loop {
            let predicate = self.parse_term(TermPosition::Predicate)?;

            // Object list, separated by commas
            loop {
                let object = self.parse_term(TermPosition::Object)?;
                self.pending_statements.push(N3Statement::new(
                    subject.clone(),
                    predicate.clone(),
                    object,
                ));

                if matches!(self.current_token, N3Token::Comma) {
                    self.advance()?;
                    continue;
                }
                break;
            }

            // Predicate-object list, separated by semicolons
            if matches!(self.current_token, N3Token::Semicolon) {
                self.advance()?;
                // Allow a trailing semicolon right before the closing bracket.
                if matches!(self.current_token, N3Token::RightBracket) {
                    break;
                }
                continue;
            }
            break;
        }

        self.expect_token(N3Token::RightBracket)?;

        Ok(N3Term::BlankNode(bnode))
    }

    /// Parse an RDF collection (`( item1 item2 ... )`) into an `rdf:first`/`rdf:rest`
    /// linked list of fresh blank nodes, terminated by `rdf:nil`. The list-structure
    /// statements are queued in `pending_statements` and flushed by the caller
    /// (mirrors `TurtleParser::parse_collection`).
    fn parse_collection(&mut self) -> TurtleResult<N3Term> {
        self.expect_token(N3Token::LeftParen)?;

        let rdf_nil =
            || NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil").expect("valid IRI");

        // Empty collection is rdf:nil
        if matches!(self.current_token, N3Token::RightParen) {
            self.advance()?;
            return Ok(N3Term::NamedNode(rdf_nil()));
        }

        let rdf_first =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#first").expect("valid IRI");
        let rdf_rest =
            NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest").expect("valid IRI");

        let first_id = self.generate_blank_node_id();
        let first_bn = BlankNode::new(&first_id).map_err(TurtleParseError::model)?;
        let mut current_bn = first_bn.clone();

        loop {
            let item = self.parse_term(TermPosition::Object)?;
            self.pending_statements.push(N3Statement::new(
                N3Term::BlankNode(current_bn.clone()),
                N3Term::NamedNode(rdf_first.clone()),
                item,
            ));

            if matches!(self.current_token, N3Token::RightParen) {
                self.advance()?;
                self.pending_statements.push(N3Statement::new(
                    N3Term::BlankNode(current_bn),
                    N3Term::NamedNode(rdf_rest),
                    N3Term::NamedNode(rdf_nil()),
                ));
                break;
            } else {
                let next_id = self.generate_blank_node_id();
                let next_bn = BlankNode::new(&next_id).map_err(TurtleParseError::model)?;
                self.pending_statements.push(N3Statement::new(
                    N3Term::BlankNode(current_bn),
                    N3Term::NamedNode(rdf_rest.clone()),
                    N3Term::BlankNode(next_bn.clone()),
                ));
                current_bn = next_bn;
            }
        }

        Ok(N3Term::BlankNode(first_bn))
    }

    /// Parse a prefix declaration (@prefix ex: <http://example.org/> .)
    fn parse_prefix_declaration(&mut self) -> TurtleResult<()> {
        self.expect_token(N3Token::PrefixDecl)?;

        let (prefix, namespace) = match &self.current_token {
            N3Token::PrefixedName { prefix, local } if local.is_empty() => {
                let prefix_name = prefix.clone();
                self.advance()?;

                if let N3Token::Iri(namespace) = &self.current_token {
                    let ns = namespace.clone();
                    self.advance()?;
                    (prefix_name, ns)
                } else {
                    return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                        message: "Expected IRI after prefix declaration".to_string(),
                        position: self.lexer.current_position(),
                    }));
                }
            }
            _ => {
                return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "Invalid prefix declaration syntax".to_string(),
                    position: self.lexer.current_position(),
                }))
            }
        };

        self.prefixes.insert(prefix, namespace);
        self.expect_token(N3Token::Dot)?;

        Ok(())
    }

    /// Parse a base declaration (@base <http://example.org/> .)
    fn parse_base_declaration(&mut self) -> TurtleResult<()> {
        self.expect_token(N3Token::BaseDecl)?;

        if let N3Token::Iri(base) = &self.current_token {
            self.base_iri = Some(base.clone());
            self.advance()?;
            self.expect_token(N3Token::Dot)?;
            Ok(())
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: "Expected IRI after @base".to_string(),
                position: self.lexer.current_position(),
            }))
        }
    }

    /// Parse a @forAll declaration (@forAll :x, :y .)
    fn parse_forall_declaration(&mut self) -> TurtleResult<()> {
        self.expect_token(N3Token::ForAll)?;

        loop {
            if let N3Token::Variable(name) = &self.current_token {
                let var = N3Variable::universal(name);
                self.universals.push(var);
                self.advance()?;

                if matches!(self.current_token, N3Token::Comma) {
                    self.advance()?;
                } else {
                    break;
                }
            } else {
                return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "Expected variable after @forAll".to_string(),
                    position: self.lexer.current_position(),
                }));
            }
        }

        self.expect_token(N3Token::Dot)?;
        Ok(())
    }

    /// Parse a @forSome declaration (@forSome :x, :y .)
    fn parse_forsome_declaration(&mut self) -> TurtleResult<()> {
        self.expect_token(N3Token::ForSome)?;

        loop {
            if let N3Token::Variable(name) = &self.current_token {
                let var = N3Variable::existential(name);
                self.existentials.push(var);
                self.advance()?;

                if matches!(self.current_token, N3Token::Comma) {
                    self.advance()?;
                } else {
                    break;
                }
            } else {
                return Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                    message: "Expected variable after @forSome".to_string(),
                    position: self.lexer.current_position(),
                }));
            }
        }

        self.expect_token(N3Token::Dot)?;
        Ok(())
    }

    /// Expand a prefixed name to a full IRI
    fn expand_prefixed_name(&self, prefix: &str, local: &str) -> TurtleResult<String> {
        if let Some(namespace) = self.prefixes.get(prefix) {
            Ok(format!("{}{}", namespace, local))
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Unknown prefix: {}", prefix),
                position: self.lexer.current_position(),
            }))
        }
    }

    /// Advance to the next token
    fn advance(&mut self) -> TurtleResult<()> {
        self.current_token = self.lexer.next_token()?;
        Ok(())
    }

    /// Expect a specific token
    fn expect_token(&mut self, expected: N3Token) -> TurtleResult<()> {
        if std::mem::discriminant(&self.current_token) == std::mem::discriminant(&expected) {
            self.advance()
        } else {
            Err(TurtleParseError::syntax(TurtleSyntaxError::Generic {
                message: format!("Expected {:?}, found {:?}", expected, self.current_token),
                position: self.lexer.current_position(),
            }))
        }
    }

    /// Skip to the next statement (for error recovery)
    fn skip_to_next_statement(&mut self) {
        while !matches!(self.current_token, N3Token::Dot | N3Token::Eof) {
            let _ = self.advance();
        }
        if matches!(self.current_token, N3Token::Dot) {
            let _ = self.advance();
        }
    }
}

/// Position of a term in a statement
#[derive(Debug, Copy, Clone)]
enum TermPosition {
    Subject,
    Predicate,
    Object,
}

impl TermPosition {
    fn as_str(&self) -> &str {
        match self {
            TermPosition::Subject => "subject",
            TermPosition::Predicate => "predicate",
            TermPosition::Object => "object",
        }
    }
}

/// N3 document containing statements and implications
#[derive(Debug, Clone)]
pub struct N3Document {
    /// Regular statements
    pub statements: Vec<N3Statement>,
    /// Implication rules
    pub implications: Vec<N3Implication>,
    /// Prefix declarations
    pub prefixes: HashMap<String, String>,
    /// Base IRI
    pub base_iri: Option<String>,
}

impl N3Document {
    /// Create a new empty N3 document
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
            implications: Vec::new(),
            prefixes: HashMap::new(),
            base_iri: None,
        }
    }

    /// Add a statement to the document
    pub fn add_statement(&mut self, statement: N3Statement) {
        self.statements.push(statement);
    }

    /// Add an implication to the document
    pub fn add_implication(&mut self, implication: N3Implication) {
        self.implications.push(implication);
    }

    /// Add a prefix declaration
    pub fn add_prefix(&mut self, prefix: String, namespace: String) {
        self.prefixes.insert(prefix, namespace);
    }

    /// Set the base IRI
    pub fn set_base_iri(&mut self, base_iri: String) {
        self.base_iri = Some(base_iri);
    }

    /// Check if the document is empty
    pub fn is_empty(&self) -> bool {
        self.statements.is_empty() && self.implications.is_empty()
    }

    /// Get the total number of statements and implications
    pub fn len(&self) -> usize {
        self.statements.len() + self.implications.len()
    }
}

impl Default for N3Document {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_statement() {
        let input = "@prefix ex: <http://example.org/> .\nex:alice ex:knows ex:bob .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");

        assert_eq!(doc.statements.len(), 1);
        assert_eq!(doc.implications.len(), 0);
    }

    #[test]
    fn test_parse_variable() {
        let input = "@prefix ex: <http://example.org/> .\n?x ex:knows ?y .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");

        assert_eq!(doc.statements.len(), 1);
        let stmt = &doc.statements[0];
        assert!(stmt.subject.is_variable());
        assert!(stmt.object.is_variable());
    }

    #[test]
    fn test_parse_formula() {
        let input = "{ <http://example.org/a> <http://example.org/p> <http://example.org/b> } .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let formula = parser.parse_formula().expect("parsing should succeed");

        assert_eq!(formula.len(), 1);
    }

    #[test]
    fn test_parse_implication() {
        let input =
            "@prefix ex: <http://example.org/> .\n{ ?x ex:knows ?y } => { ?y ex:knows ?x } .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");

        assert_eq!(doc.implications.len(), 1);
        let impl_rule = &doc.implications[0];
        assert_eq!(impl_rule.antecedent.len(), 1);
        assert_eq!(impl_rule.consequent.len(), 1);
    }

    #[test]
    fn test_parse_forall() {
        let input = "@forAll ?x, ?y .\n?x <http://ex.org/knows> ?y .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");

        assert_eq!(parser.universals.len(), 2);
        assert_eq!(doc.statements.len(), 1);
    }

    #[test]
    fn test_parse_non_empty_blank_node_property_list() {
        // Regression test: `[ :p :o ]` used to throw "Expected RightBracket"
        // for any non-empty property list.
        let input = "@prefix ex: <http://example.org/> .\nex:alice ex:knows [ ex:name \"Bob\" ; ex:age \"30\" ] .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");

        // 1 statement for `ex:alice ex:knows _:b` + 2 for the property list.
        assert_eq!(doc.statements.len(), 3);

        let names_used: Vec<_> = doc
            .statements
            .iter()
            .filter_map(|s| match &s.predicate {
                N3Term::NamedNode(n) => Some(n.as_str().to_string()),
                _ => None,
            })
            .collect();
        assert!(names_used.iter().any(|p| p == "http://example.org/knows"));
        assert!(names_used.iter().any(|p| p == "http://example.org/name"));
        assert!(names_used.iter().any(|p| p == "http://example.org/age"));
    }

    #[test]
    fn test_parse_empty_blank_node_property_list() {
        let input = "@prefix ex: <http://example.org/> .\nex:alice ex:knows [] .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");
        assert_eq!(doc.statements.len(), 1);
        assert!(matches!(&doc.statements[0].object, N3Term::BlankNode(_)));
    }

    #[test]
    fn test_parse_non_empty_collection() {
        // Regression test: `( 1 2 3 )` used to throw "Expected RightParen"
        // for any non-empty collection.
        let input = "@prefix ex: <http://example.org/> .\nex:alice ex:favorites ( 1 2 3 ) .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");

        // 1 statement for `ex:alice ex:favorites _:b0` + 3 rdf:first + 3 rdf:rest
        // (one of which points to rdf:nil) = 7 statements total.
        assert_eq!(doc.statements.len(), 7);

        let rdf_first_count = doc
            .statements
            .iter()
            .filter(
                |s| matches!(&s.predicate, N3Term::NamedNode(n) if n.as_str().ends_with("#first")),
            )
            .count();
        assert_eq!(rdf_first_count, 3);
    }

    #[test]
    fn test_parse_empty_collection_is_rdf_nil() {
        let input = "@prefix ex: <http://example.org/> .\nex:alice ex:favorites () .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");
        assert_eq!(doc.statements.len(), 1);
        assert!(matches!(
            &doc.statements[0].object,
            N3Term::NamedNode(n) if n.as_str() == "http://www.w3.org/1999/02/22-rdf-syntax-ns#nil"
        ));
    }

    #[test]
    fn test_parse_semicolon_predicate_object_list() {
        // Regression test: top-level `;`-separated predicate-object pairs
        // (not just inside a `[ ... ]` property list) must all be captured.
        let input = "@prefix ex: <http://example.org/> .\nex:alice ex:name \"Alice\" ; ex:age \"30\" ; ex:email \"a@example.org\" .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");
        assert_eq!(doc.statements.len(), 3);
    }

    #[test]
    fn test_parse_comma_object_list() {
        let input =
            "@prefix ex: <http://example.org/> .\nex:alice ex:knows ex:bob, ex:carol, ex:dave .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");
        assert_eq!(doc.statements.len(), 3);
    }

    #[test]
    fn test_blank_node_ids_do_not_collide_with_explicit_labels() {
        // Regression test: auto-generated blank node IDs must never collide
        // with an explicit label the document happens to also use.
        let input = "@prefix ex: <http://example.org/> .\n_:genid-b0 ex:p ex:o .\nex:s ex:knows [ ex:q ex:r ] .";
        let mut parser = AdvancedN3Parser::new(input).expect("construction should succeed");
        let doc = parser
            .parse_document()
            .expect("document parsing should succeed");

        let mut blank_labels = std::collections::HashSet::new();
        for stmt in &doc.statements {
            for term in [&stmt.subject, &stmt.object] {
                if let N3Term::BlankNode(b) = term {
                    blank_labels.insert(b.as_str().to_string());
                }
            }
        }
        // The explicit "_:genid-b0" and the auto-generated blank node for the
        // `[ ... ]` property list must be distinct identifiers.
        assert_eq!(blank_labels.len(), 2);
    }
}
