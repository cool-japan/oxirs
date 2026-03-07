//! N3 (Notation3) serializer implementation
//!
//! This module provides serialization of N3 documents with support for:
//! - **Variables**: `?var` syntax with quantifiers
//! - **Formulas**: `{ }` syntax for quoted graphs
//! - **Implications**: `=>` and `<=` for logical rules
//! - **Quantifiers**: `@forAll` and `@forSome` declarations
//! - **All Turtle features**: Prefixes, base IRIs, abbreviated syntax
//!
//! # N3 Serialization Features
//!
//! This serializer extends Turtle with N3-specific features:
//!
//! ## Variables and Quantifiers
//!
//! ```text
//! @forAll :x, :y .
//! @forSome :z .
//!
//! { ?x :parent ?y } => { ?y :child ?x } .
//! ```
//!
//! ## Formulas (Quoted Graphs)
//!
//! ```text
//! { :alice :knows :bob } :source :document1 .
//! ```
//!
//! ## Implications (Rules)
//!
//! ```text
//! { ?x a :Person } => { ?x :hasType :Human } .
//! ```
//!
//! # Example
//!
//! ```rust
//! use oxirs_ttl::n3::{N3Document, N3Statement, N3Term, N3Variable, N3Formula, N3Implication};
//! use oxirs_ttl::formats::n3_serializer::N3Serializer;
//! use oxirs_core::model::NamedNode;
//! use std::io::Cursor;
//!
//! // Create an N3 document with a rule
//! let mut antecedent = N3Formula::new();
//! antecedent.add_statement(N3Statement::new(
//!     N3Term::Variable(N3Variable::universal("x")),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/parent")?),
//!     N3Term::Variable(N3Variable::universal("y"))
//! ));
//!
//! let mut consequent = N3Formula::new();
//! consequent.add_statement(N3Statement::new(
//!     N3Term::Variable(N3Variable::universal("y")),
//!     N3Term::NamedNode(NamedNode::new("http://example.org/child")?),
//!     N3Term::Variable(N3Variable::universal("x"))
//! ));
//!
//! let implication = N3Implication::new(antecedent, consequent);
//!
//! // Serialize to N3
//! let mut output = Cursor::new(Vec::new());
//! let serializer = N3Serializer::new();
//! serializer.serialize_implication(&implication, &mut output)?;
//!
//! let n3_string = String::from_utf8(output.into_inner())?;
//! println!("{}", n3_string);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::TurtleResult;
use crate::formats::n3_parser::N3Document;
use crate::formats::n3_types::{N3Formula, N3Implication, N3Statement, N3Term};
use crate::toolkit::{FormattedWriter, SerializationConfig};
use oxirs_core::model::{Literal, NamedNode};
use std::collections::HashSet;
use std::io::Write;

/// N3 serializer for converting N3 documents to Notation3 format
///
/// Supports all N3 features including variables, formulas, implications,
/// and quantifiers, while maintaining full Turtle compatibility.
#[derive(Debug, Clone)]
pub struct N3Serializer {
    config: SerializationConfig,
}

impl Default for N3Serializer {
    fn default() -> Self {
        Self::new()
    }
}

impl N3Serializer {
    /// Create a new N3 serializer with default configuration
    pub fn new() -> Self {
        Self {
            config: SerializationConfig::default(),
        }
    }

    /// Create an N3 serializer with custom configuration
    pub fn with_config(config: SerializationConfig) -> Self {
        Self { config }
    }

    /// Serialize an N3 document to a writer
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_ttl::n3::N3Document;
    /// use oxirs_ttl::formats::n3_serializer::N3Serializer;
    /// use std::io::Cursor;
    ///
    /// let document = N3Document::new();
    /// let mut output = Cursor::new(Vec::new());
    ///
    /// let serializer = N3Serializer::new();
    /// serializer.serialize_document(&document, &mut output)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn serialize_document<W: Write>(
        &self,
        document: &N3Document,
        writer: &mut W,
    ) -> TurtleResult<()> {
        let mut formatter = FormattedWriter::new(writer, self.config.clone());

        // Write prefixes
        self.write_prefixes(&mut formatter)?;

        // Write base IRI
        if let Some(ref base) = self.config.base_iri {
            writeln!(formatter, "@base <{}> .", base)?;
            writeln!(formatter)?;
        }

        // Collect all quantified variables
        let (universals, existentials) = self.collect_quantified_vars(document);

        // Write quantifiers if present
        if !universals.is_empty() {
            self.write_quantifier(&mut formatter, "@forAll", &universals)?;
            writeln!(formatter)?;
        }

        if !existentials.is_empty() {
            self.write_quantifier(&mut formatter, "@forSome", &existentials)?;
            writeln!(formatter)?;
        }

        // Write statements
        for statement in &document.statements {
            self.serialize_statement_inner(statement, &mut formatter)?;
            writeln!(formatter, " .")?;
        }

        // Write implications
        for implication in &document.implications {
            self.serialize_implication(implication, &mut formatter)?;
        }

        formatter.flush()?;
        Ok(())
    }

    /// Serialize a single N3 statement
    fn serialize_statement_inner<W: Write>(
        &self,
        statement: &N3Statement,
        formatter: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        self.write_term(&statement.subject, formatter)?;
        write!(formatter, " ")?;
        self.write_term(&statement.predicate, formatter)?;
        write!(formatter, " ")?;
        self.write_term(&statement.object, formatter)?;
        Ok(())
    }

    /// Serialize a single N3 statement
    pub fn serialize_statement<W: Write>(
        &self,
        statement: &N3Statement,
        writer: &mut W,
    ) -> TurtleResult<()> {
        let mut formatter = FormattedWriter::new(writer, self.config.clone());
        self.serialize_statement_inner(statement, &mut formatter)?;
        formatter.flush()?;
        Ok(())
    }

    /// Serialize an N3 formula
    pub fn serialize_formula<W: Write>(
        &self,
        formula: &N3Formula,
        writer: &mut W,
    ) -> TurtleResult<()> {
        let mut formatter = FormattedWriter::new(writer, self.config.clone());
        self.write_formula(formula, &mut formatter)?;
        formatter.flush()?;
        Ok(())
    }

    /// Serialize an N3 implication (rule)
    pub fn serialize_implication<W: Write>(
        &self,
        implication: &N3Implication,
        writer: &mut W,
    ) -> TurtleResult<()> {
        let mut formatter = FormattedWriter::new(writer, self.config.clone());

        // Write antecedent
        self.write_formula(&implication.antecedent, &mut formatter)?;
        write!(formatter, " => ")?;
        // Write consequent
        self.write_formula(&implication.consequent, &mut formatter)?;
        writeln!(formatter, " .")?;

        formatter.flush()?;
        Ok(())
    }

    /// Write an N3 term (variable, formula, or RDF term)
    fn write_term<W: Write>(
        &self,
        term: &N3Term,
        formatter: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        match term {
            N3Term::NamedNode(nn) => {
                self.write_named_node(nn, formatter)?;
            }
            N3Term::BlankNode(bn) => {
                write!(formatter, "_:{}", bn.as_str())?;
            }
            N3Term::Literal(lit) => {
                self.write_literal(lit, formatter)?;
            }
            N3Term::Variable(var) => {
                write!(formatter, "?{}", var.name)?;
            }
            N3Term::Formula(formula) => {
                self.write_formula(formula, formatter)?;
            }
        }
        Ok(())
    }

    /// Write an N3 formula with braces
    fn write_formula<W: Write>(
        &self,
        formula: &N3Formula,
        formatter: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        // Build formula string without line wrapping
        let mut formula_str = String::new();

        if formula.triples.is_empty() {
            formula_str.push_str("{}");
        } else {
            formula_str.push_str("{ ");

            for (i, statement) in formula.triples.iter().enumerate() {
                if i > 0 {
                    formula_str.push_str(" . ");
                }
                formula_str.push_str(&self.term_to_string(&statement.subject)?);
                formula_str.push(' ');
                formula_str.push_str(&self.term_to_string(&statement.predicate)?);
                formula_str.push(' ');
                formula_str.push_str(&self.term_to_string(&statement.object)?);
            }

            formula_str.push_str(" }");
        }

        write!(formatter, "{}", formula_str)?;
        Ok(())
    }

    /// Convert an N3 term to a string
    fn term_to_string(&self, term: &N3Term) -> TurtleResult<String> {
        match term {
            N3Term::NamedNode(nn) => Ok(format!("<{}>", nn.as_str())),
            N3Term::BlankNode(bn) => Ok(format!("_:{}", bn.as_str())),
            N3Term::Literal(lit) => {
                let mut result = format!("\"{}\"", self.escape_string(lit.value()));
                if let Some(lang) = lit.language() {
                    result.push_str(&format!("@{}", lang));
                } else {
                    let dt = lit.datatype();
                    if dt.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                        result.push_str(&format!("^^<{}>", dt.as_str()));
                    }
                }
                Ok(result)
            }
            N3Term::Variable(var) => Ok(format!("?{}", var.name)),
            N3Term::Formula(formula) => {
                // Recursively format nested formulas
                if formula.triples.is_empty() {
                    Ok(String::from("{}"))
                } else {
                    let mut result = String::from("{ ");
                    for (i, statement) in formula.triples.iter().enumerate() {
                        if i > 0 {
                            result.push_str(" . ");
                        }
                        result.push_str(&self.term_to_string(&statement.subject)?);
                        result.push(' ');
                        result.push_str(&self.term_to_string(&statement.predicate)?);
                        result.push(' ');
                        result.push_str(&self.term_to_string(&statement.object)?);
                    }
                    result.push_str(" }");
                    Ok(result)
                }
            }
        }
    }

    /// Write prefixes
    fn write_prefixes<W: Write>(&self, formatter: &mut FormattedWriter<W>) -> TurtleResult<()> {
        if !self.config.use_prefixes {
            return Ok(());
        }

        // Sort prefixes for deterministic output
        let mut prefixes: Vec<_> = self.config.prefixes.iter().collect();
        prefixes.sort_by_key(|(prefix, _)| *prefix);

        for (prefix, namespace) in prefixes {
            writeln!(formatter, "@prefix {}: <{}> .", prefix, namespace)?;
        }

        if !self.config.prefixes.is_empty() {
            writeln!(formatter)?;
        }

        Ok(())
    }

    /// Write a quantifier declaration
    fn write_quantifier<W: Write>(
        &self,
        formatter: &mut FormattedWriter<W>,
        keyword: &str,
        variables: &HashSet<String>,
    ) -> TurtleResult<()> {
        write!(formatter, "{} ", keyword)?;

        let mut vars: Vec<_> = variables.iter().collect();
        vars.sort(); // Deterministic order

        for (i, var) in vars.iter().enumerate() {
            if i > 0 {
                write!(formatter, ", ")?;
            }
            write!(formatter, "?{}", var)?;
        }

        writeln!(formatter, " .")?;
        Ok(())
    }

    /// Write a named node with optional prefix abbreviation
    fn write_named_node<W: Write>(
        &self,
        nn: &NamedNode,
        formatter: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        if self.config.use_prefixes {
            if let Some((prefix, local)) = self.try_abbreviate_iri(nn.as_str()) {
                write!(formatter, "{}:{}", prefix, local)?;
                return Ok(());
            }
        }

        write!(formatter, "<{}>", nn.as_str())?;
        Ok(())
    }

    /// Write a literal
    fn write_literal<W: Write>(
        &self,
        lit: &Literal,
        formatter: &mut FormattedWriter<W>,
    ) -> TurtleResult<()> {
        // Write value
        let value = lit.value();
        let escaped = self.escape_string(value);
        write!(formatter, "\"{}\"", escaped)?;

        // Write language tag
        if let Some(lang) = lit.language() {
            write!(formatter, "@{}", lang)?;
        }
        // Write datatype
        else {
            let dt = lit.datatype();
            // Skip xsd:string as it's the default
            if dt.as_str() != "http://www.w3.org/2001/XMLSchema#string" {
                write!(formatter, "^^")?;
                let dt_node = NamedNode::new(dt.as_str()).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
                })?;
                self.write_named_node(&dt_node, formatter)?;
            }
        }

        Ok(())
    }

    /// Try to abbreviate an IRI using prefixes
    fn try_abbreviate_iri(&self, iri: &str) -> Option<(String, String)> {
        for (prefix, namespace) in &self.config.prefixes {
            if let Some(local) = iri.strip_prefix(namespace.as_str()) {
                // Check if local part is a valid NCName
                if is_valid_local_name(local) {
                    return Some((prefix.clone(), local.to_string()));
                }
            }
        }
        None
    }

    /// Escape special characters in string literals
    fn escape_string(&self, s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for ch in s.chars() {
            match ch {
                '"' => result.push_str("\\\""),
                '\\' => result.push_str("\\\\"),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\t' => result.push_str("\\t"),
                _ => result.push(ch),
            }
        }
        result
    }

    /// Collect all quantified variables from a document
    fn collect_quantified_vars(&self, document: &N3Document) -> (HashSet<String>, HashSet<String>) {
        let mut universals = HashSet::new();
        let mut existentials = HashSet::new();

        // Collect from statements
        for statement in &document.statements {
            self.collect_vars_from_statement(statement, &mut universals, &mut existentials);
        }

        // Collect from implications
        for implication in &document.implications {
            self.collect_vars_from_formula(
                &implication.antecedent,
                &mut universals,
                &mut existentials,
            );
            self.collect_vars_from_formula(
                &implication.consequent,
                &mut universals,
                &mut existentials,
            );
        }

        (universals, existentials)
    }

    /// Collect variables from a statement
    fn collect_vars_from_statement(
        &self,
        statement: &N3Statement,
        universals: &mut HashSet<String>,
        existentials: &mut HashSet<String>,
    ) {
        self.collect_vars_from_term(&statement.subject, universals, existentials);
        self.collect_vars_from_term(&statement.predicate, universals, existentials);
        self.collect_vars_from_term(&statement.object, universals, existentials);
    }

    /// Collect variables from a term
    fn collect_vars_from_term(
        &self,
        term: &N3Term,
        universals: &mut HashSet<String>,
        existentials: &mut HashSet<String>,
    ) {
        match term {
            N3Term::Variable(var) => {
                if var.universal {
                    universals.insert(var.name.clone());
                } else {
                    existentials.insert(var.name.clone());
                }
            }
            N3Term::Formula(formula) => {
                self.collect_vars_from_formula(formula, universals, existentials);
            }
            _ => {}
        }
    }

    /// Collect variables from a formula
    fn collect_vars_from_formula(
        &self,
        formula: &N3Formula,
        universals: &mut HashSet<String>,
        existentials: &mut HashSet<String>,
    ) {
        for var in &formula.universals {
            universals.insert(var.name.clone());
        }
        for var in &formula.existentials {
            existentials.insert(var.name.clone());
        }

        for statement in &formula.triples {
            self.collect_vars_from_statement(statement, universals, existentials);
        }
    }
}

/// Check if a string is a valid NCName for use as a local name in a prefixed name
fn is_valid_local_name(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let mut chars = s.chars();
    let first = chars.next().expect("iterator should have next element");

    // First character must be letter, underscore, or certain Unicode ranges
    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    // Remaining characters can be letters, digits, underscore, hyphen, or dot
    chars.all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::n3_types::N3Variable;

    #[test]
    fn test_serialize_variable() {
        let var = N3Term::Variable(N3Variable::universal("x"));
        let serializer = N3Serializer::new();
        let mut output = Vec::new();

        serializer
            .write_term(
                &var,
                &mut FormattedWriter::new(&mut output, SerializationConfig::default()),
            )
            .expect("operation should succeed");

        assert_eq!(String::from_utf8(output).expect("valid UTF-8"), "?x");
    }

    #[test]
    fn test_serialize_empty_formula() {
        let formula = N3Formula::new();
        let serializer = N3Serializer::new();
        let mut output = Vec::new();

        serializer
            .serialize_formula(&formula, &mut output)
            .expect("formula serialization should succeed");

        assert_eq!(String::from_utf8(output).expect("valid UTF-8"), "{}");
    }

    #[test]
    fn test_serialize_formula_with_statement() {
        let mut formula = N3Formula::new();
        formula.add_statement(N3Statement::new(
            N3Term::Variable(N3Variable::universal("x")),
            N3Term::NamedNode(NamedNode::new("http://example.org/knows").expect("valid IRI")),
            N3Term::Variable(N3Variable::universal("y")),
        ));

        let serializer = N3Serializer::new();
        let mut output = Vec::new();

        serializer
            .serialize_formula(&formula, &mut output)
            .expect("formula serialization should succeed");

        let result = String::from_utf8(output).expect("valid UTF-8");
        assert!(result.contains("?x"));
        assert!(result.contains("?y"));
        assert!(result.contains("http://example.org/knows"));
        assert!(result.starts_with('{'));
        assert!(result.ends_with('}'));
    }

    #[test]
    fn test_serialize_implication() {
        let mut antecedent = N3Formula::new();
        antecedent.add_statement(N3Statement::new(
            N3Term::Variable(N3Variable::universal("x")),
            N3Term::NamedNode(NamedNode::new("http://example.org/parent").expect("valid IRI")),
            N3Term::Variable(N3Variable::universal("y")),
        ));

        let mut consequent = N3Formula::new();
        consequent.add_statement(N3Statement::new(
            N3Term::Variable(N3Variable::universal("y")),
            N3Term::NamedNode(NamedNode::new("http://example.org/child").expect("valid IRI")),
            N3Term::Variable(N3Variable::universal("x")),
        ));

        let implication = N3Implication::new(antecedent, consequent);
        let serializer = N3Serializer::new();
        let mut output = Vec::new();

        serializer
            .serialize_implication(&implication, &mut output)
            .expect("serialization should succeed");

        let result = String::from_utf8(output).expect("valid UTF-8");
        assert!(result.contains("=>"));
        assert!(result.contains("?x"));
        assert!(result.contains("?y"));
        assert!(result.contains("parent"));
        assert!(result.contains("child"));
    }

    #[test]
    fn test_serialize_with_prefixes() {
        let config = SerializationConfig {
            use_prefixes: true,
            prefixes: {
                let mut map = std::collections::HashMap::new();
                map.insert("ex".to_string(), "http://example.org/".to_string());
                map
            },
            ..Default::default()
        };

        let serializer = N3Serializer::with_config(config);
        let mut output = Vec::new();
        let mut formatter = FormattedWriter::new(&mut output, serializer.config.clone());

        let nn = NamedNode::new("http://example.org/test").expect("valid IRI");
        serializer
            .write_named_node(&nn, &mut formatter)
            .expect("valid IRI");

        assert_eq!(String::from_utf8(output).expect("valid UTF-8"), "ex:test");
    }

    #[test]
    fn test_escape_string() {
        let serializer = N3Serializer::new();

        assert_eq!(serializer.escape_string("hello"), "hello");
        assert_eq!(serializer.escape_string("hello\"world"), "hello\\\"world");
        assert_eq!(serializer.escape_string("line1\nline2"), "line1\\nline2");
        assert_eq!(serializer.escape_string("tab\there"), "tab\\there");
    }

    #[test]
    fn test_is_valid_local_name() {
        assert!(is_valid_local_name("test"));
        assert!(is_valid_local_name("test123"));
        assert!(is_valid_local_name("test_name"));
        assert!(is_valid_local_name("test-name"));
        assert!(is_valid_local_name("test.name"));

        assert!(!is_valid_local_name(""));
        assert!(!is_valid_local_name("123test")); // Can't start with digit
        assert!(!is_valid_local_name("-test")); // Can't start with hyphen
    }

    #[test]
    fn test_collect_quantified_vars() {
        let mut document = N3Document::new();

        let stmt = N3Statement::new(
            N3Term::Variable(N3Variable::universal("x")),
            N3Term::NamedNode(NamedNode::new("http://example.org/type").expect("valid IRI")),
            N3Term::Variable(N3Variable::existential("z")),
        );
        document.add_statement(stmt);

        let serializer = N3Serializer::new();
        let (universals, existentials) = serializer.collect_quantified_vars(&document);

        assert!(universals.contains("x"));
        assert!(existentials.contains("z"));
        assert_eq!(universals.len(), 1);
        assert_eq!(existentials.len(), 1);
    }
}
