//! SPARQL 1.2 Triple Term support
//!
//! This module implements the `TripleTerm` type, a first-class RDF term that wraps
//! Subject + Predicate + Object for use in RDF-star reification. Unlike `QuotedTriple`
//! (which uses `Arc<Triple>` and focuses on embedding triples as subjects/objects),
//! `TripleTerm` is a lightweight, self-contained value type designed for SPARQL 1.2
//! triple term syntax: `<< s p o >>` as a standalone term in expressions, BIND clauses,
//! and CONSTRUCT templates.
//!
//! # SPARQL 1.2 Triple Terms
//!
//! SPARQL 1.2 introduces triple terms as first-class citizens in the query language.
//! A triple term `<< s p o >>` can appear:
//! - In BIND clauses: `BIND(<< :alice :knows :bob >> AS ?t)`
//! - In CONSTRUCT templates: `CONSTRUCT { ?t :source "db1" }`
//! - In FILTER expressions: `FILTER(?t = << :alice :knows :bob >>)`
//! - As function arguments: `TRIPLE(?s, ?p, ?o)`
//!
//! # Differences from QuotedTriple
//!
//! | Feature | TripleTerm | QuotedTriple |
//! |---------|-----------|--------------|
//! | Storage | Inline (owned components) | `Arc<Triple>` |
//! | Focus | SPARQL 1.2 expression term | RDF-star graph embedding |
//! | Parsing | `<< s p o >>` syntax | RDF-star syntax |
//! | Use case | Query algebra, BIND, FILTER | Subject/Object position |
//! | Hashing | Component-wise | Via `Arc<Triple>` |

use crate::model::{BlankNode, Literal, NamedNode, Object, Predicate, RdfTerm, Subject, Triple};
use crate::OxirsError;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A SPARQL 1.2 Triple Term: a first-class term wrapping Subject + Predicate + Object.
///
/// Triple terms allow triples to be treated as values in SPARQL expressions,
/// enabling powerful reification patterns without the overhead of named graphs
/// or blank node reification.
///
/// # Examples
///
/// ```
/// use oxirs_core::model::{NamedNode, Literal};
/// use oxirs_core::model::triple_term::TripleTerm;
///
/// let tt = TripleTerm::new(
///     NamedNode::new("http://example.org/alice").expect("valid IRI"),
///     NamedNode::new("http://example.org/knows").expect("valid IRI"),
///     NamedNode::new("http://example.org/bob").expect("valid IRI"),
/// );
///
/// assert_eq!(tt.to_string(), "<< <http://example.org/alice> <http://example.org/knows> <http://example.org/bob> >>");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TripleTerm {
    subject: Subject,
    predicate: Predicate,
    object: Object,
}

impl TripleTerm {
    /// Creates a new triple term from subject, predicate, and object.
    pub fn new(
        subject: impl Into<Subject>,
        predicate: impl Into<Predicate>,
        object: impl Into<Object>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// Creates a triple term from an existing Triple by cloning its components.
    pub fn from_triple(triple: &Triple) -> Self {
        Self {
            subject: triple.subject().clone(),
            predicate: triple.predicate().clone(),
            object: triple.object().clone(),
        }
    }

    /// Creates a triple term by consuming a Triple.
    pub fn from_owned_triple(triple: Triple) -> Self {
        let (subject, predicate, object) = triple.into_parts();
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Returns the subject of this triple term.
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// Returns the predicate of this triple term.
    pub fn predicate(&self) -> &Predicate {
        &self.predicate
    }

    /// Returns the object of this triple term.
    pub fn object(&self) -> &Object {
        &self.object
    }

    /// Decomposes this triple term into its components.
    pub fn into_parts(self) -> (Subject, Predicate, Object) {
        (self.subject, self.predicate, self.object)
    }

    /// Converts this triple term into a Triple.
    pub fn into_triple(self) -> Triple {
        Triple::new(self.subject, self.predicate, self.object)
    }

    /// Creates a Triple from this triple term (cloning components).
    pub fn to_triple(&self) -> Triple {
        Triple::new(
            self.subject.clone(),
            self.predicate.clone(),
            self.object.clone(),
        )
    }

    /// Returns a canonical N-Triples-style string representation.
    ///
    /// This is useful for serialization and comparison purposes.
    pub fn to_ntriples_string(&self) -> String {
        format!("<< {} {} {} >>", self.subject, self.predicate, self.object)
    }

    /// Parses a triple term from the SPARQL 1.2 `<< s p o >>` syntax.
    ///
    /// Supports named nodes (IRIs in angle brackets) and simple string literals.
    ///
    /// # Errors
    ///
    /// Returns `OxirsError::Parse` if the input is not valid triple term syntax.
    pub fn parse(input: &str) -> Result<Self, OxirsError> {
        let trimmed = input.trim();

        if !trimmed.starts_with("<<") || !trimmed.ends_with(">>") {
            return Err(OxirsError::Parse(
                "Triple term must be enclosed in << >>".to_string(),
            ));
        }

        // Strip the outer << >> delimiters
        let inner = trimmed[2..trimmed.len() - 2].trim();

        if inner.is_empty() {
            return Err(OxirsError::Parse("Triple term cannot be empty".to_string()));
        }

        // Tokenize: split into exactly 3 tokens respecting <...> and "..."
        let tokens = tokenize_triple_term(inner)?;

        if tokens.len() != 3 {
            return Err(OxirsError::Parse(format!(
                "Triple term requires exactly 3 components (subject, predicate, object), got {}",
                tokens.len()
            )));
        }

        let subject = parse_subject(&tokens[0])?;
        let predicate = parse_predicate(&tokens[1])?;
        let object = parse_object(&tokens[2])?;

        Ok(Self {
            subject,
            predicate,
            object,
        })
    }

    /// Returns true if the subject is a named node (IRI).
    pub fn has_iri_subject(&self) -> bool {
        matches!(self.subject, Subject::NamedNode(_))
    }

    /// Returns true if the subject is a blank node.
    pub fn has_blank_subject(&self) -> bool {
        matches!(self.subject, Subject::BlankNode(_))
    }

    /// Returns true if the object is a literal.
    pub fn has_literal_object(&self) -> bool {
        matches!(self.object, Object::Literal(_))
    }

    /// Returns true if the object is a named node (IRI).
    pub fn has_iri_object(&self) -> bool {
        matches!(self.object, Object::NamedNode(_))
    }

    /// Returns true if any component is a variable.
    pub fn has_variables(&self) -> bool {
        matches!(self.subject, Subject::Variable(_))
            || matches!(self.predicate, Predicate::Variable(_))
            || matches!(self.object, Object::Variable(_))
    }

    /// Returns a deterministic hash of this triple term.
    pub fn deterministic_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Checks structural equality: two triple terms are structurally equal
    /// if all their components are equal.
    pub fn structurally_equal(&self, other: &TripleTerm) -> bool {
        self.subject == other.subject
            && self.predicate == other.predicate
            && self.object == other.object
    }

    /// Returns a new triple term with the subject replaced.
    pub fn with_subject(mut self, subject: impl Into<Subject>) -> Self {
        self.subject = subject.into();
        self
    }

    /// Returns a new triple term with the predicate replaced.
    pub fn with_predicate(mut self, predicate: impl Into<Predicate>) -> Self {
        self.predicate = predicate.into();
        self
    }

    /// Returns a new triple term with the object replaced.
    pub fn with_object(mut self, object: impl Into<Object>) -> Self {
        self.object = object.into();
        self
    }

    /// Validates that this triple term is well-formed for SPARQL 1.2.
    ///
    /// A well-formed triple term has:
    /// - A non-variable subject (NamedNode or BlankNode)
    /// - A non-variable predicate (NamedNode)
    /// - A non-variable object (NamedNode, BlankNode, or Literal)
    pub fn is_ground(&self) -> bool {
        !self.has_variables()
    }

    /// Returns the number of distinct IRIs used in this triple term.
    pub fn iri_count(&self) -> usize {
        let mut count = 0;
        if matches!(self.subject, Subject::NamedNode(_)) {
            count += 1;
        }
        if matches!(self.predicate, Predicate::NamedNode(_)) {
            count += 1;
        }
        if matches!(self.object, Object::NamedNode(_)) {
            count += 1;
        }
        count
    }
}

impl fmt::Display for TripleTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "<< {} {} {} >>",
            self.subject, self.predicate, self.object
        )
    }
}

impl RdfTerm for TripleTerm {
    fn as_str(&self) -> &str {
        "<<triple-term>>"
    }

    fn is_quoted_triple(&self) -> bool {
        true
    }
}

impl From<Triple> for TripleTerm {
    fn from(triple: Triple) -> Self {
        Self::from_owned_triple(triple)
    }
}

impl From<TripleTerm> for Triple {
    fn from(tt: TripleTerm) -> Self {
        tt.into_triple()
    }
}

impl From<&Triple> for TripleTerm {
    fn from(triple: &Triple) -> Self {
        Self::from_triple(triple)
    }
}

/// A borrowed reference to a TripleTerm's components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TripleTermRef<'a> {
    subject: &'a Subject,
    predicate: &'a Predicate,
    object: &'a Object,
}

impl<'a> TripleTermRef<'a> {
    /// Creates a new triple term reference.
    pub fn new(subject: &'a Subject, predicate: &'a Predicate, object: &'a Object) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Creates a reference from a TripleTerm.
    pub fn from_triple_term(tt: &'a TripleTerm) -> Self {
        Self {
            subject: &tt.subject,
            predicate: &tt.predicate,
            object: &tt.object,
        }
    }

    /// Returns the subject.
    pub fn subject(&self) -> &'a Subject {
        self.subject
    }

    /// Returns the predicate.
    pub fn predicate(&self) -> &'a Predicate {
        self.predicate
    }

    /// Returns the object.
    pub fn object(&self) -> &'a Object {
        self.object
    }

    /// Converts to an owned TripleTerm.
    pub fn to_owned(&self) -> TripleTerm {
        TripleTerm {
            subject: self.subject.clone(),
            predicate: self.predicate.clone(),
            object: self.object.clone(),
        }
    }
}

impl fmt::Display for TripleTermRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "<< {} {} {} >>",
            self.subject, self.predicate, self.object
        )
    }
}

/// A collection of triple terms with lookup and filtering operations.
#[derive(Debug, Clone, Default)]
pub struct TripleTermSet {
    terms: Vec<TripleTerm>,
}

impl TripleTermSet {
    /// Creates a new empty triple term set.
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Creates a triple term set with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            terms: Vec::with_capacity(capacity),
        }
    }

    /// Adds a triple term to the set. Returns true if the term was not already present.
    pub fn insert(&mut self, term: TripleTerm) -> bool {
        if self.terms.contains(&term) {
            false
        } else {
            self.terms.push(term);
            true
        }
    }

    /// Returns the number of triple terms in the set.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Returns true if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Returns true if the set contains the given triple term.
    pub fn contains(&self, term: &TripleTerm) -> bool {
        self.terms.contains(term)
    }

    /// Returns an iterator over the triple terms.
    pub fn iter(&self) -> impl Iterator<Item = &TripleTerm> {
        self.terms.iter()
    }

    /// Filters triple terms by predicate IRI.
    pub fn filter_by_predicate(&self, predicate_iri: &str) -> Vec<&TripleTerm> {
        self.terms
            .iter()
            .filter(|tt| match &tt.predicate {
                Predicate::NamedNode(nn) => nn.as_str() == predicate_iri,
                Predicate::Variable(_) => false,
            })
            .collect()
    }

    /// Filters triple terms by subject IRI.
    pub fn filter_by_subject_iri(&self, subject_iri: &str) -> Vec<&TripleTerm> {
        self.terms
            .iter()
            .filter(|tt| match &tt.subject {
                Subject::NamedNode(nn) => nn.as_str() == subject_iri,
                _ => false,
            })
            .collect()
    }

    /// Returns all ground triple terms (no variables).
    pub fn ground_terms(&self) -> Vec<&TripleTerm> {
        self.terms.iter().filter(|tt| tt.is_ground()).collect()
    }

    /// Removes a triple term from the set. Returns true if the term was present.
    pub fn remove(&mut self, term: &TripleTerm) -> bool {
        if let Some(pos) = self.terms.iter().position(|t| t == term) {
            self.terms.swap_remove(pos);
            true
        } else {
            false
        }
    }

    /// Drains all triple terms from the set.
    pub fn drain(&mut self) -> impl Iterator<Item = TripleTerm> + '_ {
        self.terms.drain(..)
    }

    /// Converts all triple terms to Triples.
    pub fn to_triples(&self) -> Vec<Triple> {
        self.terms.iter().map(|tt| tt.to_triple()).collect()
    }
}

impl IntoIterator for TripleTermSet {
    type Item = TripleTerm;
    type IntoIter = std::vec::IntoIter<TripleTerm>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.into_iter()
    }
}

impl FromIterator<TripleTerm> for TripleTermSet {
    fn from_iter<I: IntoIterator<Item = TripleTerm>>(iter: I) -> Self {
        let mut set = Self::new();
        for term in iter {
            set.insert(term);
        }
        set
    }
}

// ─── Parsing helpers ───────────────────────────────────────────────────────────

/// Tokenizes the inner content of a `<< ... >>` triple term into exactly 3 tokens.
fn tokenize_triple_term(input: &str) -> Result<Vec<String>, OxirsError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let len = chars.len();

    while i < len {
        // Skip whitespace
        while i < len && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= len {
            break;
        }

        if chars[i] == '<' {
            // IRI: <...>
            let start = i;
            i += 1;
            while i < len && chars[i] != '>' {
                i += 1;
            }
            if i >= len {
                return Err(OxirsError::Parse(
                    "Unterminated IRI in triple term".to_string(),
                ));
            }
            i += 1; // consume '>'
            let token: String = chars[start..i].iter().collect();
            tokens.push(token);
        } else if chars[i] == '"' {
            // Literal: "..."
            let start = i;
            i += 1;
            while i < len && chars[i] != '"' {
                if chars[i] == '\\' {
                    i += 1; // skip escaped char
                }
                i += 1;
            }
            if i >= len {
                return Err(OxirsError::Parse(
                    "Unterminated literal in triple term".to_string(),
                ));
            }
            i += 1; // consume closing '"'

            // Check for language tag or datatype
            if i < len && chars[i] == '@' {
                // Language tag: "..."@en
                i += 1;
                while i < len && (chars[i].is_alphanumeric() || chars[i] == '-') {
                    i += 1;
                }
            } else if i + 1 < len && chars[i] == '^' && chars[i + 1] == '^' {
                // Datatype: "..."^^<...>
                i += 2;
                if i < len && chars[i] == '<' {
                    while i < len && chars[i] != '>' {
                        i += 1;
                    }
                    if i < len {
                        i += 1; // consume '>'
                    }
                }
            }

            let token: String = chars[start..i].iter().collect();
            tokens.push(token);
        } else if chars[i] == '_' && i + 1 < len && chars[i + 1] == ':' {
            // Blank node: _:...
            let start = i;
            i += 2;
            while i < len && !chars[i].is_whitespace() {
                i += 1;
            }
            let token: String = chars[start..i].iter().collect();
            tokens.push(token);
        } else if chars[i] == '?' || chars[i] == '$' {
            // Variable: ?var or $var
            let start = i;
            i += 1;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let token: String = chars[start..i].iter().collect();
            tokens.push(token);
        } else {
            // Unknown token - consume until whitespace
            let start = i;
            while i < len && !chars[i].is_whitespace() {
                i += 1;
            }
            let token: String = chars[start..i].iter().collect();
            tokens.push(token);
        }
    }

    Ok(tokens)
}

/// Parses a subject token.
fn parse_subject(token: &str) -> Result<Subject, OxirsError> {
    if token.starts_with('<') && token.ends_with('>') {
        let iri = &token[1..token.len() - 1];
        Ok(Subject::NamedNode(NamedNode::new(iri)?))
    } else if let Some(stripped) = token.strip_prefix("_:") {
        Ok(Subject::BlankNode(BlankNode::new(stripped)?))
    } else if token.starts_with('?') || token.starts_with('$') {
        let var_name = &token[1..];
        Ok(Subject::Variable(crate::model::Variable::new(var_name)?))
    } else {
        Err(OxirsError::Parse(format!(
            "Invalid subject in triple term: '{token}'"
        )))
    }
}

/// Parses a predicate token.
fn parse_predicate(token: &str) -> Result<Predicate, OxirsError> {
    if token.starts_with('<') && token.ends_with('>') {
        let iri = &token[1..token.len() - 1];
        Ok(Predicate::NamedNode(NamedNode::new(iri)?))
    } else if token.starts_with('?') || token.starts_with('$') {
        let var_name = &token[1..];
        Ok(Predicate::Variable(crate::model::Variable::new(var_name)?))
    } else {
        Err(OxirsError::Parse(format!(
            "Invalid predicate in triple term: '{token}'. Predicates must be IRIs or variables."
        )))
    }
}

/// Parses an object token.
fn parse_object(token: &str) -> Result<Object, OxirsError> {
    if token.starts_with('<') && token.ends_with('>') {
        let iri = &token[1..token.len() - 1];
        Ok(Object::NamedNode(NamedNode::new(iri)?))
    } else if let Some(stripped) = token.strip_prefix("_:") {
        Ok(Object::BlankNode(BlankNode::new(stripped)?))
    } else if token.starts_with('"') {
        // Parse literal with optional language tag or datatype
        let literal = parse_literal_token(token)?;
        Ok(Object::Literal(literal))
    } else if token.starts_with('?') || token.starts_with('$') {
        let var_name = &token[1..];
        Ok(Object::Variable(crate::model::Variable::new(var_name)?))
    } else {
        Err(OxirsError::Parse(format!(
            "Invalid object in triple term: '{token}'"
        )))
    }
}

/// Parses a literal token with optional language tag or datatype.
fn parse_literal_token(token: &str) -> Result<Literal, OxirsError> {
    if !token.starts_with('"') {
        return Err(OxirsError::Parse(format!("Invalid literal: '{token}'")));
    }

    // Find the closing quote (handle escapes)
    let bytes = token.as_bytes();
    let mut end_quote = 1;
    while end_quote < bytes.len() {
        if bytes[end_quote] == b'"' && (end_quote == 1 || bytes[end_quote - 1] != b'\\') {
            break;
        }
        end_quote += 1;
    }

    if end_quote >= bytes.len() {
        return Err(OxirsError::Parse(format!(
            "Unterminated literal: '{token}'"
        )));
    }

    let value = &token[1..end_quote];
    let rest = &token[end_quote + 1..];

    if let Some(lang) = rest.strip_prefix('@') {
        // Language-tagged literal
        Literal::new_lang(value, lang)
            .map_err(|e| OxirsError::Parse(format!("Invalid language tag '{lang}': {e}")))
    } else if let Some(dt_part) = rest.strip_prefix("^^") {
        // Typed literal
        if dt_part.starts_with('<') && dt_part.ends_with('>') {
            let dt_iri = &dt_part[1..dt_part.len() - 1];
            let datatype = NamedNode::new(dt_iri)?;
            Ok(Literal::new_typed(value, datatype))
        } else {
            Err(OxirsError::Parse(format!(
                "Invalid datatype IRI in literal: '{dt_part}'"
            )))
        }
    } else {
        // Simple literal
        Ok(Literal::new(value))
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{BlankNode, Literal, NamedNode, Variable};

    fn example_iri(local: &str) -> NamedNode {
        NamedNode::new(format!("http://example.org/{local}")).expect("valid IRI")
    }

    fn make_tt(s: &str, p: &str, o_iri: &str) -> TripleTerm {
        TripleTerm::new(example_iri(s), example_iri(p), example_iri(o_iri))
    }

    // ── Construction tests ──────────────────────────────────────────────────

    #[test]
    fn test_new_with_named_nodes() {
        let tt = make_tt("alice", "knows", "bob");
        assert!(tt.has_iri_subject());
        assert!(tt.has_iri_object());
        assert!(!tt.has_literal_object());
        assert!(!tt.has_blank_subject());
        assert!(!tt.has_variables());
        assert!(tt.is_ground());
    }

    #[test]
    fn test_new_with_literal_object() {
        let tt = TripleTerm::new(
            example_iri("alice"),
            example_iri("name"),
            Literal::new("Alice"),
        );
        assert!(tt.has_literal_object());
        assert!(!tt.has_iri_object());
        assert!(tt.is_ground());
    }

    #[test]
    fn test_new_with_blank_node_subject() {
        let bn = BlankNode::new("b1").expect("valid blank node");
        let tt = TripleTerm::new(bn, example_iri("type"), example_iri("Person"));
        assert!(tt.has_blank_subject());
        assert!(!tt.has_iri_subject());
        assert!(tt.is_ground());
    }

    #[test]
    fn test_new_with_variable() {
        let var = Variable::new("x").expect("valid var");
        let tt = TripleTerm::new(var, example_iri("knows"), example_iri("bob"));
        assert!(tt.has_variables());
        assert!(!tt.is_ground());
    }

    #[test]
    fn test_from_triple() {
        let triple = Triple::new(
            example_iri("alice"),
            example_iri("knows"),
            example_iri("bob"),
        );
        let tt = TripleTerm::from_triple(&triple);
        assert_eq!(tt.subject(), triple.subject());
        assert_eq!(tt.predicate(), triple.predicate());
        assert_eq!(tt.object(), triple.object());
    }

    #[test]
    fn test_from_owned_triple() {
        let triple = Triple::new(
            example_iri("alice"),
            example_iri("knows"),
            example_iri("bob"),
        );
        let triple_clone = triple.clone();
        let tt = TripleTerm::from_owned_triple(triple);
        assert_eq!(tt.subject(), triple_clone.subject());
    }

    // ── Into/From conversions ───────────────────────────────────────────────

    #[test]
    fn test_into_triple() {
        let tt = make_tt("alice", "knows", "bob");
        let triple = tt.into_triple();
        assert_eq!(triple.subject(), &Subject::NamedNode(example_iri("alice")));
    }

    #[test]
    fn test_from_triple_trait() {
        let triple = Triple::new(
            example_iri("alice"),
            example_iri("knows"),
            example_iri("bob"),
        );
        let tt: TripleTerm = triple.clone().into();
        assert_eq!(tt.to_triple(), triple);
    }

    #[test]
    fn test_from_ref_triple() {
        let triple = Triple::new(
            example_iri("alice"),
            example_iri("knows"),
            example_iri("bob"),
        );
        let tt: TripleTerm = (&triple).into();
        assert_eq!(tt.to_triple(), triple);
    }

    #[test]
    fn test_to_triple_roundtrip() {
        let tt = make_tt("s", "p", "o");
        let triple = tt.to_triple();
        let tt2 = TripleTerm::from_triple(&triple);
        assert_eq!(tt2, make_tt("s", "p", "o"));
    }

    // ── Display and parsing ─────────────────────────────────────────────────

    #[test]
    fn test_display() {
        let tt = make_tt("alice", "knows", "bob");
        let display = tt.to_string();
        assert!(display.starts_with("<<"));
        assert!(display.ends_with(">>"));
        assert!(display.contains("http://example.org/alice"));
        assert!(display.contains("http://example.org/knows"));
        assert!(display.contains("http://example.org/bob"));
    }

    #[test]
    fn test_ntriples_string() {
        let tt = make_tt("alice", "knows", "bob");
        let s = tt.to_ntriples_string();
        assert_eq!(
            s,
            "<< <http://example.org/alice> <http://example.org/knows> <http://example.org/bob> >>"
        );
    }

    #[test]
    fn test_parse_basic_iris() {
        let input =
            "<< <http://example.org/alice> <http://example.org/knows> <http://example.org/bob> >>";
        let tt = TripleTerm::parse(input).expect("should parse");
        assert_eq!(tt, make_tt("alice", "knows", "bob"));
    }

    #[test]
    fn test_parse_with_literal() {
        let input = r#"<< <http://example.org/alice> <http://example.org/name> "Alice" >>"#;
        let tt = TripleTerm::parse(input).expect("should parse");
        assert!(tt.has_literal_object());
    }

    #[test]
    fn test_parse_with_lang_literal() {
        let input = r#"<< <http://example.org/alice> <http://example.org/name> "Alice"@en >>"#;
        let tt = TripleTerm::parse(input).expect("should parse");
        if let Object::Literal(lit) = tt.object() {
            assert_eq!(lit.language(), Some("en"));
        } else {
            panic!("Expected literal object");
        }
    }

    #[test]
    fn test_parse_with_typed_literal() {
        let input = r#"<< <http://example.org/alice> <http://example.org/age> "30"^^<http://www.w3.org/2001/XMLSchema#integer> >>"#;
        let tt = TripleTerm::parse(input).expect("should parse");
        if let Object::Literal(lit) = tt.object() {
            assert_eq!(lit.value(), "30");
        } else {
            panic!("Expected literal object");
        }
    }

    #[test]
    fn test_parse_with_blank_node() {
        let input = "<< _:b1 <http://example.org/type> <http://example.org/Person> >>";
        let tt = TripleTerm::parse(input).expect("should parse");
        assert!(tt.has_blank_subject());
    }

    #[test]
    fn test_parse_with_variable() {
        let input = "<< ?s <http://example.org/knows> ?o >>";
        let tt = TripleTerm::parse(input).expect("should parse");
        assert!(tt.has_variables());
    }

    #[test]
    fn test_parse_error_no_delimiters() {
        let result = TripleTerm::parse("not a triple term");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_empty() {
        let result = TripleTerm::parse("<<  >>");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_too_few_components() {
        let result = TripleTerm::parse("<< <http://example.org/a> <http://example.org/b> >>");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_roundtrip() {
        let tt = make_tt("alice", "knows", "bob");
        let s = tt.to_ntriples_string();
        let parsed = TripleTerm::parse(&s).expect("should parse");
        assert_eq!(parsed, tt);
    }

    // ── Equality and hashing ────────────────────────────────────────────────

    #[test]
    fn test_equality_same_components() {
        let tt1 = make_tt("alice", "knows", "bob");
        let tt2 = make_tt("alice", "knows", "bob");
        assert_eq!(tt1, tt2);
    }

    #[test]
    fn test_inequality_different_subject() {
        let tt1 = make_tt("alice", "knows", "bob");
        let tt2 = make_tt("carol", "knows", "bob");
        assert_ne!(tt1, tt2);
    }

    #[test]
    fn test_inequality_different_predicate() {
        let tt1 = make_tt("alice", "knows", "bob");
        let tt2 = make_tt("alice", "likes", "bob");
        assert_ne!(tt1, tt2);
    }

    #[test]
    fn test_inequality_different_object() {
        let tt1 = make_tt("alice", "knows", "bob");
        let tt2 = make_tt("alice", "knows", "carol");
        assert_ne!(tt1, tt2);
    }

    #[test]
    fn test_hash_consistency() {
        let tt1 = make_tt("alice", "knows", "bob");
        let tt2 = make_tt("alice", "knows", "bob");
        assert_eq!(tt1.deterministic_hash(), tt2.deterministic_hash());
    }

    #[test]
    fn test_hash_different_for_different_terms() {
        let tt1 = make_tt("alice", "knows", "bob");
        let tt2 = make_tt("alice", "knows", "carol");
        // Different terms should (very likely) have different hashes
        assert_ne!(tt1.deterministic_hash(), tt2.deterministic_hash());
    }

    #[test]
    fn test_structural_equality() {
        let tt1 = make_tt("alice", "knows", "bob");
        let tt2 = make_tt("alice", "knows", "bob");
        assert!(tt1.structurally_equal(&tt2));
    }

    #[test]
    fn test_hash_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(make_tt("alice", "knows", "bob"));
        set.insert(make_tt("alice", "knows", "bob")); // duplicate
        set.insert(make_tt("alice", "knows", "carol"));
        assert_eq!(set.len(), 2);
    }

    // ── Ordering ────────────────────────────────────────────────────────────

    #[test]
    fn test_ordering() {
        let tt1 = make_tt("a", "b", "c");
        let tt2 = make_tt("a", "b", "d");
        assert!(tt1 < tt2);
    }

    #[test]
    fn test_sort_triple_terms() {
        let mut terms = [
            make_tt("c", "p", "o"),
            make_tt("a", "p", "o"),
            make_tt("b", "p", "o"),
        ];
        terms.sort();
        assert_eq!(terms[0], make_tt("a", "p", "o"));
        assert_eq!(terms[1], make_tt("b", "p", "o"));
        assert_eq!(terms[2], make_tt("c", "p", "o"));
    }

    // ── Builder methods ─────────────────────────────────────────────────────

    #[test]
    fn test_with_subject() {
        let tt = make_tt("alice", "knows", "bob").with_subject(example_iri("carol"));
        assert_eq!(tt.subject(), &Subject::NamedNode(example_iri("carol")));
    }

    #[test]
    fn test_with_predicate() {
        let tt = make_tt("alice", "knows", "bob").with_predicate(example_iri("likes"));
        assert_eq!(tt.predicate(), &Predicate::NamedNode(example_iri("likes")));
    }

    #[test]
    fn test_with_object() {
        let tt = make_tt("alice", "knows", "bob").with_object(Literal::new("hello"));
        assert!(tt.has_literal_object());
    }

    // ── TripleTermRef ───────────────────────────────────────────────────────

    #[test]
    fn test_triple_term_ref() {
        let tt = make_tt("alice", "knows", "bob");
        let ttref = TripleTermRef::from_triple_term(&tt);
        assert_eq!(ttref.subject(), tt.subject());
        assert_eq!(ttref.predicate(), tt.predicate());
        assert_eq!(ttref.object(), tt.object());
    }

    #[test]
    fn test_triple_term_ref_to_owned() {
        let tt = make_tt("alice", "knows", "bob");
        let ttref = TripleTermRef::from_triple_term(&tt);
        let owned = ttref.to_owned();
        assert_eq!(owned, tt);
    }

    #[test]
    fn test_triple_term_ref_display() {
        let tt = make_tt("alice", "knows", "bob");
        let ttref = TripleTermRef::from_triple_term(&tt);
        let display = ttref.to_string();
        assert!(display.contains("alice"));
    }

    // ── TripleTermSet ───────────────────────────────────────────────────────

    #[test]
    fn test_set_insert_and_len() {
        let mut set = TripleTermSet::new();
        assert!(set.is_empty());
        assert!(set.insert(make_tt("alice", "knows", "bob")));
        assert_eq!(set.len(), 1);
        assert!(!set.is_empty());
    }

    #[test]
    fn test_set_no_duplicates() {
        let mut set = TripleTermSet::new();
        assert!(set.insert(make_tt("alice", "knows", "bob")));
        assert!(!set.insert(make_tt("alice", "knows", "bob")));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_set_contains() {
        let mut set = TripleTermSet::new();
        let tt = make_tt("alice", "knows", "bob");
        set.insert(tt.clone());
        assert!(set.contains(&tt));
        assert!(!set.contains(&make_tt("alice", "knows", "carol")));
    }

    #[test]
    fn test_set_remove() {
        let mut set = TripleTermSet::new();
        let tt = make_tt("alice", "knows", "bob");
        set.insert(tt.clone());
        assert!(set.remove(&tt));
        assert!(set.is_empty());
        assert!(!set.remove(&tt)); // already removed
    }

    #[test]
    fn test_set_filter_by_predicate() {
        let mut set = TripleTermSet::new();
        set.insert(make_tt("alice", "knows", "bob"));
        set.insert(make_tt("alice", "likes", "carol"));
        set.insert(make_tt("bob", "knows", "carol"));

        let filtered = set.filter_by_predicate("http://example.org/knows");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_set_filter_by_subject() {
        let mut set = TripleTermSet::new();
        set.insert(make_tt("alice", "knows", "bob"));
        set.insert(make_tt("alice", "likes", "carol"));
        set.insert(make_tt("bob", "knows", "carol"));

        let filtered = set.filter_by_subject_iri("http://example.org/alice");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_set_ground_terms() {
        let mut set = TripleTermSet::new();
        set.insert(make_tt("alice", "knows", "bob"));
        let var = Variable::new("x").expect("valid var");
        set.insert(TripleTerm::new(
            var,
            example_iri("knows"),
            example_iri("bob"),
        ));

        let ground = set.ground_terms();
        assert_eq!(ground.len(), 1);
    }

    #[test]
    fn test_set_to_triples() {
        let mut set = TripleTermSet::new();
        set.insert(make_tt("alice", "knows", "bob"));
        set.insert(make_tt("alice", "knows", "carol"));
        let triples = set.to_triples();
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_set_from_iterator() {
        let terms = vec![
            make_tt("alice", "knows", "bob"),
            make_tt("alice", "knows", "bob"), // duplicate
            make_tt("alice", "knows", "carol"),
        ];
        let set: TripleTermSet = terms.into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_set_into_iterator() {
        let mut set = TripleTermSet::new();
        set.insert(make_tt("alice", "knows", "bob"));
        set.insert(make_tt("alice", "knows", "carol"));
        let collected: Vec<_> = set.into_iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_set_drain() {
        let mut set = TripleTermSet::new();
        set.insert(make_tt("a", "b", "c"));
        set.insert(make_tt("d", "e", "f"));
        let drained: Vec<_> = set.drain().collect();
        assert_eq!(drained.len(), 2);
        assert!(set.is_empty());
    }

    // ── RdfTerm trait ───────────────────────────────────────────────────────

    #[test]
    fn test_rdf_term_as_str() {
        let tt = make_tt("alice", "knows", "bob");
        assert_eq!(tt.as_str(), "<<triple-term>>");
    }

    #[test]
    fn test_rdf_term_is_quoted_triple() {
        let tt = make_tt("alice", "knows", "bob");
        assert!(tt.is_quoted_triple());
        assert!(!tt.is_named_node());
        assert!(!tt.is_blank_node());
        assert!(!tt.is_literal());
        assert!(!tt.is_variable());
    }

    // ── iri_count ───────────────────────────────────────────────────────────

    #[test]
    fn test_iri_count_all_iris() {
        let tt = make_tt("alice", "knows", "bob");
        assert_eq!(tt.iri_count(), 3);
    }

    #[test]
    fn test_iri_count_with_literal() {
        let tt = TripleTerm::new(
            example_iri("alice"),
            example_iri("name"),
            Literal::new("Alice"),
        );
        assert_eq!(tt.iri_count(), 2);
    }

    #[test]
    fn test_iri_count_with_blank_and_variable() {
        let bn = BlankNode::new("b1").expect("valid blank node");
        let var = Variable::new("x").expect("valid var");
        let tt = TripleTerm::new(bn, example_iri("type"), var);
        assert_eq!(tt.iri_count(), 1);
    }

    // ── into_parts ──────────────────────────────────────────────────────────

    #[test]
    fn test_into_parts() {
        let tt = make_tt("alice", "knows", "bob");
        let (s, p, o) = tt.into_parts();
        assert_eq!(s, Subject::NamedNode(example_iri("alice")));
        assert_eq!(p, Predicate::NamedNode(example_iri("knows")));
        assert_eq!(o, Object::NamedNode(example_iri("bob")));
    }

    // ── Serde ───────────────────────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip() {
        let tt = make_tt("alice", "knows", "bob");
        let json = serde_json::to_string(&tt).expect("serialize");
        let deserialized: TripleTerm = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, tt);
    }

    #[test]
    fn test_serde_with_literal() {
        let tt = TripleTerm::new(
            example_iri("alice"),
            example_iri("name"),
            Literal::new("Alice"),
        );
        let json = serde_json::to_string(&tt).expect("serialize");
        let deserialized: TripleTerm = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, tt);
    }

    // ── Clone and Debug ─────────────────────────────────────────────────────

    #[test]
    fn test_clone() {
        let tt = make_tt("alice", "knows", "bob");
        let tt2 = tt.clone();
        assert_eq!(tt, tt2);
    }

    #[test]
    fn test_debug_format() {
        let tt = make_tt("alice", "knows", "bob");
        let debug = format!("{:?}", tt);
        assert!(debug.contains("TripleTerm"));
    }

    #[test]
    fn test_with_capacity() {
        let set = TripleTermSet::with_capacity(100);
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }
}
