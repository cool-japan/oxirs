//! RDF-star (RDF*) support for statement annotations
//!
//! This module implements RDF-star extensions for SPARQL 1.2 compliance,
//! allowing triples to be used as subjects or objects in other triples.

use crate::model::{
    NamedNode, Object, ObjectTerm, Predicate, RdfTerm, Subject, SubjectTerm, Triple,
};
use crate::query::algebra::{AlgebraTriplePattern, TermPattern};
use crate::OxirsError;
use std::fmt;
use std::sync::Arc;

/// A quoted triple that can be used as a subject or object in RDF-star
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuotedTriple {
    /// The inner triple being quoted
    inner: Arc<Triple>,
}

impl QuotedTriple {
    /// Create a new quoted triple
    pub fn new(triple: Triple) -> Self {
        QuotedTriple {
            inner: Arc::new(triple),
        }
    }

    /// Create from an existing `Arc<Triple>`
    pub fn from_arc(triple: Arc<Triple>) -> Self {
        QuotedTriple { inner: triple }
    }

    /// Get the inner triple
    pub fn inner(&self) -> &Triple {
        &self.inner
    }

    /// Get the subject of the quoted triple
    pub fn subject(&self) -> &Subject {
        self.inner.subject()
    }

    /// Get the predicate of the quoted triple
    pub fn predicate(&self) -> &Predicate {
        self.inner.predicate()
    }

    /// Get the object of the quoted triple
    pub fn object(&self) -> &Object {
        self.inner.object()
    }

    /// Convert to a triple reference
    pub fn as_ref(&self) -> QuotedTripleRef<'_> {
        QuotedTripleRef { inner: &self.inner }
    }
}

impl fmt::Display for QuotedTriple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<< {} >>", self.inner)
    }
}

impl RdfTerm for QuotedTriple {
    fn as_str(&self) -> &str {
        // For quoted triples, we return a synthetic string representation
        "<<quoted-triple>>"
    }

    fn is_quoted_triple(&self) -> bool {
        true
    }
}

impl SubjectTerm for QuotedTriple {}
impl ObjectTerm for QuotedTriple {}

// Custom serialization for QuotedTriple to handle Arc<Triple>
#[cfg(feature = "serde")]
impl serde::Serialize for QuotedTriple {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize the inner triple directly
        self.inner.as_ref().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for QuotedTriple {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let triple = Triple::deserialize(deserializer)?;
        Ok(QuotedTriple::new(triple))
    }
}

/// A borrowed quoted triple reference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QuotedTripleRef<'a> {
    inner: &'a Triple,
}

impl<'a> QuotedTripleRef<'a> {
    /// Create a new quoted triple reference
    pub fn new(triple: &'a Triple) -> Self {
        QuotedTripleRef { inner: triple }
    }

    /// Get the inner triple
    pub fn inner(&self) -> &'a Triple {
        self.inner
    }

    /// Convert to owned quoted triple
    pub fn to_owned(&self) -> QuotedTriple {
        QuotedTriple::new(self.inner.clone())
    }
}

impl<'a> fmt::Display for QuotedTripleRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<< {} >>", self.inner)
    }
}

impl<'a> RdfTerm for QuotedTripleRef<'a> {
    fn as_str(&self) -> &str {
        "<<quoted-triple>>"
    }

    fn is_quoted_triple(&self) -> bool {
        true
    }
}

/// RDF-star annotation syntax support
pub struct Annotation {
    /// The statement being annotated (as a quoted triple)
    pub statement: QuotedTriple,
    /// The annotation property
    pub property: NamedNode,
    /// The annotation value
    pub value: Object,
}

impl Annotation {
    /// Create a new annotation
    pub fn new(statement: Triple, property: NamedNode, value: Object) -> Self {
        Annotation {
            statement: QuotedTriple::new(statement),
            property,
            value,
        }
    }

    /// Convert annotation to a regular triple with quoted triple as subject
    pub fn to_triple(&self) -> Triple {
        Triple::new(
            Subject::QuotedTriple(Box::new(self.statement.clone())),
            self.property.clone(),
            self.value.clone(),
        )
    }
}

/// RDF-star pattern for SPARQL queries
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StarPattern {
    /// A regular triple pattern
    Triple(AlgebraTriplePattern),
    /// A quoted triple pattern
    QuotedTriple {
        /// Subject pattern (can be nested quoted triple)
        subject: Box<StarPattern>,
        /// Predicate pattern
        predicate: TermPattern,
        /// Object pattern (can be nested quoted triple)
        object: Box<StarPattern>,
    },
    /// An annotation pattern
    Annotation {
        /// The annotated statement pattern
        statement: Box<StarPattern>,
        /// Annotation property pattern
        property: TermPattern,
        /// Annotation value pattern
        value: TermPattern,
    },
}

impl StarPattern {
    /// Check if pattern contains variables
    pub fn has_variables(&self) -> bool {
        match self {
            StarPattern::Triple(pattern) => {
                matches!(pattern.subject, TermPattern::Variable(_))
                    || matches!(pattern.predicate, TermPattern::Variable(_))
                    || matches!(pattern.object, TermPattern::Variable(_))
            }
            StarPattern::QuotedTriple {
                subject,
                predicate: _,
                object,
            } => subject.has_variables() || object.has_variables(),
            StarPattern::Annotation {
                statement,
                property: _,
                value: _,
            } => statement.has_variables(),
        }
    }

    /// Get all variables in the pattern
    pub fn variables(&self) -> Vec<crate::model::Variable> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars
    }

    fn collect_variables(&self, _vars: &mut Vec<crate::model::Variable>) {
        match self {
            StarPattern::Triple(pattern) => {
                if let TermPattern::Variable(ref v) = pattern.subject {
                    _vars.push(v.clone());
                }
                if let TermPattern::Variable(ref v) = pattern.predicate {
                    _vars.push(v.clone());
                }
                if let TermPattern::Variable(ref v) = pattern.object {
                    _vars.push(v.clone());
                }
            }
            StarPattern::QuotedTriple {
                subject,
                predicate: _,
                object,
            } => {
                subject.collect_variables(_vars);
                object.collect_variables(_vars);
            }
            StarPattern::Annotation {
                statement,
                property: _,
                value: _,
            } => {
                statement.collect_variables(_vars);
            }
        }
    }
}

/// RDF-star serialization format extensions
pub mod serialization {
    use super::*;

    /// Turtle-star syntax extensions
    pub mod turtle_star {
        use super::*;

        /// Serialize a quoted triple in Turtle-star syntax
        pub fn serialize_quoted_triple(qt: &QuotedTriple) -> String {
            format!("<< {} {} {} >>", qt.subject(), qt.predicate(), qt.object())
        }

        /// Parse a quoted triple from Turtle-star syntax.
        ///
        /// Supports the RDF-star subject/object term grammar: absolute IRIs
        /// (`<...>`), blank nodes (`_:id`), plain/typed/language-tagged
        /// literals (`"..."`, `"..."^^<...>`, `"..."@lang`), and nested
        /// quoted triples (`<< ... >>`). This is a self-contained
        /// tokenizer scoped to the `<< s p o >>` grammar rather than a
        /// full Turtle document parser; prefixed names (`ex:foo`) are not
        /// supported since there is no prefix map in scope here.
        pub fn parse_quoted_triple(input: &str) -> Result<QuotedTriple, OxirsError> {
            let trimmed = input.trim();
            if !trimmed.starts_with("<<") || !trimmed.ends_with(">>") || trimmed.len() < 4 {
                return Err(OxirsError::Parse(
                    "Invalid quoted triple syntax: expected '<< subject predicate object >>'"
                        .to_string(),
                ));
            }

            let inner = trimmed[2..trimmed.len() - 2].trim();
            let terms = split_star_terms(inner)?;
            if terms.len() != 3 {
                return Err(OxirsError::Parse(format!(
                    "Invalid quoted triple: expected exactly 3 terms (subject, predicate, object), found {}",
                    terms.len()
                )));
            }

            let subject = parse_star_subject(terms[0])?;
            let predicate = parse_star_predicate(terms[1])?;
            let object = parse_star_object(terms[2])?;

            Ok(QuotedTriple::new(Triple::new(subject, predicate, object)))
        }

        /// Split the space-separated `subject predicate object` term list
        /// of a quoted triple, respecting nesting of `<< ... >>`, `<...>`,
        /// and `"..."` (including escaped quotes) so that whitespace
        /// inside those constructs does not cause a spurious split.
        fn split_star_terms(input: &str) -> Result<Vec<&str>, OxirsError> {
            let bytes = input.as_bytes();
            let mut terms = Vec::new();
            let mut i = 0usize;
            let mut depth = 0i32;
            let mut in_string = false;
            let mut escape = false;
            let mut term_start: Option<usize> = None;

            while i < bytes.len() {
                let c = bytes[i] as char;
                if in_string {
                    if escape {
                        escape = false;
                    } else if c == '\\' {
                        escape = true;
                    } else if c == '"' {
                        in_string = false;
                    }
                    i += 1;
                    continue;
                }

                match c {
                    '"' => {
                        in_string = true;
                        if term_start.is_none() {
                            term_start = Some(i);
                        }
                    }
                    '<' if input[i..].starts_with("<<") => {
                        depth += 1;
                        if term_start.is_none() {
                            term_start = Some(i);
                        }
                        i += 1; // consume the extra '<' of "<<"
                    }
                    '>' if input[i..].starts_with(">>") => {
                        depth -= 1;
                        if depth < 0 {
                            return Err(OxirsError::Parse(
                                "Unbalanced '>>' in quoted triple term".to_string(),
                            ));
                        }
                        i += 1; // consume the extra '>' of ">>"
                    }
                    '<' => {
                        if term_start.is_none() {
                            term_start = Some(i);
                        }
                    }
                    c if c.is_whitespace() && depth == 0 => {
                        if let Some(start) = term_start.take() {
                            terms.push(input[start..i].trim());
                        }
                    }
                    _ => {
                        if term_start.is_none() {
                            term_start = Some(i);
                        }
                    }
                }
                i += 1;
            }
            if in_string {
                return Err(OxirsError::Parse(
                    "Unterminated string literal in quoted triple".to_string(),
                ));
            }
            if depth != 0 {
                return Err(OxirsError::Parse(
                    "Unbalanced '<<'/'>>' nesting in quoted triple".to_string(),
                ));
            }
            if let Some(start) = term_start {
                terms.push(input[start..].trim());
            }
            Ok(terms.into_iter().filter(|t| !t.is_empty()).collect())
        }

        fn parse_star_iri(term: &str) -> Result<NamedNode, OxirsError> {
            let inner = term
                .strip_prefix('<')
                .and_then(|s| s.strip_suffix('>'))
                .ok_or_else(|| OxirsError::Parse(format!("Invalid IRI term: {term}")))?;
            NamedNode::new(inner)
        }

        fn parse_star_literal(term: &str) -> Result<crate::model::Literal, OxirsError> {
            use crate::model::Literal;

            // Locate the closing quote of the (possibly escaped) string body.
            let bytes = term.as_bytes();
            if bytes.first() != Some(&b'"') {
                return Err(OxirsError::Parse(format!("Invalid literal term: {term}")));
            }
            let mut end = None;
            let mut escape = false;
            for (idx, b) in bytes.iter().enumerate().skip(1) {
                if escape {
                    escape = false;
                } else if *b == b'\\' {
                    escape = true;
                } else if *b == b'"' {
                    end = Some(idx);
                    break;
                }
            }
            let end =
                end.ok_or_else(|| OxirsError::Parse(format!("Unterminated literal: {term}")))?;
            let raw_value = &term[1..end];
            let value = unescape_star_string(raw_value)?;
            let suffix = &term[end + 1..];

            if let Some(lang) = suffix.strip_prefix('@') {
                Literal::new_language_tagged_literal(value, lang)
                    .map_err(|e| OxirsError::Parse(format!("Invalid language tag: {e}")))
            } else if let Some(datatype) = suffix.strip_prefix("^^") {
                let datatype = parse_star_iri(datatype)?;
                Ok(Literal::new_typed(value, datatype))
            } else if suffix.is_empty() {
                Ok(Literal::new(value))
            } else {
                Err(OxirsError::Parse(format!(
                    "Invalid literal suffix in term: {term}"
                )))
            }
        }

        fn unescape_star_string(raw: &str) -> Result<String, OxirsError> {
            let mut result = String::with_capacity(raw.len());
            let mut chars = raw.chars();
            while let Some(c) = chars.next() {
                if c != '\\' {
                    result.push(c);
                    continue;
                }
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('r') => result.push('\r'),
                    Some('"') => result.push('"'),
                    Some('\\') => result.push('\\'),
                    Some(other) => {
                        return Err(OxirsError::Parse(format!(
                            "Unsupported escape sequence '\\{other}' in quoted triple literal"
                        )))
                    }
                    None => {
                        return Err(OxirsError::Parse(
                            "Trailing backslash in quoted triple literal".to_string(),
                        ))
                    }
                }
            }
            Ok(result)
        }

        fn parse_star_subject(term: &str) -> Result<Subject, OxirsError> {
            if term.starts_with("<<") {
                Ok(Subject::QuotedTriple(Box::new(parse_quoted_triple(term)?)))
            } else if let Some(id) = term.strip_prefix("_:") {
                Ok(Subject::BlankNode(crate::model::BlankNode::new(id)?))
            } else if term.starts_with('<') {
                Ok(Subject::NamedNode(parse_star_iri(term)?))
            } else {
                Err(OxirsError::Parse(format!(
                    "Unsupported subject term in quoted triple: {term}"
                )))
            }
        }

        fn parse_star_predicate(term: &str) -> Result<Predicate, OxirsError> {
            if term.starts_with('<') {
                Ok(Predicate::NamedNode(parse_star_iri(term)?))
            } else {
                Err(OxirsError::Parse(format!(
                    "Unsupported predicate term in quoted triple: {term}"
                )))
            }
        }

        fn parse_star_object(term: &str) -> Result<Object, OxirsError> {
            if term.starts_with("<<") {
                Ok(Object::QuotedTriple(Box::new(parse_quoted_triple(term)?)))
            } else if let Some(id) = term.strip_prefix("_:") {
                Ok(Object::BlankNode(crate::model::BlankNode::new(id)?))
            } else if term.starts_with('<') {
                Ok(Object::NamedNode(parse_star_iri(term)?))
            } else if term.starts_with('"') {
                Ok(Object::Literal(parse_star_literal(term)?))
            } else {
                Err(OxirsError::Parse(format!(
                    "Unsupported object term in quoted triple: {term}"
                )))
            }
        }
    }

    /// SPARQL-star syntax extensions
    pub mod sparql_star {
        use super::*;

        /// Format a star pattern for SPARQL
        pub fn format_star_pattern(pattern: &StarPattern) -> String {
            match pattern {
                StarPattern::Triple(pattern) => pattern.to_string(),
                StarPattern::QuotedTriple {
                    subject,
                    predicate: _,
                    object,
                } => {
                    format!(
                        "<< {} {} {} >>",
                        format_star_pattern(subject),
                        "PREDICATE",
                        format_star_pattern(object)
                    )
                }
                StarPattern::Annotation {
                    statement,
                    property: _,
                    value: _,
                } => {
                    format!(
                        "{} {} {}",
                        format_star_pattern(statement),
                        "PROPERTY",
                        "VALUE"
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[test]
    fn test_quoted_triple() {
        let subject = NamedNode::new("http://example.org/alice").expect("valid IRI");
        let predicate = NamedNode::new("http://example.org/says").expect("valid IRI");
        let object = Object::Literal(Literal::new("Hello"));

        let triple = Triple::new(subject, predicate, object);
        let quoted = QuotedTriple::new(triple.clone());

        assert_eq!(quoted.inner(), &triple);
        assert_eq!(
            format!("{quoted}"),
            "<< <http://example.org/alice> <http://example.org/says> \"Hello\" . >>"
        );
    }

    #[test]
    fn test_annotation() {
        let subject = NamedNode::new("http://example.org/alice").expect("valid IRI");
        let predicate = NamedNode::new("http://example.org/age").expect("valid IRI");
        let object = Object::Literal(Literal::new_typed(
            "30",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").expect("valid IRI"),
        ));

        let statement = Triple::new(subject, predicate, object);
        let ann_property = NamedNode::new("http://example.org/confidence").expect("valid IRI");
        let ann_value = Object::Literal(Literal::new_typed(
            "0.9",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#double").expect("valid IRI"),
        ));

        let annotation = Annotation::new(statement, ann_property, ann_value);
        let ann_triple = annotation.to_triple();

        assert!(matches!(ann_triple.subject(), Subject::QuotedTriple(_)));
    }

    #[test]
    fn test_parse_quoted_triple_basic() {
        use serialization::turtle_star::parse_quoted_triple;

        let parsed = parse_quoted_triple(
            "<< <http://example.org/alice> <http://example.org/says> \"Hello\" >>",
        )
        .expect("valid quoted triple");

        assert_eq!(
            parsed.subject(),
            &Subject::NamedNode(NamedNode::new("http://example.org/alice").expect("valid IRI"))
        );
        assert_eq!(
            parsed.predicate(),
            &Predicate::NamedNode(NamedNode::new("http://example.org/says").expect("valid IRI"))
        );
        assert_eq!(parsed.object(), &Object::Literal(Literal::new("Hello")));
    }

    #[test]
    fn test_parse_quoted_triple_roundtrip_with_serialize() {
        use serialization::turtle_star::{parse_quoted_triple, serialize_quoted_triple};

        let subject = NamedNode::new("http://example.org/alice").expect("valid IRI");
        let predicate = NamedNode::new("http://example.org/age").expect("valid IRI");
        let object = Object::Literal(Literal::new_typed(
            "30",
            NamedNode::new("http://www.w3.org/2001/XMLSchema#integer").expect("valid IRI"),
        ));
        let original = QuotedTriple::new(Triple::new(subject, predicate, object));

        let text = serialize_quoted_triple(&original);
        let parsed = parse_quoted_triple(&text).expect("round-trip parse");
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_parse_quoted_triple_nested() {
        use serialization::turtle_star::parse_quoted_triple;

        let input = "<< << <http://example.org/a> <http://example.org/p> <http://example.org/b> >> <http://example.org/certainty> \"0.9\"^^<http://www.w3.org/2001/XMLSchema#double> >>";
        let parsed = parse_quoted_triple(input).expect("valid nested quoted triple");
        assert!(matches!(parsed.subject(), Subject::QuotedTriple(_)));
    }

    #[test]
    fn test_parse_quoted_triple_rejects_bad_syntax() {
        use serialization::turtle_star::parse_quoted_triple;

        assert!(parse_quoted_triple("not a quoted triple").is_err());
        assert!(
            parse_quoted_triple("<< <http://example.org/a> <http://example.org/b> >>").is_err()
        );
    }
}
