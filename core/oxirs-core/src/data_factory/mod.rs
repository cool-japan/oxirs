//! # W3C RDF DataFactory API
//!
//! A Rust implementation of the
//! [W3C RDF/JS Data Model specification](https://rdf.js.org/data-model-spec/)
//! `DataFactory` interface.  The API mirrors the JavaScript `N3.js` / `rdf-ext`
//! ecosystem so that tooling that expects the W3C interface can be ported to
//! Rust without surprises.
//!
//! ## Quick start
//!
//! ```rust
//! use oxirs_core::data_factory::{DataFactory, xsd_types, vocab};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Named nodes (IRIs)
//! let alice = DataFactory::named_node("http://example.org/alice")?;
//! let knows = DataFactory::named_node("http://xmlns.com/foaf/0.1/knows")?;
//! let bob   = DataFactory::named_node("http://example.org/bob")?;
//!
//! // Create a triple
//! let triple = DataFactory::triple(alice.clone().into(), knows.clone(), bob.clone().into());
//!
//! // Language-tagged literal
//! let hello = DataFactory::language_literal("Hello", "en")?;
//!
//! // Typed literal
//! let age = DataFactory::typed_literal("42", xsd_types::integer());
//!
//! // Blank nodes
//! let b = DataFactory::blank_node();
//! let b2 = DataFactory::blank_node_with_id("my-id");
//!
//! // Quads
//! let graph = DataFactory::default_graph();
//! let quad  = DataFactory::quad(alice.into(), knows, bob.into(), graph);
//! # Ok(())
//! # }
//! ```

use crate::model::{BlankNode, GraphName, Literal, NamedNode, Object, Quad, Subject, Triple};
use crate::{OxirsError, Result};
use oxiri::Iri;

// ── DataFactory ───────────────────────────────────────────────────────────────

/// Stateless factory for constructing RDF terms, triples and quads.
///
/// Every method is `pub` and `fn` (not `self`); the struct itself needs no
/// instance – it is used as a namespace.
pub struct DataFactory;

impl DataFactory {
    // ── Named nodes ───────────────────────────────────────────────────────────

    /// Create a [`NamedNode`] from an IRI string, validating it.
    ///
    /// Returns [`OxirsError::Parse`] if the string is not a well-formed IRI.
    ///
    /// ```rust
    /// use oxirs_core::data_factory::DataFactory;
    /// let n = DataFactory::named_node("http://example.org/s").unwrap();
    /// assert_eq!(n.as_str(), "http://example.org/s");
    /// ```
    pub fn named_node(iri: impl Into<String>) -> Result<NamedNode> {
        NamedNode::new(iri)
    }

    // ── Blank nodes ───────────────────────────────────────────────────────────

    /// Create a [`BlankNode`] with an auto-generated unique identifier.
    ///
    /// ```rust
    /// use oxirs_core::data_factory::DataFactory;
    /// let b1 = DataFactory::blank_node();
    /// let b2 = DataFactory::blank_node();
    /// assert_ne!(b1.as_str(), b2.as_str());
    /// ```
    pub fn blank_node() -> BlankNode {
        BlankNode::new_unique()
    }

    /// Create a [`BlankNode`] with a specific local identifier.
    ///
    /// The identifier must match `[a-zA-Z0-9_][a-zA-Z0-9_.-]*`; invalid
    /// strings are silently replaced with a generated one.
    ///
    /// ```rust
    /// use oxirs_core::data_factory::DataFactory;
    /// let b = DataFactory::blank_node_with_id("my-node");
    /// assert_eq!(b.as_str(), "my-node");
    /// ```
    pub fn blank_node_with_id(id: impl Into<String>) -> BlankNode {
        let s = id.into();
        BlankNode::new(s.clone()).unwrap_or_else(|_| BlankNode::new_unique())
    }

    // ── Literals ──────────────────────────────────────────────────────────────

    /// Create a plain (xsd:string) [`Literal`].
    ///
    /// ```rust
    /// use oxirs_core::data_factory::DataFactory;
    /// let l = DataFactory::literal("hello");
    /// assert_eq!(l.value(), "hello");
    /// ```
    pub fn literal(value: impl Into<String>) -> Literal {
        Literal::new(value)
    }

    /// Create an explicitly-typed [`Literal`].
    ///
    /// ```rust
    /// use oxirs_core::data_factory::{DataFactory, xsd_types};
    /// let l = DataFactory::typed_literal("42", xsd_types::integer());
    /// assert_eq!(l.value(), "42");
    /// ```
    pub fn typed_literal(value: impl Into<String>, datatype: NamedNode) -> Literal {
        Literal::new_typed(value, datatype)
    }

    /// Create a language-tagged [`Literal`] (BCP 47 language tag).
    ///
    /// Returns [`OxirsError::Parse`] if `lang` is not a valid BCP 47 tag.
    ///
    /// ```rust
    /// use oxirs_core::data_factory::DataFactory;
    /// let l = DataFactory::language_literal("Bonjour", "fr").unwrap();
    /// assert_eq!(l.language(), Some("fr"));
    /// ```
    pub fn language_literal(value: impl Into<String>, lang: impl Into<String>) -> Result<Literal> {
        let lang_str = lang.into();
        Self::validate_lang_tag(&lang_str)?;
        Literal::new_lang(value, lang_str)
    }

    // ── Triples ───────────────────────────────────────────────────────────────

    /// Create a [`Triple`] from subject, predicate, and object.
    ///
    /// ```rust
    /// use oxirs_core::data_factory::DataFactory;
    /// use oxirs_core::model::{Subject, Object};
    /// let s = DataFactory::named_node("http://example.org/s").unwrap();
    /// let p = DataFactory::named_node("http://example.org/p").unwrap();
    /// let o = DataFactory::named_node("http://example.org/o").unwrap();
    /// let triple = DataFactory::triple(s.into(), p, o.into());
    /// ```
    pub fn triple(subject: Subject, predicate: NamedNode, object: Object) -> Triple {
        Triple::new(subject, predicate, object)
    }

    // ── Quads ─────────────────────────────────────────────────────────────────

    /// Create a [`Quad`] from subject, predicate, object, and graph name.
    ///
    /// Use [`Self::default_graph()`] for the default graph.
    ///
    /// ```rust
    /// use oxirs_core::data_factory::DataFactory;
    /// use oxirs_core::model::{Subject, Object};
    /// let s = DataFactory::named_node("http://example.org/s").unwrap();
    /// let p = DataFactory::named_node("http://example.org/p").unwrap();
    /// let o = DataFactory::named_node("http://example.org/o").unwrap();
    /// let g = DataFactory::default_graph();
    /// let quad = DataFactory::quad(s.into(), p, o.into(), g);
    /// ```
    pub fn quad(subject: Subject, predicate: NamedNode, object: Object, graph: GraphName) -> Quad {
        Quad::new(subject, predicate, object, graph)
    }

    // ── Graph names ───────────────────────────────────────────────────────────

    /// Return the default [`GraphName`].
    pub fn default_graph() -> GraphName {
        GraphName::DefaultGraph
    }

    /// Create a named-graph [`GraphName`] from a validated IRI.
    ///
    /// Returns [`OxirsError::Parse`] if the string is not a well-formed IRI.
    pub fn named_graph(iri: impl Into<String>) -> Result<GraphName> {
        let nn = NamedNode::new(iri)?;
        Ok(GraphName::NamedNode(nn))
    }

    // ── Validation ────────────────────────────────────────────────────────────

    /// Validate an IRI string without constructing a [`NamedNode`].
    ///
    /// ```rust
    /// use oxirs_core::data_factory::DataFactory;
    /// assert!(DataFactory::validate_iri("http://example.org/").is_ok());
    /// assert!(DataFactory::validate_iri("not an IRI").is_err());
    /// ```
    pub fn validate_iri(iri: &str) -> Result<()> {
        Iri::parse(iri.to_owned())
            .map(|_| ())
            .map_err(|e| OxirsError::Parse(format!("Invalid IRI '{iri}': {e}")))
    }

    /// Validate a BCP 47 language tag string.
    ///
    /// ```rust
    /// use oxirs_core::data_factory::DataFactory;
    /// assert!(DataFactory::validate_lang_tag("en").is_ok());
    /// assert!(DataFactory::validate_lang_tag("en-US").is_ok());
    /// assert!(DataFactory::validate_lang_tag("zh-Hans-CN").is_ok());
    /// assert!(DataFactory::validate_lang_tag("").is_err());
    /// ```
    pub fn validate_lang_tag(lang: &str) -> Result<()> {
        if lang.is_empty() {
            return Err(OxirsError::Parse(
                "Language tag must not be empty".to_string(),
            ));
        }
        // BCP 47 primary subtag: 1–8 ASCII alpha characters.
        // We apply a lightweight structural check sufficient for practical use.
        for part in lang.split('-') {
            if part.is_empty() {
                return Err(OxirsError::Parse(format!(
                    "Invalid language tag '{lang}': empty subtag"
                )));
            }
            // Each subtag must be alphanumeric
            if !part.chars().all(|c| c.is_ascii_alphanumeric()) {
                return Err(OxirsError::Parse(format!(
                    "Invalid language tag '{lang}': subtag '{part}' contains non-alphanumeric characters"
                )));
            }
        }
        // Primary subtag must be alphabetic (BCP 47 §2.2.1)
        let primary = lang.split('-').next().unwrap_or(lang);
        if !primary.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(OxirsError::Parse(format!(
                "Invalid language tag '{lang}': primary subtag must be alphabetic"
            )));
        }
        Ok(())
    }
}

// ── XSD Datatype helpers ──────────────────────────────────────────────────────

/// Functions that return [`NamedNode`]s for the most common XSD datatypes.
///
/// These are unchecked because the IRIs are compile-time constants.
pub mod xsd_types {
    use crate::model::NamedNode;

    const XSD: &str = "http://www.w3.org/2001/XMLSchema#";

    /// `xsd:string`
    pub fn string() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}string"))
    }
    /// `xsd:integer`
    pub fn integer() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}integer"))
    }
    /// `xsd:float`
    pub fn float() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}float"))
    }
    /// `xsd:double`
    pub fn double() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}double"))
    }
    /// `xsd:boolean`
    pub fn boolean() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}boolean"))
    }
    /// `xsd:dateTime`
    pub fn date_time() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}dateTime"))
    }
    /// `xsd:date`
    pub fn date() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}date"))
    }
    /// `xsd:decimal`
    pub fn decimal() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}decimal"))
    }
    /// `xsd:long`
    pub fn long() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}long"))
    }
    /// `xsd:int`
    pub fn int() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}int"))
    }
    /// `xsd:short`
    pub fn short() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}short"))
    }
    /// `xsd:byte`
    pub fn byte() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}byte"))
    }
    /// `xsd:unsignedLong`
    pub fn unsigned_long() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}unsignedLong"))
    }
    /// `xsd:unsignedInt`
    pub fn unsigned_int() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}unsignedInt"))
    }
    /// `xsd:nonNegativeInteger`
    pub fn non_negative_integer() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}nonNegativeInteger"))
    }
    /// `xsd:positiveInteger`
    pub fn positive_integer() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}positiveInteger"))
    }
    /// `xsd:anyURI`
    pub fn any_uri() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}anyURI"))
    }
    /// `xsd:base64Binary`
    pub fn base64_binary() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}base64Binary"))
    }
    /// `xsd:hexBinary`
    pub fn hex_binary() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}hexBinary"))
    }
    /// `xsd:gYear`
    pub fn g_year() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}gYear"))
    }
    /// `xsd:duration`
    pub fn duration() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}duration"))
    }
    /// `xsd:time`
    pub fn time() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}time"))
    }
    /// `xsd:normalizedString`
    pub fn normalized_string() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}normalizedString"))
    }
    /// `xsd:token`
    pub fn token() -> NamedNode {
        NamedNode::new_unchecked(format!("{XSD}token"))
    }
}

// ── Vocabulary constants ──────────────────────────────────────────────────────

/// Common RDF/RDFS/OWL/XSD vocabulary terms as [`NamedNode`] functions.
///
/// These mirror the JavaScript `rdf-ext` / `@rdfjs/namespace` pattern.
pub mod vocab {
    use crate::model::NamedNode;

    /// Core RDF vocabulary (`rdf:` prefix).
    pub mod rdf {
        use super::NamedNode;
        const NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
        /// `rdf:type`
        pub fn r#type() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}type"))
        }
        /// `rdf:subject`
        pub fn subject() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}subject"))
        }
        /// `rdf:predicate`
        pub fn predicate() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}predicate"))
        }
        /// `rdf:object`
        pub fn object() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}object"))
        }
        /// `rdf:Property`
        pub fn property() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Property"))
        }
        /// `rdf:Statement`
        pub fn statement() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Statement"))
        }
        /// `rdf:first`
        pub fn first() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}first"))
        }
        /// `rdf:rest`
        pub fn rest() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}rest"))
        }
        /// `rdf:nil`
        pub fn nil() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}nil"))
        }
        /// `rdf:List`
        pub fn list() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}List"))
        }
        /// `rdf:Bag`
        pub fn bag() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Bag"))
        }
        /// `rdf:Seq`
        pub fn seq() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Seq"))
        }
        /// `rdf:Alt`
        pub fn alt() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Alt"))
        }
        /// `rdf:value`
        pub fn value() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}value"))
        }
        /// `rdf:langString`
        pub fn lang_string() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}langString"))
        }
        /// RDF namespace IRI
        pub const NAMESPACE: &str = NS;
    }

    /// RDFS vocabulary (`rdfs:` prefix).
    pub mod rdfs {
        use super::NamedNode;
        const NS: &str = "http://www.w3.org/2000/01/rdf-schema#";
        /// `rdfs:label`
        pub fn label() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}label"))
        }
        /// `rdfs:comment`
        pub fn comment() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}comment"))
        }
        /// `rdfs:subClassOf`
        pub fn sub_class_of() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}subClassOf"))
        }
        /// `rdfs:subPropertyOf`
        pub fn sub_property_of() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}subPropertyOf"))
        }
        /// `rdfs:domain`
        pub fn domain() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}domain"))
        }
        /// `rdfs:range`
        pub fn range() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}range"))
        }
        /// `rdfs:Class`
        pub fn class() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Class"))
        }
        /// `rdfs:Resource`
        pub fn resource() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Resource"))
        }
        /// `rdfs:Literal`
        pub fn literal() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Literal"))
        }
        /// `rdfs:Datatype`
        pub fn datatype() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Datatype"))
        }
        /// `rdfs:isDefinedBy`
        pub fn is_defined_by() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}isDefinedBy"))
        }
        /// `rdfs:seeAlso`
        pub fn see_also() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}seeAlso"))
        }
        /// `rdfs:member`
        pub fn member() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}member"))
        }
        /// `rdfs:Container`
        pub fn container() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Container"))
        }
        /// RDFS namespace IRI
        pub const NAMESPACE: &str = NS;
    }

    /// OWL vocabulary (`owl:` prefix).
    pub mod owl {
        use super::NamedNode;
        const NS: &str = "http://www.w3.org/2002/07/owl#";
        /// `owl:Class`
        pub fn class() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Class"))
        }
        /// `owl:ObjectProperty`
        pub fn object_property() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}ObjectProperty"))
        }
        /// `owl:DatatypeProperty`
        pub fn datatype_property() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}DatatypeProperty"))
        }
        /// `owl:AnnotationProperty`
        pub fn annotation_property() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}AnnotationProperty"))
        }
        /// `owl:Thing`
        pub fn thing() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Thing"))
        }
        /// `owl:Nothing`
        pub fn nothing() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Nothing"))
        }
        /// `owl:sameAs`
        pub fn same_as() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}sameAs"))
        }
        /// `owl:equivalentClass`
        pub fn equivalent_class() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}equivalentClass"))
        }
        /// `owl:equivalentProperty`
        pub fn equivalent_property() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}equivalentProperty"))
        }
        /// `owl:inverseOf`
        pub fn inverse_of() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}inverseOf"))
        }
        /// `owl:disjointWith`
        pub fn disjoint_with() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}disjointWith"))
        }
        /// `owl:FunctionalProperty`
        pub fn functional_property() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}FunctionalProperty"))
        }
        /// `owl:Ontology`
        pub fn ontology() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Ontology"))
        }
        /// OWL namespace IRI
        pub const NAMESPACE: &str = NS;
    }

    /// XSD vocabulary (as named nodes, complements [`crate::data_factory::xsd_types`]).
    pub mod xsd {
        use super::NamedNode;
        const NS: &str = "http://www.w3.org/2001/XMLSchema#";
        /// `xsd:string`
        pub fn string() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}string"))
        }
        /// `xsd:integer`
        pub fn integer() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}integer"))
        }
        /// `xsd:boolean`
        pub fn boolean() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}boolean"))
        }
        /// `xsd:double`
        pub fn double() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}double"))
        }
        /// `xsd:dateTime`
        pub fn date_time() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}dateTime"))
        }
        /// XSD namespace IRI
        pub const NAMESPACE: &str = NS;
    }

    /// FOAF vocabulary (`foaf:` prefix) – commonly used in examples.
    pub mod foaf {
        use super::NamedNode;
        const NS: &str = "http://xmlns.com/foaf/0.1/";
        /// `foaf:name`
        pub fn name() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}name"))
        }
        /// `foaf:Person`
        pub fn person() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}Person"))
        }
        /// `foaf:knows`
        pub fn knows() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}knows"))
        }
        /// `foaf:mbox`
        pub fn mbox() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}mbox"))
        }
        /// `foaf:homepage`
        pub fn homepage() -> NamedNode {
            NamedNode::new_unchecked(format!("{NS}homepage"))
        }
        /// FOAF namespace IRI
        pub const NAMESPACE: &str = NS;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Named node ────────────────────────────────────────────────────────────

    #[test]
    fn test_named_node_valid_http() {
        let n = DataFactory::named_node("http://example.org/s").unwrap();
        assert_eq!(n.as_str(), "http://example.org/s");
    }

    #[test]
    fn test_named_node_valid_https() {
        let n = DataFactory::named_node("https://schema.org/Person").unwrap();
        assert_eq!(n.as_str(), "https://schema.org/Person");
    }

    #[test]
    fn test_named_node_valid_urn() {
        let n = DataFactory::named_node("urn:example:foo").unwrap();
        assert_eq!(n.as_str(), "urn:example:foo");
    }

    #[test]
    fn test_named_node_invalid_returns_err() {
        assert!(DataFactory::named_node("not an IRI").is_err());
    }

    #[test]
    fn test_named_node_empty_string_is_err() {
        assert!(DataFactory::named_node("").is_err());
    }

    #[test]
    fn test_named_node_with_fragment() {
        let n = DataFactory::named_node("http://example.org/ont#Class").unwrap();
        assert!(n.as_str().ends_with("#Class"));
    }

    #[test]
    fn test_named_node_with_query() {
        let n = DataFactory::named_node("http://example.org/q?a=1").unwrap();
        assert!(n.as_str().contains("a=1"));
    }

    // ── Blank nodes ───────────────────────────────────────────────────────────

    #[test]
    fn test_blank_node_auto_id_is_nonempty() {
        let b = DataFactory::blank_node();
        assert!(!b.as_str().is_empty());
    }

    #[test]
    fn test_blank_node_auto_ids_are_unique() {
        let b1 = DataFactory::blank_node();
        let b2 = DataFactory::blank_node();
        assert_ne!(b1.as_str(), b2.as_str());
    }

    #[test]
    fn test_blank_node_with_id() {
        let b = DataFactory::blank_node_with_id("my-node");
        assert_eq!(b.as_str(), "my-node");
    }

    #[test]
    fn test_blank_node_with_id_alpha() {
        let b = DataFactory::blank_node_with_id("abc");
        assert_eq!(b.as_str(), "abc");
    }

    #[test]
    fn test_blank_node_with_id_alphanumeric() {
        let b = DataFactory::blank_node_with_id("node1");
        assert_eq!(b.as_str(), "node1");
    }

    #[test]
    fn test_blank_node_with_invalid_id_still_returns_node() {
        // Invalid IDs fall back to a generated one (no panic)
        let b = DataFactory::blank_node_with_id("  spaces  ");
        assert!(!b.as_str().is_empty());
    }

    // ── Plain literals ────────────────────────────────────────────────────────

    #[test]
    fn test_literal_plain_value() {
        let l = DataFactory::literal("hello");
        assert_eq!(l.value(), "hello");
    }

    #[test]
    fn test_literal_plain_no_language() {
        let l = DataFactory::literal("hello");
        assert_eq!(l.language(), None);
    }

    #[test]
    fn test_literal_plain_datatype_is_xsd_string() {
        let l = DataFactory::literal("hello");
        assert!(l.datatype().as_str().contains("string"));
    }

    #[test]
    fn test_literal_empty_string() {
        let l = DataFactory::literal("");
        assert_eq!(l.value(), "");
    }

    // ── Typed literals ────────────────────────────────────────────────────────

    #[test]
    fn test_typed_literal_integer() {
        let l = DataFactory::typed_literal("42", xsd_types::integer());
        assert_eq!(l.value(), "42");
        assert!(l.datatype().as_str().ends_with("integer"));
    }

    #[test]
    fn test_typed_literal_boolean() {
        let l = DataFactory::typed_literal("true", xsd_types::boolean());
        assert_eq!(l.value(), "true");
        assert!(l.datatype().as_str().ends_with("boolean"));
    }

    #[test]
    fn test_typed_literal_double() {
        let l = DataFactory::typed_literal("3.14", xsd_types::double());
        assert!(l.datatype().as_str().ends_with("double"));
    }

    #[test]
    fn test_typed_literal_date_time() {
        let l = DataFactory::typed_literal("2026-02-24T00:00:00Z", xsd_types::date_time());
        assert!(l.datatype().as_str().ends_with("dateTime"));
    }

    #[test]
    fn test_typed_literal_custom_datatype() {
        let dt = DataFactory::named_node("http://example.org/myType").unwrap();
        let l = DataFactory::typed_literal("custom", dt);
        assert!(l.datatype().as_str().contains("myType"));
    }

    // ── Language literals ─────────────────────────────────────────────────────

    #[test]
    fn test_language_literal_value_and_lang() {
        let l = DataFactory::language_literal("Bonjour", "fr").unwrap();
        assert_eq!(l.value(), "Bonjour");
        assert_eq!(l.language(), Some("fr"));
    }

    #[test]
    fn test_language_literal_en() {
        let l = DataFactory::language_literal("Hello", "en").unwrap();
        assert_eq!(l.language(), Some("en"));
    }

    #[test]
    fn test_language_literal_zh_hans() {
        let l = DataFactory::language_literal("你好", "zh-Hans").unwrap();
        assert_eq!(l.language(), Some("zh-hans"));
    }

    #[test]
    fn test_language_literal_en_us() {
        let l = DataFactory::language_literal("Color", "en-US").unwrap();
        assert_eq!(l.language(), Some("en-us"));
    }

    #[test]
    fn test_language_literal_empty_lang_is_err() {
        assert!(DataFactory::language_literal("hello", "").is_err());
    }

    #[test]
    fn test_language_literal_invalid_lang_is_err() {
        // Contains space → invalid
        assert!(DataFactory::language_literal("hello", "en US").is_err());
    }

    // ── Triples ───────────────────────────────────────────────────────────────

    #[test]
    fn test_triple_subject_predicate_object() {
        let s = DataFactory::named_node("http://example.org/s").unwrap();
        let p = DataFactory::named_node("http://example.org/p").unwrap();
        let o = DataFactory::named_node("http://example.org/o").unwrap();
        let t = DataFactory::triple(s.into(), p.clone(), o.into());
        // Access via Display
        let text = format!("{t}");
        assert!(text.contains("http://example.org/s"));
    }

    #[test]
    fn test_triple_with_literal_object() {
        let s = DataFactory::named_node("http://example.org/s").unwrap();
        let p = DataFactory::named_node("http://example.org/p").unwrap();
        let o: Object = DataFactory::literal("hello").into();
        let t = DataFactory::triple(s.into(), p, o);
        let text = format!("{t}");
        assert!(text.contains("hello"));
    }

    #[test]
    fn test_triple_with_blank_node_subject() {
        let s: Subject = DataFactory::blank_node().into();
        let p = DataFactory::named_node("http://example.org/p").unwrap();
        let o: Object = DataFactory::literal("val").into();
        let _t = DataFactory::triple(s, p, o);
    }

    // ── Quads ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_quad_default_graph() {
        let s = DataFactory::named_node("http://example.org/s").unwrap();
        let p = DataFactory::named_node("http://example.org/p").unwrap();
        let o = DataFactory::named_node("http://example.org/o").unwrap();
        let g = DataFactory::default_graph();
        let q = DataFactory::quad(s.into(), p, o.into(), g);
        let text = format!("{q}");
        assert!(text.contains("http://example.org/s"));
    }

    #[test]
    fn test_quad_named_graph() {
        let s = DataFactory::named_node("http://example.org/s").unwrap();
        let p = DataFactory::named_node("http://example.org/p").unwrap();
        let o = DataFactory::named_node("http://example.org/o").unwrap();
        let g = DataFactory::named_graph("http://example.org/graph1").unwrap();
        let q = DataFactory::quad(s.into(), p, o.into(), g);
        let text = format!("{q}");
        assert!(text.contains("graph1"));
    }

    #[test]
    fn test_quad_named_graph_invalid_iri_is_err() {
        assert!(DataFactory::named_graph("not an IRI").is_err());
    }

    #[test]
    fn test_default_graph_is_default_graph_variant() {
        let g = DataFactory::default_graph();
        assert!(matches!(g, GraphName::DefaultGraph));
    }

    // ── IRI validation ────────────────────────────────────────────────────────

    #[test]
    fn test_validate_iri_http_ok() {
        assert!(DataFactory::validate_iri("http://example.org/").is_ok());
    }

    #[test]
    fn test_validate_iri_https_ok() {
        assert!(DataFactory::validate_iri("https://example.org/path").is_ok());
    }

    #[test]
    fn test_validate_iri_urn_ok() {
        assert!(DataFactory::validate_iri("urn:isbn:0451450523").is_ok());
    }

    #[test]
    fn test_validate_iri_bare_word_is_err() {
        assert!(DataFactory::validate_iri("hello").is_err());
    }

    #[test]
    fn test_validate_iri_empty_is_err() {
        assert!(DataFactory::validate_iri("").is_err());
    }

    #[test]
    fn test_validate_iri_space_is_err() {
        assert!(DataFactory::validate_iri("http://example.org/hello world").is_err());
    }

    // ── Language tag validation ───────────────────────────────────────────────

    #[test]
    fn test_validate_lang_tag_en_ok() {
        assert!(DataFactory::validate_lang_tag("en").is_ok());
    }

    #[test]
    fn test_validate_lang_tag_en_us_ok() {
        assert!(DataFactory::validate_lang_tag("en-US").is_ok());
    }

    #[test]
    fn test_validate_lang_tag_zh_hans_cn_ok() {
        assert!(DataFactory::validate_lang_tag("zh-Hans-CN").is_ok());
    }

    #[test]
    fn test_validate_lang_tag_empty_is_err() {
        assert!(DataFactory::validate_lang_tag("").is_err());
    }

    #[test]
    fn test_validate_lang_tag_space_is_err() {
        assert!(DataFactory::validate_lang_tag("en US").is_err());
    }

    #[test]
    fn test_validate_lang_tag_double_dash_is_err() {
        assert!(DataFactory::validate_lang_tag("en--US").is_err());
    }

    #[test]
    fn test_validate_lang_tag_numeric_primary_is_err() {
        // primary subtag must be alphabetic
        assert!(DataFactory::validate_lang_tag("123").is_err());
    }

    // ── XSD datatype helpers ──────────────────────────────────────────────────

    #[test]
    fn test_xsd_string_iri() {
        assert_eq!(
            xsd_types::string().as_str(),
            "http://www.w3.org/2001/XMLSchema#string"
        );
    }

    #[test]
    fn test_xsd_integer_iri() {
        assert_eq!(
            xsd_types::integer().as_str(),
            "http://www.w3.org/2001/XMLSchema#integer"
        );
    }

    #[test]
    fn test_xsd_float_iri() {
        assert!(xsd_types::float().as_str().ends_with("float"));
    }

    #[test]
    fn test_xsd_double_iri() {
        assert!(xsd_types::double().as_str().ends_with("double"));
    }

    #[test]
    fn test_xsd_boolean_iri() {
        assert!(xsd_types::boolean().as_str().ends_with("boolean"));
    }

    #[test]
    fn test_xsd_date_time_iri() {
        assert!(xsd_types::date_time().as_str().ends_with("dateTime"));
    }

    #[test]
    fn test_xsd_date_iri() {
        assert!(xsd_types::date().as_str().ends_with("date"));
    }

    #[test]
    fn test_xsd_decimal_iri() {
        assert!(xsd_types::decimal().as_str().ends_with("decimal"));
    }

    #[test]
    fn test_xsd_long_iri() {
        assert!(xsd_types::long().as_str().ends_with("long"));
    }

    #[test]
    fn test_xsd_int_iri() {
        assert!(xsd_types::int().as_str().ends_with("#int"));
    }

    #[test]
    fn test_xsd_any_uri_iri() {
        assert!(xsd_types::any_uri().as_str().ends_with("anyURI"));
    }

    #[test]
    fn test_xsd_base64_binary_iri() {
        assert!(xsd_types::base64_binary()
            .as_str()
            .ends_with("base64Binary"));
    }

    #[test]
    fn test_xsd_hex_binary_iri() {
        assert!(xsd_types::hex_binary().as_str().ends_with("hexBinary"));
    }

    // ── Vocabulary constants ──────────────────────────────────────────────────

    #[test]
    fn test_vocab_rdf_type() {
        assert_eq!(
            vocab::rdf::r#type().as_str(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        );
    }

    #[test]
    fn test_vocab_rdf_first_last() {
        assert!(vocab::rdf::first().as_str().ends_with("first"));
        assert!(vocab::rdf::rest().as_str().ends_with("rest"));
        assert!(vocab::rdf::nil().as_str().ends_with("nil"));
    }

    #[test]
    fn test_vocab_rdfs_label() {
        assert_eq!(
            vocab::rdfs::label().as_str(),
            "http://www.w3.org/2000/01/rdf-schema#label"
        );
    }

    #[test]
    fn test_vocab_rdfs_comment() {
        assert!(vocab::rdfs::comment().as_str().ends_with("comment"));
    }

    #[test]
    fn test_vocab_rdfs_sub_class_of() {
        assert!(vocab::rdfs::sub_class_of().as_str().ends_with("subClassOf"));
    }

    #[test]
    fn test_vocab_rdfs_domain_range() {
        assert!(vocab::rdfs::domain().as_str().ends_with("domain"));
        assert!(vocab::rdfs::range().as_str().ends_with("range"));
    }

    #[test]
    fn test_vocab_owl_class() {
        assert_eq!(
            vocab::owl::class().as_str(),
            "http://www.w3.org/2002/07/owl#Class"
        );
    }

    #[test]
    fn test_vocab_owl_same_as() {
        assert!(vocab::owl::same_as().as_str().ends_with("sameAs"));
    }

    #[test]
    fn test_vocab_owl_thing_nothing() {
        assert!(vocab::owl::thing().as_str().ends_with("Thing"));
        assert!(vocab::owl::nothing().as_str().ends_with("Nothing"));
    }

    #[test]
    fn test_vocab_owl_object_property() {
        assert!(vocab::owl::object_property()
            .as_str()
            .ends_with("ObjectProperty"));
    }

    #[test]
    fn test_vocab_xsd_string() {
        assert!(vocab::xsd::string().as_str().ends_with("string"));
    }

    #[test]
    fn test_vocab_foaf_name() {
        assert_eq!(
            vocab::foaf::name().as_str(),
            "http://xmlns.com/foaf/0.1/name"
        );
    }

    #[test]
    fn test_vocab_foaf_person() {
        assert!(vocab::foaf::person().as_str().ends_with("Person"));
    }

    #[test]
    fn test_vocab_foaf_knows() {
        assert!(vocab::foaf::knows().as_str().ends_with("knows"));
    }

    // ── Round-trip tests ──────────────────────────────────────────────────────

    #[test]
    fn test_roundtrip_named_node_via_string() {
        let iri = "http://example.org/roundtrip";
        let n = DataFactory::named_node(iri).unwrap();
        let s = n.as_str().to_string();
        let n2 = DataFactory::named_node(s).unwrap();
        assert_eq!(n, n2);
    }

    #[test]
    fn test_roundtrip_typed_literal() {
        let l = DataFactory::typed_literal("123", xsd_types::integer());
        let val = l.value().to_string();
        let dt = l.datatype().into_owned();
        let l2 = DataFactory::typed_literal(val, dt);
        assert_eq!(l.value(), l2.value());
    }

    #[test]
    fn test_roundtrip_language_literal() {
        let l = DataFactory::language_literal("Hola", "es").unwrap();
        let val = l.value().to_string();
        let lang = l.language().unwrap().to_string();
        let l2 = DataFactory::language_literal(val, lang).unwrap();
        assert_eq!(l.value(), l2.value());
        assert_eq!(l.language(), l2.language());
    }

    #[test]
    fn test_roundtrip_blank_node_with_id() {
        let b = DataFactory::blank_node_with_id("stable");
        let id = b.as_str().to_string();
        let b2 = DataFactory::blank_node_with_id(id.clone());
        assert_eq!(b2.as_str(), id);
    }

    #[test]
    fn test_quad_default_graph_roundtrip() {
        let s = DataFactory::named_node("http://example.org/s").unwrap();
        let p = vocab::rdf::r#type();
        let o = vocab::owl::class();
        let g = DataFactory::default_graph();
        let q = DataFactory::quad(s.into(), p, o.into(), g.clone());
        // graph name is default
        assert!(matches!(q.graph_name(), GraphName::DefaultGraph));
    }

    #[test]
    fn test_namespace_constants() {
        assert!(vocab::rdf::NAMESPACE.starts_with("http://"));
        assert!(vocab::rdfs::NAMESPACE.starts_with("http://"));
        assert!(vocab::owl::NAMESPACE.starts_with("http://"));
        assert!(vocab::xsd::NAMESPACE.starts_with("http://"));
        assert!(vocab::foaf::NAMESPACE.starts_with("http://"));
    }

    #[test]
    fn test_xsd_types_all_in_xsd_namespace() {
        let checks = [
            xsd_types::string(),
            xsd_types::integer(),
            xsd_types::float(),
            xsd_types::double(),
            xsd_types::boolean(),
            xsd_types::date_time(),
            xsd_types::date(),
            xsd_types::decimal(),
            xsd_types::long(),
            xsd_types::int(),
            xsd_types::short(),
            xsd_types::byte(),
            xsd_types::unsigned_long(),
            xsd_types::unsigned_int(),
            xsd_types::non_negative_integer(),
            xsd_types::positive_integer(),
            xsd_types::any_uri(),
            xsd_types::base64_binary(),
            xsd_types::hex_binary(),
            xsd_types::g_year(),
            xsd_types::duration(),
            xsd_types::time(),
            xsd_types::normalized_string(),
            xsd_types::token(),
        ];
        for node in &checks {
            assert!(
                node.as_str()
                    .starts_with("http://www.w3.org/2001/XMLSchema#"),
                "Not in XSD namespace: {}",
                node.as_str()
            );
        }
    }

    #[test]
    fn test_multiple_blank_nodes_in_triple() {
        let s: Subject = DataFactory::blank_node().into();
        let p = vocab::rdf::r#type();
        let o: Object = DataFactory::blank_node().into();
        let _t = DataFactory::triple(s, p, o);
    }

    #[test]
    fn test_literal_with_unicode_value() {
        let l = DataFactory::literal("日本語テスト");
        assert_eq!(l.value(), "日本語テスト");
    }

    #[test]
    fn test_typed_literal_float() {
        let l = DataFactory::typed_literal("1.5", xsd_types::float());
        assert!(l.datatype().as_str().ends_with("float"));
    }

    #[test]
    fn test_typed_literal_decimal() {
        let l = DataFactory::typed_literal("9.99", xsd_types::decimal());
        assert!(l.datatype().as_str().ends_with("decimal"));
    }

    #[test]
    fn test_rdfs_all_vocabs_are_valid_iris() {
        let nodes = [
            vocab::rdfs::label(),
            vocab::rdfs::comment(),
            vocab::rdfs::sub_class_of(),
            vocab::rdfs::sub_property_of(),
            vocab::rdfs::domain(),
            vocab::rdfs::range(),
            vocab::rdfs::class(),
            vocab::rdfs::resource(),
            vocab::rdfs::literal(),
            vocab::rdfs::datatype(),
            vocab::rdfs::is_defined_by(),
            vocab::rdfs::see_also(),
            vocab::rdfs::member(),
            vocab::rdfs::container(),
        ];
        for n in &nodes {
            assert!(
                DataFactory::validate_iri(n.as_str()).is_ok(),
                "Invalid IRI for vocab node: {}",
                n.as_str()
            );
        }
    }

    #[test]
    fn test_rdf_all_vocabs_are_valid_iris() {
        let nodes = [
            vocab::rdf::r#type(),
            vocab::rdf::subject(),
            vocab::rdf::predicate(),
            vocab::rdf::object(),
            vocab::rdf::property(),
            vocab::rdf::statement(),
            vocab::rdf::first(),
            vocab::rdf::rest(),
            vocab::rdf::nil(),
            vocab::rdf::list(),
            vocab::rdf::bag(),
            vocab::rdf::seq(),
            vocab::rdf::alt(),
            vocab::rdf::value(),
            vocab::rdf::lang_string(),
        ];
        for n in &nodes {
            assert!(
                DataFactory::validate_iri(n.as_str()).is_ok(),
                "Invalid IRI: {}",
                n.as_str()
            );
        }
    }

    #[test]
    fn test_owl_all_vocabs_are_valid_iris() {
        let nodes = [
            vocab::owl::class(),
            vocab::owl::object_property(),
            vocab::owl::datatype_property(),
            vocab::owl::annotation_property(),
            vocab::owl::thing(),
            vocab::owl::nothing(),
            vocab::owl::same_as(),
            vocab::owl::equivalent_class(),
            vocab::owl::equivalent_property(),
            vocab::owl::inverse_of(),
            vocab::owl::disjoint_with(),
            vocab::owl::functional_property(),
            vocab::owl::ontology(),
        ];
        for n in &nodes {
            assert!(
                DataFactory::validate_iri(n.as_str()).is_ok(),
                "Invalid IRI: {}",
                n.as_str()
            );
        }
    }
}
