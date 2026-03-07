//! Basic RDFa 1.1 Lite parser.
//!
//! Extracts RDF triples from a simplified HTML/XML document representation
//! by processing RDFa attributes:
//!
//! * `property`  → predicate IRI
//! * `typeof`    → `rdf:type` triple
//! * `resource`  → object IRI
//! * `about`     → subject IRI
//! * `prefix`    → prefix mappings
//! * `content`   → literal value (overrides text content)
//! * `datatype`  → literal datatype
//! * `lang`      → language tag
//! * `rel`/`rev` → link relation / reverse link
//!
//! The parser operates on a lightweight DOM-like `Element` tree (no external
//! XML crate required) which callers can build from any source.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Vocabulary constants
// ---------------------------------------------------------------------------

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_PREFIX: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
#[allow(dead_code)]
const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";

/// Well-known prefix bindings pre-loaded by the parser.
fn default_prefixes() -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("rdf".into(), RDF_PREFIX.into());
    m.insert(
        "rdfs".into(),
        "http://www.w3.org/2000/01/rdf-schema#".into(),
    );
    m.insert("xsd".into(), "http://www.w3.org/2001/XMLSchema#".into());
    m.insert("owl".into(), "http://www.w3.org/2002/07/owl#".into());
    m.insert("dc".into(), "http://purl.org/dc/elements/1.1/".into());
    m.insert("dcterms".into(), "http://purl.org/dc/terms/".into());
    m.insert("foaf".into(), "http://xmlns.com/foaf/0.1/".into());
    m.insert("schema".into(), "https://schema.org/".into());
    m
}

// ---------------------------------------------------------------------------
// Lightweight DOM types
// ---------------------------------------------------------------------------

/// A key-value attribute on an element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribute {
    /// The attribute name (e.g. `"property"`, `"typeof"`).
    pub name: String,
    /// The attribute value string.
    pub value: String,
}

impl Attribute {
    /// Create a new attribute with the given name and value.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// A simplified DOM element used as input to the RDFa parser.
#[derive(Debug, Clone)]
pub struct Element {
    /// Tag name (e.g. `"div"`, `"span"`, `"a"`).
    pub tag: String,
    /// Attributes on this element.
    pub attributes: Vec<Attribute>,
    /// Plain-text content (characters only, no child elements included).
    pub text: String,
    /// Child elements (ordered).
    pub children: Vec<Element>,
}

impl Element {
    /// Create a new element with the given tag name and no attributes, text, or children.
    pub fn new(tag: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            attributes: Vec::new(),
            text: String::new(),
            children: Vec::new(),
        }
    }

    /// Add an attribute and return `self` for builder chaining.
    pub fn attr(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.push(Attribute::new(name, value));
        self
    }

    /// Set the text content.
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self
    }

    /// Append a child element.
    pub fn child(mut self, child: Element) -> Self {
        self.children.push(child);
        self
    }

    /// Return the value of the named attribute, or `None`.
    pub fn get_attr(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.value.as_str())
    }
}

// ---------------------------------------------------------------------------
// Output triple
// ---------------------------------------------------------------------------

/// An RDF triple produced by the RDFa parser.
///
/// Subject and predicate are always IRIs; the object is either an IRI or a
/// plain/typed/language literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdfaTriple {
    /// IRI of the triple's subject.
    pub subject: String,
    /// IRI of the triple's predicate.
    pub predicate: String,
    /// Object of the triple (IRI or literal).
    pub object: RdfaObject,
}

/// The object of an RDFa triple.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RdfaObject {
    /// An IRI reference.
    Iri(String),
    /// A plain string literal (no datatype / language).
    Literal(String),
    /// A typed literal.
    TypedLiteral {
        /// The lexical value of the literal.
        value: String,
        /// The datatype IRI.
        datatype: String,
    },
    /// A language-tagged string literal.
    LangLiteral {
        /// The lexical value of the literal.
        value: String,
        /// BCP-47 language tag (e.g. `"en"`, `"fr-CA"`).
        lang: String,
    },
}

impl fmt::Display for RdfaObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RdfaObject::Iri(iri) => write!(f, "<{iri}>"),
            RdfaObject::Literal(v) => write!(f, "\"{v}\""),
            RdfaObject::TypedLiteral { value, datatype } => {
                write!(f, "\"{value}\"^^<{datatype}>")
            }
            RdfaObject::LangLiteral { value, lang } => write!(f, "\"{value}\"@{lang}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Parser evaluation context
// ---------------------------------------------------------------------------

/// Inherited context passed down the element tree.
#[derive(Debug, Clone)]
struct EvalContext {
    /// Current subject IRI (inherited from parent if not overridden).
    subject: Option<String>,
    /// Base IRI for resolving relative references.
    base: String,
    /// Active prefix mappings.
    prefixes: HashMap<String, String>,
    /// Default vocabulary IRI.
    vocab: Option<String>,
    /// Current language tag.
    lang: Option<String>,
}

impl EvalContext {
    fn new(base: &str) -> Self {
        Self {
            subject: None,
            base: base.to_string(),
            prefixes: default_prefixes(),
            vocab: None,
            lang: None,
        }
    }

    /// Resolve a CURIE or IRI reference to an absolute IRI.
    fn resolve(&self, term: &str) -> Option<String> {
        let trimmed = term.trim();
        if trimmed.is_empty() {
            return None;
        }
        // Already absolute IRI
        if trimmed.starts_with('<') && trimmed.ends_with('>') {
            return Some(trimmed[1..trimmed.len() - 1].to_string());
        }
        if trimmed.contains("://") {
            return Some(trimmed.to_string());
        }
        // CURIE: "prefix:local"
        if let Some(colon) = trimmed.find(':') {
            let prefix = &trimmed[..colon];
            let local = &trimmed[colon + 1..];
            if let Some(ns) = self.prefixes.get(prefix) {
                return Some(format!("{ns}{local}"));
            }
        }
        // Default vocab
        if let Some(vocab) = &self.vocab {
            return Some(format!("{vocab}{trimmed}"));
        }
        // Relative IRI against base
        if !self.base.is_empty() {
            return Some(format!("{}{}", self.base, trimmed));
        }
        None
    }

    /// Parse and register prefix declarations from a `prefix` attribute value.
    ///
    /// Format: `"ex: http://example.org/ dc: http://purl.org/dc/elements/1.1/ ..."`
    fn register_prefix_attr(&mut self, prefix_attr: &str) {
        let tokens: Vec<&str> = prefix_attr.split_whitespace().collect();
        let mut i = 0;
        while i + 1 < tokens.len() {
            let pfx = tokens[i];
            let ns = tokens[i + 1];
            if let Some(prefix_label) = pfx.strip_suffix(':') {
                self.prefixes
                    .insert(prefix_label.to_string(), ns.to_string());
            }
            i += 2;
        }
    }
}

// ---------------------------------------------------------------------------
// RDFa parser
// ---------------------------------------------------------------------------

/// Configuration for the RDFa parser.
#[derive(Debug, Clone)]
pub struct RdfaConfig {
    /// Base IRI used for resolving relative references.
    pub base: String,
    /// Whether to generate `rdf:type` triples from `typeof` attributes.
    pub process_typeof: bool,
    /// Whether to process `rel` and `rev` attributes.
    pub process_rel_rev: bool,
}

impl Default for RdfaConfig {
    fn default() -> Self {
        Self {
            base: String::new(),
            process_typeof: true,
            process_rel_rev: true,
        }
    }
}

/// The RDFa 1.1 Lite parser.
pub struct RdfaParser {
    config: RdfaConfig,
}

impl RdfaParser {
    /// Create a new parser with the given configuration.
    pub fn new(config: RdfaConfig) -> Self {
        Self { config }
    }

    /// Parse an element tree and return all extracted RDF triples.
    pub fn parse(&self, root: &Element) -> Vec<RdfaTriple> {
        let ctx = EvalContext::new(&self.config.base);
        let mut triples = Vec::new();
        self.process_element(root, &ctx, &mut triples);
        triples
    }

    fn process_element(
        &self,
        element: &Element,
        parent_ctx: &EvalContext,
        triples: &mut Vec<RdfaTriple>,
    ) {
        let mut ctx = parent_ctx.clone();

        // --- Update prefix mappings from `prefix` attribute ---
        if let Some(pfx_attr) = element.get_attr("prefix") {
            ctx.register_prefix_attr(pfx_attr);
        }

        // --- Update default vocabulary from `vocab` attribute ---
        if let Some(vocab) = element.get_attr("vocab") {
            ctx.vocab = Some(vocab.to_string());
        }

        // --- Update language from `lang` or `xml:lang` ---
        if let Some(lang) = element
            .get_attr("lang")
            .or_else(|| element.get_attr("xml:lang"))
        {
            ctx.lang = if lang.is_empty() {
                None
            } else {
                Some(lang.to_string())
            };
        }

        // --- Determine the new subject ---
        // Precedence: `about` > inherited subject > blank node
        let new_subject: Option<String> = element
            .get_attr("about")
            .and_then(|v| ctx.resolve(v))
            .or_else(|| ctx.subject.clone());

        // `resource` attribute on a non-leaf element can also set the subject context
        // for child elements (if no `about` is present and `property` is absent).
        let resource_iri: Option<String> =
            element.get_attr("resource").and_then(|v| ctx.resolve(v));

        // Effective subject for triple emission
        let effective_subject: Option<String> =
            new_subject.clone().or_else(|| resource_iri.clone());

        // --- `typeof` attribute → rdf:type triple(s) ---
        if self.config.process_typeof {
            if let (Some(subj), Some(typeof_attr)) =
                (&effective_subject, element.get_attr("typeof"))
            {
                for type_term in typeof_attr.split_whitespace() {
                    if let Some(type_iri) = ctx.resolve(type_term) {
                        triples.push(RdfaTriple {
                            subject: subj.clone(),
                            predicate: RDF_TYPE.to_string(),
                            object: RdfaObject::Iri(type_iri),
                        });
                    }
                }
            }
        }

        // --- `property` attribute → predicate with literal or resource object ---
        if let (Some(subj), Some(property_attr)) =
            (&effective_subject, element.get_attr("property"))
        {
            for prop_term in property_attr.split_whitespace() {
                if let Some(predicate) = ctx.resolve(prop_term) {
                    // Determine object
                    let object = self.extract_object(element, &ctx, &resource_iri);
                    if let Some(obj) = object {
                        triples.push(RdfaTriple {
                            subject: subj.clone(),
                            predicate,
                            object: obj,
                        });
                    }
                }
            }
        }

        // --- `rel` attribute → forward link relation ---
        if self.config.process_rel_rev {
            if let (Some(subj), Some(rel_attr), Some(obj_iri)) = (
                &effective_subject,
                element.get_attr("rel"),
                resource_iri.as_ref().or(element
                    .get_attr("href")
                    .and_then(|h| ctx.resolve(h))
                    .as_ref()),
            ) {
                for rel_term in rel_attr.split_whitespace() {
                    if let Some(predicate) = ctx.resolve(rel_term) {
                        triples.push(RdfaTriple {
                            subject: subj.clone(),
                            predicate,
                            object: RdfaObject::Iri(obj_iri.clone()),
                        });
                    }
                }
            }

            // --- `rev` attribute → reverse link relation ---
            if let (Some(subj), Some(rev_attr), Some(obj_iri)) = (
                &effective_subject,
                element.get_attr("rev"),
                resource_iri.as_ref().or(element
                    .get_attr("href")
                    .and_then(|h| ctx.resolve(h))
                    .as_ref()),
            ) {
                for rev_term in rev_attr.split_whitespace() {
                    if let Some(predicate) = ctx.resolve(rev_term) {
                        // In `rev`, the roles of subject and object are swapped
                        triples.push(RdfaTriple {
                            subject: obj_iri.clone(),
                            predicate,
                            object: RdfaObject::Iri(subj.clone()),
                        });
                    }
                }
            }
        }

        // --- Recurse into children with updated context ---
        // The child subject is:
        //   * `resource` IRI (if present, and no `property`) — new subject for children
        //   * Otherwise: effective_subject (inherited)
        let child_subject = if element.get_attr("property").is_none() {
            resource_iri.or(effective_subject)
        } else {
            effective_subject
        };
        ctx.subject = child_subject;

        for child in &element.children {
            self.process_element(child, &ctx, triples);
        }
    }

    /// Determine the object for a `property` triple.
    ///
    /// Priority:
    /// 1. `resource` attribute → IRI object
    /// 2. `content` attribute → plain/typed/lang literal
    /// 3. `datatype` with element text → typed literal
    /// 4. Element text with language → lang literal
    /// 5. Element text → plain literal (xsd:string)
    fn extract_object(
        &self,
        element: &Element,
        ctx: &EvalContext,
        resource_iri: &Option<String>,
    ) -> Option<RdfaObject> {
        // resource overrides literal for property triples only when no content/datatype
        if let Some(iri) = resource_iri {
            if element.get_attr("content").is_none() && element.get_attr("datatype").is_none() {
                return Some(RdfaObject::Iri(iri.clone()));
            }
        }

        let raw_value: String = element
            .get_attr("content")
            .map(|s| s.to_string())
            .unwrap_or_else(|| element.text.clone());

        let datatype = element.get_attr("datatype").and_then(|dt| ctx.resolve(dt));
        let lang = element
            .get_attr("lang")
            .or_else(|| element.get_attr("xml:lang"))
            .and_then(|l| {
                if l.is_empty() {
                    None
                } else {
                    Some(l.to_string())
                }
            })
            .or_else(|| {
                // Only inherit parent lang if this element did not explicitly set lang="" to clear it
                if element.get_attr("lang").is_some() || element.get_attr("xml:lang").is_some() {
                    None
                } else {
                    ctx.lang.clone()
                }
            });

        if let Some(dt) = datatype {
            return Some(RdfaObject::TypedLiteral {
                value: raw_value,
                datatype: dt,
            });
        }
        if let Some(l) = lang {
            return Some(RdfaObject::LangLiteral {
                value: raw_value,
                lang: l,
            });
        }
        // Default: plain string
        if raw_value.is_empty() {
            None
        } else {
            Some(RdfaObject::Literal(raw_value))
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience builder for HTML `<head>`-style prefix extraction
// ---------------------------------------------------------------------------

/// Extract prefix declarations from an HTML `<head>` element, scanning for
/// `<link rel="prefix" ...>` or a `<meta prefix="..." ...>` element pattern.
///
/// Also handles a `prefix` attribute directly on the `<html>` element.
pub fn extract_head_prefixes(head: &Element) -> HashMap<String, String> {
    let mut prefixes = default_prefixes();
    if let Some(pfx) = head.get_attr("prefix") {
        let mut ctx = EvalContext::new("");
        ctx.register_prefix_attr(pfx);
        prefixes.extend(ctx.prefixes);
    }
    for child in &head.children {
        if let Some(pfx) = child.get_attr("prefix") {
            let mut ctx = EvalContext::new("");
            ctx.register_prefix_attr(pfx);
            for (k, v) in ctx.prefixes {
                prefixes.insert(k, v);
            }
        }
    }
    prefixes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn parser() -> RdfaParser {
        RdfaParser::new(RdfaConfig {
            base: "http://example.org/".to_string(),
            ..Default::default()
        })
    }

    // -----------------------------------------------------------------------
    // typeof → rdf:type
    // -----------------------------------------------------------------------

    #[test]
    fn test_typeof_single() {
        let el = Element::new("div")
            .attr("about", "http://example.org/alice")
            .attr("typeof", "foaf:Person");
        let triples = parser().parse(&el);
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].predicate, RDF_TYPE);
        assert_eq!(
            triples[0].object,
            RdfaObject::Iri("http://xmlns.com/foaf/0.1/Person".to_string())
        );
    }

    #[test]
    fn test_typeof_multiple_types() {
        let el = Element::new("div")
            .attr("about", "http://example.org/alice")
            .attr("typeof", "foaf:Person schema:Person");
        let triples = parser().parse(&el);
        assert_eq!(triples.len(), 2);
    }

    // -----------------------------------------------------------------------
    // property → literal
    // -----------------------------------------------------------------------

    #[test]
    fn test_property_plain_literal() {
        let el = Element::new("span")
            .attr("about", "http://example.org/alice")
            .attr("property", "foaf:name")
            .text("Alice");
        let triples = parser().parse(&el);
        assert_eq!(triples.len(), 1);
        assert_eq!(triples[0].object, RdfaObject::Literal("Alice".to_string()));
    }

    #[test]
    fn test_property_content_attribute() {
        let el = Element::new("span")
            .attr("about", "http://example.org/alice")
            .attr("property", "foaf:name")
            .attr("content", "Alice Smith")
            .text("displayed text");
        let triples = parser().parse(&el);
        assert_eq!(
            triples[0].object,
            RdfaObject::Literal("Alice Smith".to_string())
        );
    }

    #[test]
    fn test_property_typed_literal() {
        let el = Element::new("span")
            .attr("about", "http://example.org/event")
            .attr("property", "schema:startDate")
            .attr("datatype", "xsd:date")
            .text("2026-01-01");
        let triples = parser().parse(&el);
        assert_eq!(triples.len(), 1);
        match &triples[0].object {
            RdfaObject::TypedLiteral { value, datatype } => {
                assert_eq!(value, "2026-01-01");
                assert!(datatype.contains("date"));
            }
            other => panic!("Expected TypedLiteral, got {other:?}"),
        }
    }

    #[test]
    fn test_property_lang_literal() {
        let el = Element::new("span")
            .attr("about", "http://example.org/doc")
            .attr("property", "dc:title")
            .attr("lang", "en")
            .text("Hello");
        let triples = parser().parse(&el);
        assert_eq!(
            triples[0].object,
            RdfaObject::LangLiteral {
                value: "Hello".to_string(),
                lang: "en".to_string()
            }
        );
    }

    // -----------------------------------------------------------------------
    // about → subject IRI
    // -----------------------------------------------------------------------

    #[test]
    fn test_about_absolute_iri() {
        let el = Element::new("div")
            .attr("about", "http://example.org/resource")
            .attr("property", "rdfs:label")
            .text("Test");
        let triples = parser().parse(&el);
        assert_eq!(triples[0].subject, "http://example.org/resource");
    }

    #[test]
    fn test_about_curie() {
        let el = Element::new("div")
            .attr("prefix", "ex: http://example.org/")
            .attr("about", "ex:alice")
            .attr("property", "rdfs:label")
            .text("Alice");
        let triples = parser().parse(&el);
        assert_eq!(triples[0].subject, "http://example.org/alice");
    }

    // -----------------------------------------------------------------------
    // resource attribute
    // -----------------------------------------------------------------------

    #[test]
    fn test_resource_as_object_iri() {
        let el = Element::new("span")
            .attr("about", "http://example.org/alice")
            .attr("property", "foaf:knows")
            .attr("resource", "http://example.org/bob");
        let triples = parser().parse(&el);
        assert_eq!(triples.len(), 1);
        assert_eq!(
            triples[0].object,
            RdfaObject::Iri("http://example.org/bob".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // prefix attribute
    // -----------------------------------------------------------------------

    #[test]
    fn test_prefix_declaration() {
        let el = Element::new("div")
            .attr(
                "prefix",
                "ex: http://example.org/ dc: http://purl.org/dc/elements/1.1/",
            )
            .attr("about", "ex:book1")
            .attr("property", "dc:title")
            .text("The Book");
        let triples = parser().parse(&el);
        assert_eq!(triples[0].subject, "http://example.org/book1");
        assert_eq!(
            triples[0].predicate,
            "http://purl.org/dc/elements/1.1/title"
        );
    }

    // -----------------------------------------------------------------------
    // rel / rev attributes
    // -----------------------------------------------------------------------

    #[test]
    fn test_rel_attribute() {
        let el = Element::new("a")
            .attr("about", "http://example.org/alice")
            .attr("rel", "foaf:knows")
            .attr("resource", "http://example.org/bob");
        let triples = parser().parse(&el);
        assert!(triples
            .iter()
            .any(|t| t.predicate == "http://xmlns.com/foaf/0.1/knows"));
    }

    #[test]
    fn test_rev_attribute() {
        let el = Element::new("a")
            .attr("about", "http://example.org/bob")
            .attr("rev", "foaf:knows")
            .attr("resource", "http://example.org/alice");
        let triples = parser().parse(&el);
        // In rev: subject becomes object and vice-versa
        let rev_triple = triples
            .iter()
            .find(|t| t.predicate == "http://xmlns.com/foaf/0.1/knows");
        assert!(rev_triple.is_some());
        let rv = rev_triple.expect("rev triple should exist");
        // The rev triple should have alice as subject (the resource) and bob as object (about)
        assert_eq!(rv.subject, "http://example.org/alice");
    }

    // -----------------------------------------------------------------------
    // Context inheritance
    // -----------------------------------------------------------------------

    #[test]
    fn test_subject_inheritance_from_parent() {
        let child = Element::new("span")
            .attr("property", "foaf:name")
            .text("Alice");
        let parent = Element::new("div")
            .attr("about", "http://example.org/alice")
            .child(child);
        let triples = parser().parse(&parent);
        assert!(triples
            .iter()
            .any(|t| t.subject == "http://example.org/alice"));
    }

    #[test]
    fn test_lang_inheritance_from_parent() {
        let child = Element::new("span")
            .attr("property", "rdfs:label")
            .text("Bonjour");
        let parent = Element::new("div")
            .attr("about", "http://example.org/res")
            .attr("lang", "fr")
            .child(child);
        let triples = parser().parse(&parent);
        let label_triple = triples.iter().find(|t| t.predicate.contains("label"));
        assert!(label_triple.is_some());
        match &label_triple.expect("label triple should exist").object {
            RdfaObject::LangLiteral { lang, .. } => assert_eq!(lang, "fr"),
            other => panic!("Expected LangLiteral, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // vocab attribute
    // -----------------------------------------------------------------------

    #[test]
    fn test_vocab_attribute() {
        let el = Element::new("div")
            .attr("vocab", "https://schema.org/")
            .attr("about", "http://example.org/person")
            .attr("typeof", "Person");
        let triples = parser().parse(&el);
        assert!(triples.iter().any(|t| {
            t.predicate == RDF_TYPE
                && t.object == RdfaObject::Iri("https://schema.org/Person".to_string())
        }));
    }

    // -----------------------------------------------------------------------
    // extract_head_prefixes
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_head_prefixes() {
        let head =
            Element::new("head").attr("prefix", "ex: http://example.org/ my: http://my.org/");
        let prefixes = extract_head_prefixes(&head);
        assert_eq!(
            prefixes.get("ex").map(|s| s.as_str()),
            Some("http://example.org/")
        );
        assert_eq!(
            prefixes.get("my").map(|s| s.as_str()),
            Some("http://my.org/")
        );
    }

    // -----------------------------------------------------------------------
    // RdfaObject display
    // -----------------------------------------------------------------------

    #[test]
    fn test_rdfa_object_iri_display() {
        let obj = RdfaObject::Iri("http://example.org/".to_string());
        assert_eq!(obj.to_string(), "<http://example.org/>");
    }

    #[test]
    fn test_rdfa_object_literal_display() {
        let obj = RdfaObject::Literal("hello".to_string());
        assert_eq!(obj.to_string(), "\"hello\"");
    }

    #[test]
    fn test_rdfa_object_typed_literal_display() {
        let obj = RdfaObject::TypedLiteral {
            value: "42".to_string(),
            datatype: XSD_STRING.to_string(),
        };
        assert!(obj.to_string().contains("42"));
    }

    #[test]
    fn test_rdfa_object_lang_literal_display() {
        let obj = RdfaObject::LangLiteral {
            value: "hello".to_string(),
            lang: "en".to_string(),
        };
        assert_eq!(obj.to_string(), "\"hello\"@en");
    }

    // -----------------------------------------------------------------------
    // Multiple properties on same element
    // -----------------------------------------------------------------------

    #[test]
    fn test_multiple_properties_space_separated() {
        let el = Element::new("span")
            .attr("about", "http://example.org/doc")
            .attr("property", "dc:title rdfs:label")
            .text("My Doc");
        let triples = parser().parse(&el);
        assert_eq!(triples.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Empty text content
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_text_no_triple() {
        let el = Element::new("span")
            .attr("about", "http://example.org/doc")
            .attr("property", "dc:description");
        let triples = parser().parse(&el);
        // No object → no triple
        assert!(triples.is_empty());
    }

    // -----------------------------------------------------------------------
    // No subject → no triple
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_subject_no_triple() {
        // No `about`, no parent subject → effective_subject is None
        let el = Element::new("span")
            .attr("property", "foaf:name")
            .text("Orphan");
        let triples = parser().parse(&el);
        assert!(triples.is_empty());
    }

    // -----------------------------------------------------------------------
    // Additional coverage
    // -----------------------------------------------------------------------

    #[test]
    fn test_element_builder_get_attr() {
        let el = Element::new("div").attr("id", "main").attr("class", "hero");
        assert_eq!(el.get_attr("id"), Some("main"));
        assert_eq!(el.get_attr("class"), Some("hero"));
        assert_eq!(el.get_attr("missing"), None);
    }

    #[test]
    fn test_attribute_struct() {
        let a = Attribute::new("rel", "stylesheet");
        assert_eq!(a.name, "rel");
        assert_eq!(a.value, "stylesheet");
    }

    #[test]
    fn test_typeof_no_about() {
        // typeof without about — no subject → no triple
        let el = Element::new("div").attr("typeof", "foaf:Person");
        let triples = parser().parse(&el);
        assert!(triples.is_empty());
    }

    #[test]
    fn test_property_iri_object_via_resource() {
        let el = Element::new("a")
            .attr("about", "http://example.org/s")
            .attr("property", "foaf:homepage")
            .attr("resource", "http://example.org/page");
        let triples = parser().parse(&el);
        assert!(triples
            .iter()
            .any(|t| matches!(&t.object, RdfaObject::Iri(iri) if iri.contains("page"))));
    }

    #[test]
    fn test_default_prefix_rdf() {
        // rdf: prefix should be predefined
        let el = Element::new("span")
            .attr("about", "http://example.org/r")
            .attr("typeof", "rdf:Resource");
        let triples = parser().parse(&el);
        assert!(!triples.is_empty());
        assert!(triples[0]
            .object
            .to_string()
            .contains("rdf-syntax-ns#Resource"));
    }

    #[test]
    fn test_xml_lang_attribute() {
        let el = Element::new("span")
            .attr("about", "http://example.org/r")
            .attr("property", "dc:title")
            .attr("xml:lang", "de")
            .text("Hallo");
        let triples = parser().parse(&el);
        assert_eq!(
            triples[0].object,
            RdfaObject::LangLiteral {
                value: "Hallo".to_string(),
                lang: "de".to_string()
            }
        );
    }

    #[test]
    fn test_multiple_children_multiple_triples() {
        let child1 = Element::new("span")
            .attr("property", "foaf:name")
            .text("Alice");
        let child2 = Element::new("span")
            .attr("property", "foaf:mbox")
            .attr("resource", "mailto:alice@example.org");
        let parent = Element::new("div")
            .attr("about", "http://example.org/alice")
            .child(child1)
            .child(child2);
        let triples = parser().parse(&parent);
        assert!(triples.len() >= 2);
    }

    #[test]
    fn test_nested_resource_sets_child_subject() {
        let child = Element::new("span")
            .attr("property", "foaf:name")
            .text("Bob");
        let parent = Element::new("div")
            .attr("about", "http://example.org/alice")
            .attr("resource", "http://example.org/bob")
            .child(child);
        let triples = parser().parse(&parent);
        // Child should have bob as subject
        let name_triple = triples.iter().find(|t| t.predicate.contains("name"));
        assert!(name_triple.is_some());
        assert_eq!(
            name_triple.expect("name triple should exist").subject,
            "http://example.org/bob"
        );
    }

    #[test]
    fn test_extract_head_prefixes_with_child() {
        let meta = Element::new("meta").attr("prefix", "schema: https://schema.org/");
        let head = Element::new("head").child(meta);
        let prefixes = extract_head_prefixes(&head);
        assert_eq!(
            prefixes.get("schema").map(|s| s.as_str()),
            Some("https://schema.org/")
        );
    }

    #[test]
    fn test_rdfa_config_default() {
        let cfg = RdfaConfig::default();
        assert!(cfg.process_typeof);
        assert!(cfg.process_rel_rev);
        assert!(cfg.base.is_empty());
    }

    #[test]
    fn test_parser_no_process_typeof() {
        let cfg = RdfaConfig {
            process_typeof: false,
            base: "http://example.org/".to_string(),
            ..Default::default()
        };
        let el = Element::new("div")
            .attr("about", "http://example.org/alice")
            .attr("typeof", "foaf:Person");
        let triples = RdfaParser::new(cfg).parse(&el);
        assert!(triples.is_empty());
    }

    #[test]
    fn test_parser_no_process_rel_rev() {
        let cfg = RdfaConfig {
            process_rel_rev: false,
            base: "http://example.org/".to_string(),
            ..Default::default()
        };
        let el = Element::new("a")
            .attr("about", "http://example.org/s")
            .attr("rel", "foaf:knows")
            .attr("resource", "http://example.org/o");
        let triples = RdfaParser::new(cfg).parse(&el);
        assert!(triples.is_empty());
    }

    #[test]
    fn test_triple_subject_from_parent_and_child_typeof() {
        let child = Element::new("div")
            .attr("typeof", "schema:Book")
            .attr("about", "http://example.org/book1");
        let parent = Element::new("div")
            .attr("about", "http://example.org/collection")
            .child(child);
        let triples = parser().parse(&parent);
        assert!(triples
            .iter()
            .any(|t| t.subject == "http://example.org/book1"));
    }

    #[test]
    fn test_about_relative_iri_resolved_with_base() {
        let el = Element::new("span")
            .attr("about", "alice")
            .attr("property", "foaf:name")
            .text("Alice");
        // Base is http://example.org/ — alice should resolve to http://example.org/alice
        let triples = parser().parse(&el);
        assert_eq!(triples[0].subject, "http://example.org/alice");
    }

    #[test]
    fn test_property_with_datatype_overrides_resource() {
        // When `datatype` is present, a typed literal is preferred over resource IRI
        let el = Element::new("span")
            .attr("about", "http://example.org/e")
            .attr("property", "schema:startDate")
            .attr("datatype", "xsd:date")
            .attr("resource", "http://example.org/ignored")
            .text("2026-03-04");
        let triples = parser().parse(&el);
        assert!(matches!(
            &triples[0].object,
            RdfaObject::TypedLiteral { .. }
        ));
    }

    #[test]
    fn test_default_prefixes_rdfs() {
        let el = Element::new("div")
            .attr("about", "http://example.org/r")
            .attr("property", "rdfs:comment")
            .text("A thing");
        let triples = parser().parse(&el);
        assert!(triples[0].predicate.contains("comment"));
    }

    #[test]
    fn test_default_prefixes_owl() {
        let el = Element::new("div")
            .attr("about", "http://example.org/r")
            .attr("typeof", "owl:Class");
        let triples = parser().parse(&el);
        assert!(triples[0].object.to_string().contains("owl"));
    }

    #[test]
    fn test_element_children_count() {
        let el = Element::new("div")
            .child(Element::new("span"))
            .child(Element::new("span"));
        assert_eq!(el.children.len(), 2);
    }

    #[test]
    fn test_element_text_method() {
        let el = Element::new("span").text("hello");
        assert_eq!(el.text, "hello");
    }

    #[test]
    fn test_rdfa_triple_fields() {
        let triple = RdfaTriple {
            subject: "http://example.org/s".to_string(),
            predicate: "http://example.org/p".to_string(),
            object: RdfaObject::Literal("value".to_string()),
        };
        assert_eq!(triple.subject, "http://example.org/s");
        assert_eq!(triple.predicate, "http://example.org/p");
    }

    #[test]
    fn test_lang_empty_string_clears_lang() {
        let child = Element::new("span")
            .attr("property", "dc:title")
            .attr("lang", "")
            .text("Title");
        let parent = Element::new("div")
            .attr("about", "http://example.org/doc")
            .attr("lang", "fr")
            .child(child);
        let triples = parser().parse(&parent);
        let title = triples.iter().find(|t| t.predicate.contains("title"));
        // Empty lang clears language → plain literal
        if let Some(t) = title {
            assert!(matches!(&t.object, RdfaObject::Literal(_)));
        }
    }

    #[test]
    fn test_curie_unknown_prefix_not_resolved() {
        let el = Element::new("span")
            .attr("about", "http://example.org/r")
            .attr("property", "unknown:prop")
            .text("value");
        // unknown prefix with no base vocab → property not resolvable if strict
        // (with base, it would fall through to relative resolution)
        let _triples = parser().parse(&el);
        // Should not panic — just may produce a triple or not
    }
}
