//! Public-facing RDF/XML parser types — tokenizer / lexical phase.
//!
//! Contains `RdfXmlParser` (the entry-point builder), reader/slice/async
//! parser wrappers, and the prefix iterator `RdfXmlPrefixesIter`.

use crate::model::{NamedOrBlankNode, Term, Triple};
use crate::rdfxml::error::{RdfXmlParseError, RdfXmlSyntaxError};
use crate::rdfxml::parser_types::InternalRdfXmlParser;
use oxiri::{Iri, IriParseError};
use quick_xml::escape::unescape_with;
use quick_xml::name::{NamespaceBindingsIter, PrefixDeclaration};
use quick_xml::{Decoder, NsReader};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::{BufReader, Read};
#[cfg(feature = "async-tokio")]
use tokio::io::{AsyncRead, BufReader as AsyncBufReader};

use crate::rdfxml::utils::is_nc_name;

impl From<NamedOrBlankNode> for Term {
    fn from(node: NamedOrBlankNode) -> Self {
        match node {
            NamedOrBlankNode::NamedNode(n) => Term::NamedNode(n),
            NamedOrBlankNode::BlankNode(n) => Term::BlankNode(n),
        }
    }
}

/// A [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) streaming parser.
///
/// It reads the file in streaming.
/// It does not keep data in memory except a stack for handling nested XML tags, and a set of all
/// seen `rdf:ID`s to detect duplicate ids and fail according to the specification.
///
/// Its performances are not optimized yet and hopefully could be significantly enhanced by reducing the
/// number of allocations and copies done by the parser.
///
/// Count the number of people:
/// ```
/// use oxirs_core::model::NamedNode;
/// use oxirs_core::{Predicate, Object};
/// use oxirs_core::rdfxml::RdfXmlParser;
///
/// let file = br#"<?xml version="1.0"?>
/// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
///  <rdf:Description rdf:about="http://example.com/foo">
///    <rdf:type rdf:resource="http://schema.org/Person" />
///    <schema:name>Foo</schema:name>
///  </rdf:Description>
///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
/// </rdf:RDF>"#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person").expect("valid IRI");
/// let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI");
/// let mut count = 0;
/// for triple in RdfXmlParser::new().for_reader(file.as_ref()) {
///     let triple = triple.expect("triple should be valid");
///     if matches!(triple.predicate(), oxirs_core::Predicate::NamedNode(n) if n == &rdf_type) && matches!(triple.object(), oxirs_core::Object::NamedNode(n) if n == &schema_person) {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// ```
#[derive(Default, Clone)]
#[must_use]
pub struct RdfXmlParser {
    pub(super) lenient: bool,
    pub(super) base: Option<Iri<String>>,
}

impl RdfXmlParser {
    /// Builds a new [`RdfXmlParser`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Assumes the file is valid to make parsing faster.
    ///
    /// It will skip some validations.
    ///
    /// Note that if the file is actually not valid, the parser might emit broken RDF.
    #[inline]
    pub fn lenient(mut self) -> Self {
        self.lenient = true;
        self
    }

    #[deprecated(note = "Use `lenient()` instead", since = "0.2.0")]
    #[inline]
    pub fn unchecked(self) -> Self {
        self.lenient()
    }

    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.base = Some(Iri::parse(base_iri.into())?);
        Ok(self)
    }

    /// Parses a RDF/XML file from a [`Read`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// use oxirs_core::model::NamedNode;
    /// use oxirs_core::{Predicate, Object};
    /// use oxirs_core::rdfxml::RdfXmlParser;
    ///
    /// let file = br#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///  <rdf:Description rdf:about="http://example.com/foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///    <schema:name>Foo</schema:name>
    ///  </rdf:Description>
    ///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person").expect("valid IRI");
    /// let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI");
    /// let mut count = 0;
    /// for triple in RdfXmlParser::new().for_reader(file.as_ref()) {
    ///     let triple = triple.expect("triple should be valid");
    ///     if matches!(triple.predicate(), oxirs_core::Predicate::NamedNode(n) if n == &rdf_type) && matches!(triple.object(), oxirs_core::Object::NamedNode(n) if n == &schema_person) {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// ```
    pub fn for_reader<R: Read>(self, reader: R) -> ReaderRdfXmlParser<R> {
        ReaderRdfXmlParser {
            results: Vec::new(),
            parser: self.into_internal(BufReader::new(reader)),
            reader_buffer: Vec::default(),
        }
    }

    /// Parses a RDF/XML file from a [`AsyncRead`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use oxirs_core::model::NamedNode;
    /// use oxirs_core::{Predicate, Object};
    /// use oxirs_core::rdfxml::RdfXmlParser;
    ///
    /// let file = br#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///   <rdf:Description rdf:about="http://example.com/foo">
    ///     <rdf:type rdf:resource="http://schema.org/Person" />
    ///     <schema:name>Foo</schema:name>
    ///   </rdf:Description>
    ///   <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person").expect("valid IRI");
    /// let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI");
    /// let mut count = 0;
    /// let mut parser = RdfXmlParser::new().for_tokio_async_reader(file.as_ref());
    /// while let Some(triple) = parser.next().await {
    ///     let triple = triple.expect("triple should be valid");
    ///     if matches!(triple.predicate(), oxirs_core::Predicate::NamedNode(n) if n == &rdf_type) && matches!(triple.object(), oxirs_core::Object::NamedNode(n) if n == &schema_person) {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> TokioAsyncReaderRdfXmlParser<R> {
        TokioAsyncReaderRdfXmlParser {
            results: Vec::new(),
            parser: self.into_internal(AsyncBufReader::new(reader)),
            reader_buffer: Vec::default(),
        }
    }

    /// Parses a RDF/XML file from a byte slice.
    ///
    /// Count the number of people:
    /// ```
    /// use oxirs_core::model::NamedNode;
    /// use oxirs_core::{Predicate, Object};
    /// use oxirs_core::rdfxml::RdfXmlParser;
    ///
    /// let file = br#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///  <rdf:Description rdf:about="http://example.com/foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///    <schema:name>Foo</schema:name>
    ///  </rdf:Description>
    ///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let schema_person = NamedNode::new("http://schema.org/Person").expect("valid IRI");
    /// let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI");
    /// let mut count = 0;
    /// for triple in RdfXmlParser::new().for_slice(file) {
    ///     let triple = triple.expect("triple should be valid");
    ///     if matches!(triple.predicate(), oxirs_core::Predicate::NamedNode(n) if n == &rdf_type) && matches!(triple.object(), oxirs_core::Object::NamedNode(n) if n == &schema_person) {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// ```
    pub fn for_slice(self, slice: &[u8]) -> SliceRdfXmlParser<'_> {
        SliceRdfXmlParser {
            results: Vec::new(),
            parser: self.into_internal(slice),
            reader_buffer: Vec::default(),
        }
    }

    pub(super) fn into_internal<T>(self, reader: T) -> InternalRdfXmlParser<T> {
        use crate::rdfxml::parser_types::RdfXmlState;
        let mut reader = NsReader::from_reader(reader);
        reader.config_mut().expand_empty_elements = true;
        InternalRdfXmlParser {
            reader,
            state: vec![RdfXmlState::Doc {
                base_iri: self.base.clone(),
            }],
            custom_entities: HashMap::new(),
            in_literal_depth: 0,
            known_rdf_id: HashSet::default(),
            is_end: false,
            lenient: self.lenient,
        }
    }
}

/// Parses a RDF/XML file from a [`Read`] implementation.
///
/// Can be built using [`RdfXmlParser::for_reader`].
///
/// Count the number of people:
/// ```
/// use oxirs_core::model::NamedNode;
/// use oxirs_core::{Predicate, Object};
/// use oxirs_core::rdfxml::RdfXmlParser;
///
/// let file = br#"<?xml version="1.0"?>
/// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
///  <rdf:Description rdf:about="http://example.com/foo">
///    <rdf:type rdf:resource="http://schema.org/Person" />
///    <schema:name>Foo</schema:name>
///  </rdf:Description>
///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
/// </rdf:RDF>"#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person").expect("valid IRI");
/// let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI");
/// let mut count = 0;
/// for triple in RdfXmlParser::new().for_reader(file.as_ref()) {
///     let triple = triple.expect("triple should be valid");
///     if matches!(triple.predicate(), oxirs_core::Predicate::NamedNode(n) if n == &rdf_type) && matches!(triple.object(), oxirs_core::Object::NamedNode(n) if n == &schema_person) {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// ```
#[must_use]
pub struct ReaderRdfXmlParser<R: Read> {
    results: Vec<Triple>,
    parser: InternalRdfXmlParser<BufReader<R>>,
    reader_buffer: Vec<u8>,
}

impl<R: Read> Iterator for ReaderRdfXmlParser<R> {
    type Item = Result<Triple, RdfXmlParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(triple) = self.results.pop() {
                return Some(Ok(triple));
            } else if self.parser.is_end {
                return None;
            }
            if let Err(e) = self.parse_step() {
                return Some(Err(e));
            }
        }
    }
}

impl<R: Read> ReaderRdfXmlParser<R> {
    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxirs_core::rdfxml::RdfXmlParser;
    ///
    /// let file = br#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///  <rdf:Description rdf:about="http://example.com/foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///    <schema:name>Foo</schema:name>
    ///  </rdf:Description>
    ///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_reader(file.as_ref());
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.next().expect("should have next item").expect("operation should succeed"); // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [
    ///         ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ///         ("schema", "http://schema.org/")
    ///     ]
    /// ); // There are now prefixes
    /// ```
    pub fn prefixes(&self) -> RdfXmlPrefixesIter<'_> {
        RdfXmlPrefixesIter {
            inner: self.parser.reader.resolver().bindings(),
            decoder: self.parser.reader.decoder(),
            lenient: self.parser.lenient,
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxirs_core::rdfxml::RdfXmlParser;
    ///
    /// let file = br#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xml:base="http://example.com/">
    ///  <rdf:Description rdf:about="foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///  </rdf:Description>
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_reader(file.as_ref());
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// parser.next().expect("should have next item").expect("operation should succeed"); // We read the first triple
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        Some(self.parser.current_base_iri()?.as_str())
    }

    /// The current byte position in the input data.
    pub fn buffer_position(&self) -> u64 {
        self.parser.reader.buffer_position()
    }

    fn parse_step(&mut self) -> Result<(), RdfXmlParseError> {
        self.reader_buffer.clear();
        let event = self
            .parser
            .reader
            .read_event_into(&mut self.reader_buffer)?;
        self.parser.parse_event(event, &mut self.results)
    }
}

/// Parses a RDF/XML file from a [`AsyncRead`] implementation.
///
/// Can be built using [`RdfXmlParser::for_tokio_async_reader`].
///
/// Count the number of people:
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use oxirs_core::model::NamedNode;
/// use oxirs_core::{Predicate, Object};
/// use oxirs_core::rdfxml::RdfXmlParser;
///
/// let file = br#"<?xml version="1.0"?>
/// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
///   <rdf:Description rdf:about="http://example.com/foo">
///     <rdf:type rdf:resource="http://schema.org/Person" />
///     <schema:name>Foo</schema:name>
///   </rdf:Description>
///   <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
/// </rdf:RDF>"#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person").expect("valid IRI");
/// let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI");
/// let mut count = 0;
/// let mut parser = RdfXmlParser::new().for_tokio_async_reader(file.as_ref());
/// while let Some(triple) = parser.next().await {
///     let triple = triple.expect("triple should be valid");
///     if matches!(triple.predicate(), oxirs_core::Predicate::NamedNode(n) if n == &rdf_type) && matches!(triple.object(), oxirs_core::Object::NamedNode(n) if n == &schema_person) {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncReaderRdfXmlParser<R: AsyncRead + Unpin> {
    results: Vec<Triple>,
    parser: InternalRdfXmlParser<AsyncBufReader<R>>,
    reader_buffer: Vec<u8>,
}

#[cfg(feature = "async-tokio")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderRdfXmlParser<R> {
    /// Reads the next triple or returns `None` if the file is finished.
    pub async fn next(&mut self) -> Option<Result<Triple, RdfXmlParseError>> {
        loop {
            if let Some(triple) = self.results.pop() {
                return Some(Ok(triple));
            } else if self.parser.is_end {
                return None;
            }
            if let Err(e) = self.parse_step().await {
                return Some(Err(e));
            }
        }
    }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use oxirs_core::rdfxml::RdfXmlParser;
    ///
    /// let file = br#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///  <rdf:Description rdf:about="http://example.com/foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///    <schema:name>Foo</schema:name>
    ///  </rdf:Description>
    ///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_tokio_async_reader(file.as_ref());
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.next().await.expect("async operation should succeed").expect("operation should succeed"); // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [
    ///         ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ///         ("schema", "http://schema.org/")
    ///     ]
    /// ); // There are now prefixes
    /// //
    /// # Ok(())
    /// # }
    /// ```
    pub fn prefixes(&self) -> RdfXmlPrefixesIter<'_> {
        RdfXmlPrefixesIter {
            inner: self.parser.reader.resolver().bindings(),
            decoder: self.parser.reader.decoder(),
            lenient: self.parser.lenient,
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// # [tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use oxirs_core::rdfxml::RdfXmlParser;
    ///
    /// let file = br#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xml:base="http://example.com/">
    ///  <rdf:Description rdf:about="foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///  </rdf:Description>
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_tokio_async_reader(file.as_ref());
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// parser.next().await.expect("async operation should succeed").expect("operation should succeed"); // We read the first triple
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// # Ok(())
    /// # }
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        Some(self.parser.current_base_iri()?.as_str())
    }

    /// The current byte position in the input data.
    pub fn buffer_position(&self) -> u64 {
        self.parser.reader.buffer_position()
    }

    async fn parse_step(&mut self) -> Result<(), RdfXmlParseError> {
        self.reader_buffer.clear();
        let event = self
            .parser
            .reader
            .read_event_into_async(&mut self.reader_buffer)
            .await?;
        self.parser.parse_event(event, &mut self.results)
    }
}

/// Parses a RDF/XML file from a byte slice.
///
/// Can be built using [`RdfXmlParser::for_slice`].
///
/// Count the number of people:
/// ```
/// use oxirs_core::model::NamedNode;
/// use oxirs_core::{Predicate, Object};
/// use oxirs_core::rdfxml::RdfXmlParser;
///
/// let file = br#"<?xml version="1.0"?>
/// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
///  <rdf:Description rdf:about="http://example.com/foo">
///    <rdf:type rdf:resource="http://schema.org/Person" />
///    <schema:name>Foo</schema:name>
///  </rdf:Description>
///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
/// </rdf:RDF>"#;
///
/// let schema_person = NamedNode::new("http://schema.org/Person").expect("valid IRI");
/// let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type").expect("valid IRI");
/// let mut count = 0;
/// for triple in RdfXmlParser::new().for_slice(file) {
///     let triple = triple.expect("triple should be valid");
///     if matches!(triple.predicate(), oxirs_core::Predicate::NamedNode(n) if n == &rdf_type) && matches!(triple.object(), oxirs_core::Object::NamedNode(n) if n == &schema_person) {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// ```
#[must_use]
pub struct SliceRdfXmlParser<'a> {
    results: Vec<Triple>,
    parser: InternalRdfXmlParser<&'a [u8]>,
    reader_buffer: Vec<u8>,
}

impl Iterator for SliceRdfXmlParser<'_> {
    type Item = Result<Triple, RdfXmlSyntaxError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(triple) = self.results.pop() {
                return Some(Ok(triple));
            } else if self.parser.is_end {
                return None;
            }
            if let Err(RdfXmlParseError::Syntax(e)) = self.parse_step() {
                // I/O errors can't happen
                return Some(Err(e));
            }
        }
    }
}

impl SliceRdfXmlParser<'_> {
    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxirs_core::rdfxml::RdfXmlParser;
    ///
    /// let file = br#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:schema="http://schema.org/">
    ///  <rdf:Description rdf:about="http://example.com/foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///    <schema:name>Foo</schema:name>
    ///  </rdf:Description>
    ///  <schema:Person rdf:about="http://example.com/bar" schema:name="Bar" />
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_slice(file);
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.next().expect("should have next item").expect("operation should succeed"); // We read the first triple
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [
    ///         ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
    ///         ("schema", "http://schema.org/")
    ///     ]
    /// ); // There are now prefixes
    /// ```
    pub fn prefixes(&self) -> RdfXmlPrefixesIter<'_> {
        RdfXmlPrefixesIter {
            inner: self.parser.reader.resolver().bindings(),
            decoder: self.parser.reader.decoder(),
            lenient: self.parser.lenient,
        }
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxirs_core::rdfxml::RdfXmlParser;
    ///
    /// let file = br#"<?xml version="1.0"?>
    /// <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xml:base="http://example.com/">
    ///  <rdf:Description rdf:about="foo">
    ///    <rdf:type rdf:resource="http://schema.org/Person" />
    ///  </rdf:Description>
    /// </rdf:RDF>"#;
    ///
    /// let mut parser = RdfXmlParser::new().for_slice(file);
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// parser.next().expect("should have next item").expect("operation should succeed"); // We read the first triple
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        Some(self.parser.current_base_iri()?.as_str())
    }

    /// The current byte position in the input data.
    pub fn buffer_position(&self) -> u64 {
        self.parser.reader.buffer_position()
    }

    fn parse_step(&mut self) -> Result<(), RdfXmlParseError> {
        self.reader_buffer.clear();
        let event = self
            .parser
            .reader
            .read_event_into(&mut self.reader_buffer)?;
        self.parser.parse_event(event, &mut self.results)
    }
}

/// Iterator on the file prefixes.
///
/// See [`ReaderRdfXmlParser::prefixes`].
pub struct RdfXmlPrefixesIter<'a> {
    inner: NamespaceBindingsIter<'a>,
    decoder: Decoder,
    lenient: bool,
}

impl<'a> Iterator for RdfXmlPrefixesIter<'a> {
    type Item = (&'a str, &'a str);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (key, value) = self.inner.next()?;
            return Some((
                match key {
                    PrefixDeclaration::Default => "",
                    PrefixDeclaration::Named(name) => {
                        let Ok(Cow::Borrowed(name)) = self.decoder.decode(name) else {
                            continue;
                        };
                        let Ok(Cow::Borrowed(name)) = unescape_with(name, |_| None) else {
                            continue;
                        };
                        if !self.lenient && !is_nc_name(name) {
                            continue; // We don't return invalid prefixes
                        }
                        name
                    }
                },
                {
                    let Ok(Cow::Borrowed(value)) = self.decoder.decode(value.0) else {
                        continue;
                    };
                    let Ok(Cow::Borrowed(value)) = unescape_with(value, |_| None) else {
                        continue;
                    };
                    if !self.lenient && Iri::parse(value).is_err() {
                        continue; // We don't return invalid prefixes
                    }
                    value
                },
            ));
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
