//! The [`JsonLdParser`] builder.
//!
//! Provides the public configuration entry point for JSON-LD parsing and the
//! `for_reader` / `for_slice` / `for_tokio_async_reader` constructors that
//! produce the streaming parser iterators.

use super::expansion::JsonLdExpansionConverter;
use super::profile::{JsonLdProcessingMode, JsonLdProfile, JsonLdProfileSet};
use super::to_rdf_converter::{JsonLdToRdfConverter, JsonLdToRdfState};
#[cfg(feature = "async")]
use super::to_rdf_readers::TokioAsyncReaderJsonLdParser;
use super::to_rdf_readers::{InternalJsonLdParser, ReaderJsonLdParser, SliceJsonLdParser};
use crate::model::*;
#[cfg(feature = "async")]
use json_event_parser::TokioAsyncReaderJsonParser;
use json_event_parser::{ReaderJsonParser, SliceJsonParser};
use oxiri::{Iri, IriParseError};
use std::io::Read;
#[cfg(feature = "async")]
use tokio::io::AsyncRead;

/// A [JSON-LD](https://www.w3.org/TR/json-ld/) parser.
///
/// The parser is a work in progress.
/// Only JSON-LD 1.0 is supported at the moment. JSON-LD 1.1 is not supported yet.
///
/// The parser supports two modes:
/// - regular JSON-LD parsing that needs to buffer the full file into memory.
/// - [Streaming JSON-LD](https://www.w3.org/TR/json-ld11-streaming/) that can avoid buffering in a few cases.
///   To enable it call the [`with_profile(JsonLdProfile::Streaming)`](JsonLdParser::with_profile) method.
///
/// Count the number of people:
/// ```
/// use oxjsonld::JsonLdParser;
/// use oxrdf::NamedNodeRef;
/// use oxrdf::vocab::rdf;
///
/// let file = br#"{
///     "@context": {"schema": "http://schema.org/"},
///     "@graph": [
///         {
///             "@type": "schema:Person",
///             "@id": "http://example.com/foo",
///             "schema:name": "Foo"
///         },
///         {
///             "@type": "schema:Person",
///             "schema:name": "Bar"
///         }
///     ]
/// }"#;
///
/// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
/// let mut count = 0;
/// for quad in JsonLdParser::new().for_reader(file.as_ref()) {
///     let quad = quad?;
///     if quad.predicate == rdf::TYPE && quad.object == schema_person.into() {
///         count += 1;
///     }
/// }
/// assert_eq!(2, count);
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default, Clone)]
#[must_use]
pub struct JsonLdParser {
    processing_mode: JsonLdProcessingMode,
    lenient: bool,
    profile: JsonLdProfileSet,
    base: Option<Iri<String>>,
}

impl JsonLdParser {
    /// Builds a new [`JsonLdParser`].
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

    /// Assume the given profile(s) during parsing.
    ///
    /// If you set the [Streaming JSON-LD](https://www.w3.org/TR/json-ld11-streaming/) profile ([`JsonLdProfile::Streaming`]),
    /// the parser will skip some buffering to make parsing faster and memory consumption lower.
    ///
    /// ```
    /// use oxjsonld::{JsonLdParser, JsonLdProfile};
    /// use oxrdf::NamedNodeRef;
    /// use oxrdf::vocab::rdf;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/"},
    ///     "@graph": [
    ///         {
    ///             "@type": "schema:Person",
    ///             "@id": "http://example.com/foo",
    ///             "schema:name": "Foo"
    ///         }
    ///     ]
    /// }"#;
    ///
    /// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in JsonLdParser::new()
    ///     .with_profile(JsonLdProfile::Streaming)
    ///     .for_slice(file)
    /// {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person.into() {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(1, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[inline]
    pub fn with_profile(mut self, profile: impl Into<JsonLdProfileSet>) -> Self {
        self.profile = profile.into();
        self
    }

    /// Set the [processing mode](https://www.w3.org/TR/json-ld11/#dfn-processing-mode) of the parser.
    #[inline]
    #[doc(hidden)] // TODO: expose after implementing JSON-LD 1.1
    pub fn with_processing_mode(mut self, processing_mode: JsonLdProcessingMode) -> Self {
        self.processing_mode = processing_mode;
        self
    }

    /// Base IRI to use when expanding the document.
    ///
    /// It corresponds to the [`base` option from the algorithm specification](https://www.w3.org/TR/json-ld-api/#dom-jsonldoptions-base).
    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.base = Some(Iri::parse(base_iri.into())?);
        Ok(self)
    }

    /// Parses a JSON-LD file from a [`Read`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// use oxjsonld::JsonLdParser;
    /// use oxrdf::NamedNodeRef;
    /// use oxrdf::vocab::rdf;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/"},
    ///     "@graph": [
    ///         {
    ///             "@type": "schema:Person",
    ///             "@id": "http://example.com/foo",
    ///             "schema:name": "Foo"
    ///         },
    ///         {
    ///             "@type": "schema:Person",
    ///             "schema:name": "Bar"
    ///         }
    ///     ]
    /// }"#;
    ///
    /// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in JsonLdParser::new().for_reader(file.as_ref()) {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person.into() {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_reader<R: Read>(self, reader: R) -> ReaderJsonLdParser<R> {
        ReaderJsonLdParser {
            results: Vec::new(),
            errors: Vec::new(),
            inner: self.into_inner(),
            json_parser: ReaderJsonParser::new(reader),
        }
    }

    /// Parses a JSON-LD file from a [`AsyncRead`] implementation.
    ///
    /// Count the number of people:
    /// ```
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use oxjsonld::JsonLdParser;
    /// use oxrdf::NamedNodeRef;
    /// use oxrdf::vocab::rdf;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/"},
    ///     "@graph": [
    ///         {
    ///             "@type": "schema:Person",
    ///             "@id": "http://example.com/foo",
    ///             "schema:name": "Foo"
    ///         },
    ///         {
    ///             "@type": "schema:Person",
    ///             "schema:name": "Bar"
    ///         }
    ///     ]
    /// }"#;
    ///
    /// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// let mut parser = JsonLdParser::new().for_tokio_async_reader(file.as_ref());
    /// while let Some(quad) = parser.next().await {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person.into() {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async")]
    pub fn for_tokio_async_reader<R: AsyncRead + Unpin>(
        self,
        reader: R,
    ) -> TokioAsyncReaderJsonLdParser<R> {
        TokioAsyncReaderJsonLdParser {
            results: Vec::new(),
            errors: Vec::new(),
            inner: self.into_inner(),
            json_parser: TokioAsyncReaderJsonParser::new(reader),
        }
    }

    /// Parses a JSON-LD file from a byte slice.
    ///
    /// Count the number of people:
    /// ```
    /// use oxjsonld::JsonLdParser;
    /// use oxrdf::NamedNodeRef;
    /// use oxrdf::vocab::rdf;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/"},
    ///     "@graph": [
    ///         {
    ///             "@type": "schema:Person",
    ///             "@id": "http://example.com/foo",
    ///             "schema:name": "Foo"
    ///         },
    ///         {
    ///             "@type": "schema:Person",
    ///             "schema:name": "Bar"
    ///         }
    ///     ]
    /// }"#;
    ///
    /// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in JsonLdParser::new().for_slice(file) {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person.into() {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(2, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_slice(self, slice: &[u8]) -> SliceJsonLdParser<'_> {
        SliceJsonLdParser {
            results: Vec::new(),
            errors: Vec::new(),
            inner: self.into_inner(),
            json_parser: SliceJsonParser::new(slice),
        }
    }

    fn into_inner(self) -> InternalJsonLdParser {
        InternalJsonLdParser {
            expansion: JsonLdExpansionConverter::new(
                self.base,
                self.profile.contains(JsonLdProfile::Streaming),
                self.lenient,
                self.processing_mode,
            ),
            expended_events: Vec::new(),
            to_rdf: JsonLdToRdfConverter {
                state: vec![JsonLdToRdfState::Graph(Some(GraphName::DefaultGraph))],
                lenient: self.lenient,
            },
            json_error: false,
        }
    }
}
