//! Streaming JSON-LD parser iterators.
//!
//! Provides the [`Read`]-, [`AsyncRead`]- and slice-backed parser iterators
//! ([`ReaderJsonLdParser`], [`TokioAsyncReaderJsonLdParser`],
//! [`SliceJsonLdParser`]), the [`JsonLdPrefixesIter`] prefix iterator and the
//! shared [`InternalJsonLdParser`] driver.

use super::context::{JsonLdLoadDocumentOptions, JsonLdRemoteDocument, JsonLdTermDefinition};
use super::error::{JsonLdParseError, JsonLdSyntaxError};
use super::expansion::{JsonLdEvent, JsonLdExpansionConverter};
use super::to_rdf_converter::JsonLdToRdfConverter;
use crate::model::*;
#[cfg(feature = "async")]
use json_event_parser::TokioAsyncReaderJsonParser;
use json_event_parser::{JsonEvent, ReaderJsonParser, SliceJsonParser};
use oxiri::Iri;
use std::error::Error;
use std::io::Read;
use std::panic::{RefUnwindSafe, UnwindSafe};
#[cfg(feature = "async")]
use tokio::io::AsyncRead;

/// Parses a JSON-LD file from a [`Read`] implementation.
///
/// Can be built using [`JsonLdParser::for_reader`](super::JsonLdParser::for_reader).
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
#[must_use]
pub struct ReaderJsonLdParser<R: Read> {
    pub(super) results: Vec<Quad>,
    pub(super) errors: Vec<JsonLdSyntaxError>,
    pub(super) inner: InternalJsonLdParser,
    pub(super) json_parser: ReaderJsonParser<R>,
}

impl<R: Read> Iterator for ReaderJsonLdParser<R> {
    type Item = Result<Quad, JsonLdParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(error) = self.errors.pop() {
                return Some(Err(error.into()));
            } else if let Some(quad) = self.results.pop() {
                return Some(Ok(quad));
            } else if self.inner.is_end() {
                return None;
            }
            let step = self.parse_step();
            if let Err(e) = step {
                return Some(Err(e));
            }
            // We make sure to have data in the right order
            self.results.reverse();
            self.errors.reverse();
        }
    }
}

impl<R: Read> ReaderJsonLdParser<R> {
    /// Allows setting a callback to load remote documents and contexts
    ///
    /// The first argument is the document URL.
    ///
    /// It corresponds to the [`documentLoader` option from the algorithm specification](https://www.w3.org/TR/json-ld11-api/#dom-jsonldoptions-documentloader).
    ///
    /// See [`LoadDocumentCallback` API documentation](https://www.w3.org/TR/json-ld-api/#loaddocumentcallback) for more details
    ///
    /// ```
    /// use oxjsonld::{JsonLdParser, JsonLdRemoteDocument};
    /// use oxrdf::NamedNodeRef;
    /// use oxrdf::vocab::rdf;
    ///
    /// let file = br#"{
    ///     "@context": "file://context.jsonld",
    ///     "@type": "schema:Person",
    ///     "@id": "http://example.com/foo",
    ///     "schema:name": "Foo"
    /// }"#;
    ///
    /// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in JsonLdParser::new()
    ///     .for_reader(file.as_ref())
    ///     .with_load_document_callback(|url, _options| {
    ///         assert_eq!(url, "file://context.jsonld");
    ///         Ok(JsonLdRemoteDocument {
    ///             document: br#"{"@context":{"schema": "http://schema.org/"}}"#.to_vec(),
    ///             document_url: "file://context.jsonld".into(),
    ///         })
    ///     })
    /// {
    ///     let quad = quad?;
    ///     if quad.predicate == rdf::TYPE && quad.object == schema_person.into() {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(1, count);
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn with_load_document_callback(
        mut self,
        callback: impl Fn(
                &str,
                &JsonLdLoadDocumentOptions,
            ) -> Result<JsonLdRemoteDocument, Box<dyn Error + Send + Sync>>
            + Send
            + Sync
            + UnwindSafe
            + RefUnwindSafe
            + 'static,
    ) -> Self {
        self.inner.expansion = self.inner.expansion.with_load_document_callback(callback);
        self
    }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxjsonld::JsonLdParser;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/", "@base": "http://example.com/"},
    ///     "@type": "schema:Person",
    ///     "@id": "foo",
    ///     "schema:name": "Foo"
    /// }"#;
    ///
    /// let mut parser = JsonLdParser::new().for_reader(file.as_ref());
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.next().expect("should have next item")?; // We read the first quad
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// //
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> JsonLdPrefixesIter<'_> {
        self.inner.prefixes()
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxjsonld::JsonLdParser;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/", "@base": "http://example.com/"},
    ///     "@type": "schema:Person",
    ///     "@id": "foo",
    ///     "schema:name": "Foo"
    /// }"#;
    ///
    /// let mut parser = JsonLdParser::new().for_reader(file.as_ref());
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// parser.next().expect("should have next item")?; // We read the first quad
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        self.inner.base_iri()
    }

    fn parse_step(&mut self) -> Result<(), JsonLdParseError> {
        let event = match self.json_parser.parse_next() {
            Ok(event) => event,
            Err(e) => {
                self.inner.json_error = true;
                return Err(e.into());
            }
        };
        self.inner
            .parse_event(event, &mut self.results, &mut self.errors);
        Ok(())
    }
}

/// Parses a JSON-LD file from a [`AsyncRead`] implementation.
///
/// Can be built using [`JsonLdParser::for_tokio_async_reader`](super::JsonLdParser::for_tokio_async_reader).
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
#[must_use]
pub struct TokioAsyncReaderJsonLdParser<R: AsyncRead + Unpin> {
    pub(super) results: Vec<Quad>,
    pub(super) errors: Vec<JsonLdSyntaxError>,
    pub(super) inner: InternalJsonLdParser,
    pub(super) json_parser: TokioAsyncReaderJsonParser<R>,
}

#[cfg(feature = "async")]
impl<R: AsyncRead + Unpin> TokioAsyncReaderJsonLdParser<R> {
    /// Reads the next quad or returns `None` if the file is finished.
    pub async fn next(&mut self) -> Option<Result<Quad, JsonLdParseError>> {
        loop {
            if let Some(error) = self.errors.pop() {
                return Some(Err(error.into()));
            } else if let Some(quad) = self.results.pop() {
                return Some(Ok(quad));
            } else if self.inner.is_end() {
                return None;
            }
            if let Err(e) = self.parse_step().await {
                return Some(Err(e));
            }
            // We make sure to have data in the right order
            self.results.reverse();
            self.errors.reverse();
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
    /// use oxjsonld::JsonLdParser;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/", "@base": "http://example.com/"},
    ///     "@type": "schema:Person",
    ///     "@id": "foo",
    ///     "schema:name": "Foo"
    /// }"#;
    ///
    /// let mut parser = JsonLdParser::new().for_tokio_async_reader(file.as_ref());
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// let _ = parser.next().await.transpose()?; // We read the first quad
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// //
    /// # Ok(())
    /// # }
    /// ```
    pub fn prefixes(&self) -> JsonLdPrefixesIter<'_> {
        self.inner.prefixes()
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use oxjsonld::JsonLdParser;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/", "@base": "http://example.com/"},
    ///     "@type": "schema:Person",
    ///     "@id": "foo",
    ///     "schema:name": "Foo"
    /// }"#;
    ///
    /// let mut parser = JsonLdParser::new().for_tokio_async_reader(file.as_ref());
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// let _ = parser.next().await.transpose()?; // We read the first quad
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// # Ok(())
    /// # }
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        self.inner.base_iri()
    }

    async fn parse_step(&mut self) -> Result<(), JsonLdParseError> {
        let event = match self.json_parser.parse_next().await {
            Ok(e) => e,
            Err(err) => {
                self.inner.json_error = true;
                return Err(err.into());
            }
        };
        self.inner
            .parse_event(event, &mut self.results, &mut self.errors);
        Ok(())
    }
}

/// Parses a JSON-LD file from a byte slice.
///
/// Can be built using [`JsonLdParser::for_slice`](super::JsonLdParser::for_slice).
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
#[must_use]
pub struct SliceJsonLdParser<'a> {
    pub(super) results: Vec<Quad>,
    pub(super) errors: Vec<JsonLdSyntaxError>,
    pub(super) inner: InternalJsonLdParser,
    pub(super) json_parser: SliceJsonParser<'a>,
}

impl Iterator for SliceJsonLdParser<'_> {
    type Item = Result<Quad, JsonLdSyntaxError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(error) = self.errors.pop() {
                return Some(Err(error));
            } else if let Some(quad) = self.results.pop() {
                return Some(Ok(quad));
            } else if self.inner.is_end() {
                return None;
            }
            if let Err(e) = self.parse_step() {
                // I/O errors cannot happen
                return Some(Err(e));
            }
            // We make sure to have data in the right order
            self.results.reverse();
            self.errors.reverse();
        }
    }
}

impl SliceJsonLdParser<'_> {
    /// Allows setting a callback to load remote documents and contexts
    ///
    /// The first argument is the document URL.
    ///
    /// It corresponds to the [`documentLoader` option from the algorithm specification](https://www.w3.org/TR/json-ld11-api/#dom-jsonldoptions-documentloader).
    ///
    /// See [`LoadDocumentCallback` API documentation](https://www.w3.org/TR/json-ld-api/#loaddocumentcallback) for more details
    ///
    /// ```ignore
    /// use oxirs_core::jsonld::{JsonLdParser, JsonLdRemoteDocument};
    /// use oxrdf::NamedNodeRef;
    /// use oxrdf::vocab::rdf;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let file = br#"{
    ///     "@context": "file://context.jsonld",
    ///     "@type": "schema:Person",
    ///     "@id": "http://example.com/foo",
    ///     "schema:name": "Foo"
    /// }"#;
    ///
    /// let schema_person = NamedNodeRef::new("http://schema.org/Person")?;
    /// let mut count = 0;
    /// for quad in JsonLdParser::new()
    ///     .for_slice(file)
    ///     .with_load_document_callback(|url, _options| {
    ///         assert_eq!(url, "file://context.jsonld");
    ///         Ok(JsonLdRemoteDocument {
    ///             document: br#"{"@context":{"schema": "http://schema.org/"}}"#.to_vec(),
    ///             document_url: "file://context.jsonld".into(),
    ///         })
    ///     })
    /// {
    ///     let quad = quad?;
    ///     if quad.predicate() == rdf::TYPE && quad.object() == schema_person.into() {
    ///         count += 1;
    ///     }
    /// }
    /// assert_eq!(1, count);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_load_document_callback(
        mut self,
        callback: impl Fn(
                &str,
                &JsonLdLoadDocumentOptions,
            ) -> Result<JsonLdRemoteDocument, Box<dyn Error + Send + Sync>>
            + Send
            + Sync
            + UnwindSafe
            + RefUnwindSafe
            + 'static,
    ) -> Self {
        self.inner.expansion = self.inner.expansion.with_load_document_callback(callback);
        self
    }

    /// The list of IRI prefixes considered at the current step of the parsing.
    ///
    /// This method returns (prefix name, prefix value) tuples.
    /// It is empty at the beginning of the parsing and gets updated when prefixes are encountered.
    /// It should be full at the end of the parsing (but if a prefix is overridden, only the latest version will be returned).
    ///
    /// ```
    /// use oxjsonld::JsonLdParser;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/", "@base": "http://example.com/"},
    ///     "@type": "schema:Person",
    ///     "@id": "foo",
    ///     "schema:name": "Foo"
    /// }"#;
    ///
    /// let mut parser = JsonLdParser::new().for_slice(file);
    /// assert_eq!(parser.prefixes().collect::<Vec<_>>(), []); // No prefix at the beginning
    ///
    /// parser.next().expect("should have next item")?; // We read the first quad
    /// assert_eq!(
    ///     parser.prefixes().collect::<Vec<_>>(),
    ///     [("schema", "http://schema.org/")]
    /// ); // There are now prefixes
    /// //
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn prefixes(&self) -> JsonLdPrefixesIter<'_> {
        self.inner.prefixes()
    }

    /// The base IRI considered at the current step of the parsing.
    ///
    /// ```
    /// use oxjsonld::JsonLdParser;
    ///
    /// let file = br#"{
    ///     "@context": {"schema": "http://schema.org/", "@base": "http://example.com/"},
    ///     "@type": "schema:Person",
    ///     "@id": "foo",
    ///     "schema:name": "Foo"
    /// }"#;
    ///
    /// let mut parser = JsonLdParser::new().for_slice(file);
    /// assert!(parser.base_iri().is_none()); // No base at the beginning because none has been given to the parser.
    ///
    /// parser.next().expect("should have next item")?; // We read the first quad
    /// assert_eq!(parser.base_iri(), Some("http://example.com/")); // There is now a base IRI.
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn base_iri(&self) -> Option<&str> {
        self.inner.base_iri()
    }

    fn parse_step(&mut self) -> Result<(), JsonLdSyntaxError> {
        let event = match self.json_parser.parse_next() {
            Ok(event) => event,
            Err(e) => {
                self.inner.json_error = true;
                return Err(e.into());
            }
        };
        self.inner
            .parse_event(event, &mut self.results, &mut self.errors);
        Ok(())
    }
}

/// Iterator on the file prefixes.
///
/// See [`ReaderJsonLdParser::prefixes`].
pub struct JsonLdPrefixesIter<'a> {
    pub(super) term_definitions: std::collections::hash_map::Iter<'a, String, JsonLdTermDefinition>,
    pub(super) lenient: bool,
}

impl<'a> Iterator for JsonLdPrefixesIter<'a> {
    type Item = (&'a str, &'a str);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (prefix, term_definition) = self.term_definitions.next()?;
            if term_definition.prefix_flag {
                if let Some(Some(mapping)) = &term_definition.iri_mapping {
                    if self.lenient || Iri::parse(mapping.as_str()).is_ok() {
                        return Some((prefix, mapping));
                    }
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.term_definitions.size_hint().1)
    }
}

pub(super) struct InternalJsonLdParser {
    pub(super) expansion: JsonLdExpansionConverter,
    pub(super) expended_events: Vec<JsonLdEvent>,
    pub(super) to_rdf: JsonLdToRdfConverter,
    pub(super) json_error: bool,
}

impl InternalJsonLdParser {
    pub(super) fn parse_event(
        &mut self,
        event: JsonEvent<'_>,
        results: &mut Vec<Quad>,
        errors: &mut Vec<JsonLdSyntaxError>,
    ) {
        self.expansion
            .convert_event(event, &mut self.expended_events, errors);
        for event in self.expended_events.drain(..) {
            self.to_rdf.convert_event(event, results);
        }
    }

    pub(super) fn is_end(&self) -> bool {
        self.json_error || self.expansion.is_end()
    }

    pub(super) fn base_iri(&self) -> Option<&str> {
        Some(self.expansion.context().base_iri.as_ref()?.as_str())
    }

    pub(super) fn prefixes(&self) -> JsonLdPrefixesIter<'_> {
        JsonLdPrefixesIter {
            term_definitions: self.expansion.context().term_definitions.iter(),
            lenient: self.to_rdf.lenient,
        }
    }
}
