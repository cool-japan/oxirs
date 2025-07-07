use crate::model::*;
use crate::model::{NamedOrBlankNode, NamedOrBlankNodeRef};
use crate::rdfxml::utils::*;
use crate::vocab::{rdf, xsd};
use oxiri::{Iri, IriParseError};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::Writer;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io;
use std::io::Write;
#[cfg(feature = "async-tokio")]
use std::sync::Arc;
#[cfg(feature = "async-tokio")]
use tokio::io::AsyncWrite;

// Helper function to convert SubjectRef to NamedOrBlankNodeRef
fn subject_to_named_or_blank<'a>(subject: SubjectRef<'a>) -> Option<NamedOrBlankNodeRef<'a>> {
    match subject {
        SubjectRef::NamedNode(n) => Some(NamedOrBlankNodeRef::NamedNode(n)),
        SubjectRef::BlankNode(b) => Some(NamedOrBlankNodeRef::BlankNode(b)),
        _ => None,
    }
}

// Helper function for owned conversion
fn subject_to_named_or_blank_owned(subject: &Subject) -> Option<NamedOrBlankNode> {
    match subject {
        Subject::NamedNode(n) => Some(NamedOrBlankNode::NamedNode(n.clone())),
        Subject::BlankNode(b) => Some(NamedOrBlankNode::BlankNode(b.clone())),
        _ => None,
    }
}

/// A [RDF/XML](https://www.w3.org/TR/rdf-syntax-grammar/) serializer.
///
/// ```
/// use oxrdf::{LiteralRef, NamedNodeRef, TripleRef};
/// use oxrdf::vocab::rdf;
/// use oxrdfxml::RdfXmlSerializer;
///
/// let mut serializer = RdfXmlSerializer::new().with_prefix("schema", "http://schema.org/")?.for_writer(Vec::new());
/// serializer.serialize_triple(TripleRef::new(
///     NamedNodeRef::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNodeRef::new("http://schema.org/Person")?,
/// ))?;
/// serializer.serialize_triple(TripleRef::new(
///     NamedNodeRef::new("http://example.com#me")?,
///     NamedNodeRef::new("http://schema.org/name")?,
///     LiteralRef::new_language_tagged_literal_unchecked("Foo Bar", "en"),
/// ))?;
/// assert_eq!(
///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:schema=\"http://schema.org/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\t<schema:Person rdf:about=\"http://example.com#me\">\n\t\t<schema:name xml:lang=\"en\">Foo Bar</schema:name>\n\t</schema:Person>\n</rdf:RDF>",
///     serializer.finish()?.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[derive(Default, Clone)]
#[must_use]
pub struct RdfXmlSerializer {
    prefixes: BTreeMap<String, String>,
    base_iri: Option<Iri<String>>,
}

impl RdfXmlSerializer {
    /// Builds a new [`RdfXmlSerializer`].
    #[inline]
    pub fn new() -> Self {
        Self {
            prefixes: BTreeMap::new(),
            base_iri: None,
        }
    }

    #[inline]
    pub fn with_prefix(
        mut self,
        prefix_name: impl Into<String>,
        prefix_iri: impl Into<String>,
    ) -> Result<Self, IriParseError> {
        let prefix_name = prefix_name.into();
        if prefix_name == "oxprefix" {
            return Ok(self); // It is reserved
        }
        self.prefixes
            .insert(prefix_name, Iri::parse(prefix_iri.into())?.into_inner());
        Ok(self)
    }

    /// ```
    /// use oxrdf::{NamedNodeRef, TripleRef};
    /// use oxrdfxml::RdfXmlSerializer;
    ///
    /// let mut serializer = RdfXmlSerializer::new()
    ///     .with_base_iri("http://example.com")?
    ///     .with_prefix("ex", "http://example.com/ns#")?
    ///     .for_writer(Vec::new());
    /// serializer.serialize_triple(TripleRef::new(
    ///     NamedNodeRef::new("http://example.com#me")?,
    ///     NamedNodeRef::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?,
    ///     NamedNodeRef::new("http://example.com/ns#Person")?,
    /// ))?;
    /// serializer.serialize_triple(TripleRef::new(
    ///     NamedNodeRef::new("http://example.com#me")?,
    ///     NamedNodeRef::new("http://example.com/ns#parent")?,
    ///     NamedNodeRef::new("http://example.com#other")?,
    /// ))?;
    /// assert_eq!(
    ///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xml:base=\"http://example.com\" xmlns:ex=\"http://example.com/ns#\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\t<ex:Person rdf:about=\"#me\">\n\t\t<ex:parent rdf:resource=\"#other\"/>\n\t</ex:Person>\n</rdf:RDF>",
    ///     serializer.finish()?.as_slice()
    /// );
    /// # Result::<_,Box<dyn std::error::Error>>::Ok(())
    /// ```
    #[inline]
    pub fn with_base_iri(mut self, base_iri: impl Into<String>) -> Result<Self, IriParseError> {
        self.base_iri = Some(Iri::parse(base_iri.into())?);
        Ok(self)
    }

    /// Serializes a RDF/XML file to a [`Write`] implementation.
    ///
    /// This writer does unbuffered writes.
    ///
    /// ```
    /// use oxrdf::{LiteralRef, NamedNodeRef, TripleRef};
    /// use oxrdf::vocab::rdf;
    /// use oxrdfxml::RdfXmlSerializer;
    ///
    /// let mut serializer = RdfXmlSerializer::new().with_prefix("schema", "http://schema.org/")?.for_writer(Vec::new());
    /// serializer.serialize_triple(TripleRef::new(
    ///     NamedNodeRef::new("http://example.com#me")?,
    ///     rdf::TYPE,
    ///     NamedNodeRef::new("http://schema.org/Person")?,
    /// ))?;
    /// serializer.serialize_triple(TripleRef::new(
    ///     NamedNodeRef::new("http://example.com#me")?,
    ///     NamedNodeRef::new("http://schema.org/name")?,
    ///     LiteralRef::new_language_tagged_literal_unchecked("Foo Bar", "en"),
    /// ))?;
    /// assert_eq!(
    ///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:schema=\"http://schema.org/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\t<schema:Person rdf:about=\"http://example.com#me\">\n\t\t<schema:name xml:lang=\"en\">Foo Bar</schema:name>\n\t</schema:Person>\n</rdf:RDF>",
    ///     serializer.finish()?.as_slice()
    /// );
    /// # Result::<_, Box<dyn std::error::Error>>::Ok(())
    /// ```
    pub fn for_writer<W: Write>(self, writer: W) -> WriterRdfXmlSerializer<W> {
        WriterRdfXmlSerializer {
            writer: Writer::new_with_indent(writer, b'\t', 1),
            inner: self.inner_writer(),
        }
    }

    /// Serializes a RDF/XML file to a [`AsyncWrite`] implementation.
    ///
    /// This writer does unbuffered writes.
    ///
    /// ```
    /// # #[tokio::main(flavor = "current_thread")]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use oxrdf::{NamedNodeRef, TripleRef, LiteralRef};
    /// use oxrdf::vocab::rdf;
    /// use oxrdfxml::RdfXmlSerializer;
    ///
    /// let mut serializer = RdfXmlSerializer::new().with_prefix("schema", "http://schema.org/")?.for_tokio_async_writer(Vec::new());
    /// serializer.serialize_triple(TripleRef::new(
    ///     NamedNodeRef::new("http://example.com#me")?,
    ///     rdf::TYPE,
    ///     NamedNodeRef::new("http://schema.org/Person")?,
    /// )).await?;
    /// serializer.serialize_triple(TripleRef::new(
    ///     NamedNodeRef::new("http://example.com#me")?,
    ///     NamedNodeRef::new("http://schema.org/name")?,
    ///     LiteralRef::new_language_tagged_literal_unchecked("Foo Bar", "en"),
    /// )).await?;
    /// assert_eq!(
    ///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:schema=\"http://schema.org/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\t<schema:Person rdf:about=\"http://example.com#me\">\n\t\t<schema:name xml:lang=\"en\">Foo Bar</schema:name>\n\t</schema:Person>\n</rdf:RDF>",
    ///     serializer.finish().await?.as_slice()
    /// );
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "async-tokio")]
    pub fn for_tokio_async_writer<W: AsyncWrite + Unpin>(
        self,
        writer: W,
    ) -> TokioAsyncWriterRdfXmlSerializer<W> {
        TokioAsyncWriterRdfXmlSerializer {
            writer: Writer::new_with_indent(writer, b'\t', 1),
            inner: self.inner_writer(),
        }
    }

    fn inner_writer(mut self) -> InnerRdfXmlWriter {
        // Makes sure rdf is the proper prefix, by first removing it
        self.prefixes.remove("rdf");
        let custom_default_prefix = self.prefixes.contains_key("");
        // The serializer want to have the URL first, we swap
        let mut prefixes = self
            .prefixes
            .into_iter()
            .map(|(key, value)| (value, key))
            .collect::<BTreeMap<_, _>>();
        prefixes.insert(
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".into(),
            "rdf".into(),
        );
        InnerRdfXmlWriter {
            current_subject: None,
            current_resource_tag: None,
            custom_default_prefix,
            prefixes_by_iri: prefixes,
            base_iri: self.base_iri,
        }
    }
}

/// Serializes a RDF/XML file to a [`Write`] implementation.
///
/// Can be built using [`RdfXmlSerializer::for_writer`].
///
/// ```
/// use oxrdf::{LiteralRef, NamedNodeRef, TripleRef};
/// use oxrdf::vocab::rdf;
/// use oxrdfxml::RdfXmlSerializer;
///
/// let mut serializer = RdfXmlSerializer::new().with_prefix("schema", "http://schema.org/")?.for_writer(Vec::new());
/// serializer.serialize_triple(TripleRef::new(
///     NamedNodeRef::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNodeRef::new("http://schema.org/Person")?,
/// ))?;
/// serializer.serialize_triple(TripleRef::new(
///     NamedNodeRef::new("http://example.com#me")?,
///     NamedNodeRef::new("http://schema.org/name")?,
///     LiteralRef::new_language_tagged_literal_unchecked("Foo Bar", "en"),
/// ))?;
/// assert_eq!(
///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:schema=\"http://schema.org/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\t<schema:Person rdf:about=\"http://example.com#me\">\n\t\t<schema:name xml:lang=\"en\">Foo Bar</schema:name>\n\t</schema:Person>\n</rdf:RDF>",
///     serializer.finish()?.as_slice()
/// );
/// # Result::<_, Box<dyn std::error::Error>>::Ok(())
/// ```
#[must_use]
pub struct WriterRdfXmlSerializer<W: Write> {
    writer: Writer<W>,
    inner: InnerRdfXmlWriter,
}

impl<W: Write> WriterRdfXmlSerializer<W> {
    /// Serializes an extra triple.
    pub fn serialize_triple<'a>(&mut self, t: impl Into<TripleRef<'a>>) -> io::Result<()> {
        let triple_ref = t.into();
        let mut buffer = Vec::new();
        // Split the borrow to avoid conflict
        let WriterRdfXmlSerializer { inner, writer } = self;
        inner.serialize_triple(triple_ref, &mut buffer)?;
        // Write buffer using appropriate writer method
        for event in buffer {
            writer
                .write_event(event)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        }
        Ok(())
    }

    /// Ends the write process and returns the underlying [`Write`].
    pub fn finish(mut self) -> io::Result<W> {
        let mut buffer = Vec::new();
        self.inner.finish(&mut buffer);
        self.flush_buffer(&mut buffer)?;
        Ok(self.writer.into_inner())
    }

    fn flush_buffer(&mut self, buffer: &mut Vec<Event<'_>>) -> io::Result<()> {
        for event in buffer.drain(0..) {
            self.writer
                .write_event(event)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        }
        Ok(())
    }
}

/// Serializes a RDF/XML file to a [`AsyncWrite`] implementation.
///
/// Can be built using [`RdfXmlSerializer::for_tokio_async_writer`].
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use oxrdf::{NamedNodeRef, TripleRef, LiteralRef};
/// use oxrdf::vocab::rdf;
/// use oxrdfxml::RdfXmlSerializer;
///
/// let mut serializer = RdfXmlSerializer::new().with_prefix("schema", "http://schema.org/")?.for_tokio_async_writer(Vec::new());
/// serializer.serialize_triple(TripleRef::new(
///     NamedNodeRef::new("http://example.com#me")?,
///     rdf::TYPE,
///     NamedNodeRef::new("http://schema.org/Person")?,
/// )).await?;
/// serializer.serialize_triple(TripleRef::new(
///     NamedNodeRef::new("http://example.com#me")?,
///     NamedNodeRef::new("http://schema.org/name")?,
///     LiteralRef::new_language_tagged_literal_unchecked("Foo Bar", "en"),
/// )).await?;
/// assert_eq!(
///     b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:schema=\"http://schema.org/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\t<schema:Person rdf:about=\"http://example.com#me\">\n\t\t<schema:name xml:lang=\"en\">Foo Bar</schema:name>\n\t</schema:Person>\n</rdf:RDF>",
///     serializer.finish().await?.as_slice()
/// );
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "async-tokio")]
#[must_use]
pub struct TokioAsyncWriterRdfXmlSerializer<W: AsyncWrite + Unpin> {
    writer: Writer<W>,
    inner: InnerRdfXmlWriter,
}

#[cfg(feature = "async-tokio")]
impl<W: AsyncWrite + Unpin> TokioAsyncWriterRdfXmlSerializer<W> {
    /// Serializes an extra triple.
    pub async fn serialize_triple<'a>(&mut self, t: impl Into<TripleRef<'a>>) -> io::Result<()> {
        let mut buffer = Vec::new();
        self.inner.serialize_triple(t, &mut buffer)?;
        self.flush_buffer(&mut buffer).await
    }

    /// Ends the write process and returns the underlying [`Write`].
    pub async fn finish(mut self) -> io::Result<W> {
        let mut buffer = Vec::new();
        self.inner.finish(&mut buffer);
        self.flush_buffer(&mut buffer).await?;
        Ok(self.writer.into_inner())
    }

    async fn flush_buffer(&mut self, buffer: &mut Vec<Event<'_>>) -> io::Result<()> {
        for event in buffer.drain(0..) {
            self.writer
                .write_event_async(event)
                .await
                .map_err(map_err)?;
        }
        Ok(())
    }
}

const RESERVED_SYNTAX_TERMS: [&str; 9] = [
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Description",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#li",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#RDF",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#ID",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#about",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#parseType",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#resource",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#nodeID",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#datatype",
];

pub struct InnerRdfXmlWriter {
    current_subject: Option<NamedOrBlankNode>,
    current_resource_tag: Option<String>,
    custom_default_prefix: bool,
    prefixes_by_iri: BTreeMap<String, String>,
    base_iri: Option<Iri<String>>,
}

impl InnerRdfXmlWriter {
    fn serialize_triple<'a>(
        &'a mut self,
        t: impl Into<TripleRef<'a>>,
        output: &mut Vec<Event<'a>>,
    ) -> io::Result<()> {
        if self.current_subject.is_none() {
            self.write_start(output);
        }

        let triple = t.into();
        // We open a new rdf:Description if useful
        if self.current_subject.as_ref().map(NamedOrBlankNode::as_ref)
            != subject_to_named_or_blank(triple.subject())
        {
            if self.current_subject.is_some() {
                output.push(Event::End(
                    self.current_resource_tag
                        .take()
                        .map_or_else(|| BytesEnd::new("rdf:Description"), BytesEnd::new),
                ));
            }
            if let Some(subj) = subject_to_named_or_blank_owned(&triple.subject().to_owned()) {
                self.current_subject = Some(subj);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "RDF/XML only supports named or blank subject",
                ));
            }

            let (mut description_open, with_type_tag) = if matches!(triple.predicate(), PredicateRef::NamedNode(n) if n == &*rdf::TYPE)
            {
                if let ObjectRef::NamedNode(t) = triple.object() {
                    if RESERVED_SYNTAX_TERMS.contains(&t.as_str()) {
                        (BytesStart::new("rdf:Description"), false)
                    } else {
                        let (prop_qname, prop_xmlns) = self.uri_to_qname_and_xmlns(t);
                        let prop_qname_owned = prop_qname.into_owned();
                        let mut description_open = BytesStart::new(prop_qname_owned.clone());
                        if let Some((attr_name, attr_value)) = prop_xmlns {
                            description_open
                                .push_attribute((attr_name.as_str(), attr_value.as_str()));
                        }
                        self.current_resource_tag = Some(prop_qname_owned);
                        (description_open, true)
                    }
                } else {
                    (BytesStart::new("rdf:Description"), false)
                }
            } else {
                (BytesStart::new("rdf:Description"), false)
            };
            #[allow(
                unreachable_patterns,
                clippy::match_wildcard_for_single_variants,
                clippy::allow_attributes
            )]
            match triple.subject() {
                SubjectRef::NamedNode(node) => description_open.push_attribute((
                    "rdf:about",
                    relative_iri(node.as_str(), &self.base_iri).as_ref(),
                )),
                SubjectRef::BlankNode(node) => {
                    description_open.push_attribute(("rdf:nodeID", node.as_str()))
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "RDF/XML only supports named or blank subject",
                    ));
                }
            }
            output.push(Event::Start(description_open));
            if with_type_tag {
                return Ok(()); // No need for a value
            }
        }

        let pred_node = match triple.predicate() {
            PredicateRef::NamedNode(n) => n,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "RDF/XML only supports named node predicates",
                ))
            }
        };

        if RESERVED_SYNTAX_TERMS.contains(&pred_node.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "RDF/XML reserved syntax term is not allowed as a predicate",
            ));
        }
        let (prop_qname, prop_xmlns) = self.uri_to_qname_and_xmlns(pred_node);
        let prop_qname_owned = prop_qname.into_owned();
        let mut property_open = BytesStart::new(prop_qname_owned.clone());
        if let Some((attr_name, attr_value)) = prop_xmlns {
            property_open.push_attribute((attr_name.as_str(), attr_value.as_str()));
        }
        #[allow(
            unreachable_patterns,
            clippy::match_wildcard_for_single_variants,
            clippy::allow_attributes
        )]
        let content = match triple.object() {
            ObjectRef::NamedNode(node) => {
                property_open.push_attribute((
                    "rdf:resource",
                    relative_iri(node.as_str(), &self.base_iri).as_ref(),
                ));
                None
            }
            ObjectRef::BlankNode(node) => {
                property_open.push_attribute(("rdf:nodeID", node.as_str()));
                None
            }
            ObjectRef::Literal(literal) => {
                if let Some(language) = literal.language() {
                    property_open.push_attribute(("xml:lang", language));
                } else if literal.datatype() != xsd::STRING.as_ref() {
                    property_open.push_attribute((
                        "rdf:datatype",
                        relative_iri(literal.datatype().as_str(), &self.base_iri).as_ref(),
                    ));
                }
                Some(literal.value())
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "RDF/XML only supports named, blank or literal object",
                ));
            }
        };
        if let Some(content) = content {
            output.push(Event::Start(property_open));
            output.push(Event::Text(BytesText::new(content)));
            output.push(Event::End(BytesEnd::new(prop_qname_owned)));
        } else {
            output.push(Event::Empty(property_open));
        }
        Ok(())
    }

    fn write_start(&self, output: &mut Vec<Event<'_>>) {
        output.push(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)));
        let mut rdf_open = BytesStart::new("rdf:RDF");
        if let Some(base_iri) = &self.base_iri {
            rdf_open.push_attribute(("xml:base", base_iri.as_str()));
        }
        for (prefix_value, prefix_name) in &self.prefixes_by_iri {
            rdf_open.push_attribute((
                if prefix_name.is_empty() {
                    "xmlns".into()
                } else {
                    format!("xmlns:{prefix_name}")
                }
                .as_str(),
                prefix_value.as_str(),
            ));
        }
        output.push(Event::Start(rdf_open))
    }

    fn finish(&mut self, output: &mut Vec<Event<'static>>) {
        if self.current_subject.is_some() {
            output.push(Event::End(
                self.current_resource_tag
                    .take()
                    .map_or_else(|| BytesEnd::new("rdf:Description"), BytesEnd::new),
            ));
        } else {
            self.write_start(output);
        }
        output.push(Event::End(BytesEnd::new("rdf:RDF")));
    }

    fn uri_to_qname_and_xmlns(&self, uri: &NamedNode) -> (Cow<str>, Option<(String, String)>) {
        let uri_str = uri.as_str();
        let (prop_prefix, prop_value) = split_iri(uri_str);
        if let Some(prefix) = self.prefixes_by_iri.get(prop_prefix) {
            (
                if prefix.is_empty() {
                    Cow::Owned(prop_value.to_string())
                } else {
                    Cow::Owned(format!("{prefix}:{prop_value}"))
                },
                None,
            )
        } else if prop_prefix == "http://www.w3.org/2000/xmlns/" {
            (Cow::Owned(format!("xmlns:{prop_value}")), None)
        } else if !prop_value.is_empty() && !self.custom_default_prefix {
            (
                Cow::Owned(prop_value.to_string()),
                Some(("xmlns".to_string(), prop_prefix.to_string())),
            )
        } else {
            // TODO: does not work on recursive elements
            (
                Cow::Owned(format!("oxprefix:{prop_value}")),
                Some(("xmlns:oxprefix".to_string(), prop_prefix.to_string())),
            )
        }
    }
}

#[cfg(feature = "async-tokio")]
fn map_err(error: quick_xml::Error) -> io::Error {
    if let quick_xml::Error::Io(error) = error {
        Arc::try_unwrap(error).unwrap_or_else(|error| io::Error::new(error.kind(), error))
    } else {
        io::Error::other(error)
    }
}

fn split_iri(iri: &str) -> (&str, &str) {
    if let Some(position_base) = iri.rfind(|c| !is_name_char(c) || c == ':') {
        if let Some(position_add) = iri[position_base..].find(|c| is_name_start_char(c) && c != ':')
        {
            (
                &iri[..position_base + position_add],
                &iri[position_base + position_add..],
            )
        } else {
            (iri, "")
        }
    } else {
        (iri, "")
    }
}

fn relative_iri<'a>(iri: &'a str, base_iri: &Option<Iri<String>>) -> Cow<'a, str> {
    if let Some(base_iri) = base_iri {
        if let Ok(relative) = base_iri.relativize(&Iri::parse_unchecked(iri)) {
            return relative.into_inner().into();
        }
    }
    iri.into()
}

#[cfg(test)]
#[expect(clippy::panic_in_result_fn)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_split_iri() {
        assert_eq!(
            split_iri("http://schema.org/Person"),
            ("http://schema.org/", "Person")
        );
        assert_eq!(split_iri("http://schema.org/"), ("http://schema.org/", ""));
        assert_eq!(
            split_iri("http://schema.org#foo"),
            ("http://schema.org#", "foo")
        );
        assert_eq!(split_iri("urn:isbn:foo"), ("urn:isbn:", "foo"));
    }

    #[test]
    fn test_custom_rdf_ns() -> Result<(), Box<dyn Error>> {
        let output = RdfXmlSerializer::new()
            .with_prefix("rdf", "http://example.com/")?
            .for_writer(Vec::new())
            .finish()?;
        assert_eq!(output, b"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n</rdf:RDF>");
        Ok(())
    }

    #[test]
    fn test_custom_empty_ns() -> Result<(), Box<dyn Error>> {
        let mut serializer = RdfXmlSerializer::new()
            .with_prefix("", "http://example.com/")?
            .for_writer(Vec::new());
        serializer.serialize_triple(TripleRef::new(
            SubjectRef::NamedNode(&NamedNode::new("http://example.com/s")?),
            PredicateRef::NamedNode(&rdf::TYPE),
            ObjectRef::NamedNode(&NamedNode::new("http://example.org/o")?),
        ))?;
        serializer.serialize_triple(TripleRef::new(
            SubjectRef::NamedNode(&NamedNode::new("http://example.com/s")?),
            PredicateRef::NamedNode(&NamedNode::new("http://example.com/p")?),
            ObjectRef::NamedNode(&NamedNode::new("http://example.com/o2")?),
        ))?;
        let output = serializer.finish()?;
        assert_eq!(
            String::from_utf8_lossy(&output),
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<rdf:RDF xmlns=\"http://example.com/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n\t<oxprefix:o xmlns:oxprefix=\"http://example.org/\" rdf:about=\"http://example.com/s\">\n\t\t<p rdf:resource=\"http://example.com/o2\"/>\n\t</oxprefix:o>\n</rdf:RDF>"
        );
        Ok(())
    }
}
