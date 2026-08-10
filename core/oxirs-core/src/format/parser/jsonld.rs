//! JSON-LD format parsing implementation
//!
//! Uses oxjsonld for parsing until native implementation is complete

use super::{RdfParser, ReaderQuadParser, SliceQuadParser};
use oxjsonld::JsonLdParser;
use std::io::Read;
use std::sync::mpsc::sync_channel;
use std::thread;

use super::helpers::convert_quad;
// convert_quad imported from helpers

/// Number of parsed quads buffered between the JSON-LD parsing thread and its consumer.
///
/// The channel is bounded so a slow consumer applies backpressure to the parser instead of
/// letting the whole document accumulate in memory.
const READER_CHANNEL_CAPACITY: usize = 1024;

pub(super) fn parse_reader<R: Read + Send + 'static>(
    parser: RdfParser,
    reader: R,
) -> ReaderQuadParser<'static, R> {
    // The JSON-LD reader parser owns a boxed document loader and is therefore not `Send`,
    // while `ReaderQuadParser` exposes a `Send` iterator. Build the parser on a dedicated
    // thread (the reader itself is `Send + 'static`) and stream the quads back through a
    // bounded channel: that keeps both the `Send` guarantee and the streaming behaviour.
    let base_iri = parser.base_iri().map(str::to_owned);
    let (sender, receiver) = sync_channel(READER_CHANNEL_CAPACITY);

    thread::spawn(move || {
        let oxjsonld_parser = if let Some(base) = &base_iri {
            JsonLdParser::new()
                .with_base_iri(base.as_str())
                .unwrap_or_else(|_| JsonLdParser::new())
        } else {
            JsonLdParser::new()
        };

        for result in oxjsonld_parser.for_reader(reader) {
            let quad = result
                .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
                .and_then(convert_quad);
            // A send failure means the consumer dropped the iterator: stop parsing.
            if sender.send(quad).is_err() {
                break;
            }
        }
    });

    ReaderQuadParser::new(Box::new(receiver.into_iter()))
}

pub(super) fn parse_slice<'a>(parser: RdfParser, slice: &'a [u8]) -> SliceQuadParser<'a> {
    let oxjsonld_parser = if let Some(base) = parser.base_iri() {
        JsonLdParser::new()
            .with_base_iri(base)
            .unwrap_or_else(|_| JsonLdParser::new())
    } else {
        JsonLdParser::new()
    };

    // Parse the slice
    let iter = oxjsonld_parser.for_slice(slice).map(|result| {
        result
            .map_err(|e| crate::format::error::RdfParseError::syntax(e.to_string()))
            .and_then(convert_quad)
    });

    SliceQuadParser::new(Box::new(iter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::format::{JsonLdProfileSet, RdfFormat};
    use std::io::Cursor;

    const DOCUMENT: &[u8] = br#"{
        "@context": {"schema": "http://schema.org/"},
        "@graph": [
            {"@id": "http://example.com/foo", "@type": "schema:Person", "schema:name": "Foo"},
            {"@id": "http://example.com/bar", "@type": "schema:Person", "schema:name": "Bar"}
        ]
    }"#;

    fn jsonld_parser() -> RdfParser {
        RdfParser::new(RdfFormat::JsonLd {
            profile: JsonLdProfileSet::empty(),
        })
    }

    /// The reader path hands the parse off to a worker thread, so this also covers the
    /// channel handshake, not just the parse.
    #[test]
    fn reader_streams_every_quad() {
        let quads = jsonld_parser()
            .for_reader(Cursor::new(DOCUMENT))
            .collect::<Result<Vec<_>, _>>()
            .expect("JSON-LD document should parse");
        assert_eq!(quads.len(), 4);
        assert!(quads
            .iter()
            .any(|quad| quad.subject().to_string().contains("http://example.com/foo")));
    }

    /// The reader and slice paths run different code, so they must agree.
    #[test]
    fn reader_and_slice_agree() {
        let from_reader = jsonld_parser()
            .for_reader(Cursor::new(DOCUMENT))
            .collect::<Result<Vec<_>, _>>()
            .expect("reader parse should succeed");
        let from_slice = jsonld_parser()
            .for_slice(DOCUMENT)
            .collect::<Result<Vec<_>, _>>()
            .expect("slice parse should succeed");
        assert_eq!(from_reader, from_slice);
    }

    /// Dropping the iterator early must let the worker thread wind down instead of
    /// blocking forever on a full channel.
    #[test]
    fn dropping_the_reader_early_is_clean() {
        let mut parser = jsonld_parser().for_reader(Cursor::new(DOCUMENT));
        let first = parser.next().expect("at least one quad");
        assert!(first.is_ok());
        drop(parser);
    }

    /// A malformed document must surface as an error item rather than a silent truncation.
    #[test]
    fn reader_reports_syntax_errors() {
        let results = jsonld_parser()
            .for_reader(Cursor::new(&b"{\"@id\": "[..]))
            .collect::<Vec<_>>();
        assert!(results.iter().any(std::result::Result::is_err));
    }
}
