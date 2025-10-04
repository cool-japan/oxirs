//! Content Negotiation for GSP

use super::types::{GspError, RdfFormat, RdfFormatExt};

/// Parse Accept header and determine best RDF format
pub fn negotiate_format(accept_header: Option<&str>) -> Result<RdfFormat, GspError> {
    let accept = accept_header.unwrap_or("*/*");

    // Parse Accept header
    let media_types = parse_accept_header(accept);

    // Find first matching format
    for (media_type, _quality) in media_types {
        if let Some(format) = RdfFormat::from_media_type_gsp(&media_type) {
            return Ok(format);
        }
        // Handle wildcards
        if media_type == "*/*" || media_type == "text/*" {
            return Ok(RdfFormat::Turtle); // Default to Turtle
        }
    }

    // No acceptable format found
    Err(GspError::NotAcceptable)
}

/// Parse Accept header into list of (media_type, quality) pairs
fn parse_accept_header(accept: &str) -> Vec<(String, f32)> {
    let mut types: Vec<(String, f32)> = accept
        .split(',')
        .filter_map(|part| {
            let part = part.trim();
            let mut segments = part.split(';');

            let media_type = segments.next()?.trim().to_lowercase();

            // Parse quality value
            let quality = segments
                .find_map(|seg| {
                    let seg = seg.trim();
                    if seg.starts_with("q=") {
                        seg[2..].parse::<f32>().ok()
                    } else {
                        None
                    }
                })
                .unwrap_or(1.0);

            Some((media_type, quality))
        })
        .collect();

    // Sort by quality (descending)
    types.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    types
}

/// Parse Content-Type header to determine input format
pub fn parse_content_type(content_type: Option<&str>) -> Result<RdfFormat, GspError> {
    let content_type =
        content_type.ok_or_else(|| GspError::BadRequest("No Content-Type header".to_string()))?;

    RdfFormat::from_media_type_gsp(content_type).ok_or_else(|| {
        GspError::UnsupportedMediaType(format!("Unsupported Content-Type: {}", content_type))
    })
}

/// Serialize RDF triples to string in specified format
pub fn serialize_triples(
    triples: &[oxirs_core::model::Triple],
    format: RdfFormat,
) -> Result<String, GspError> {
    use oxirs_core::model::Graph;
    use oxirs_core::serializer::Serializer;

    // Convert triples to Graph for serialization
    let graph = Graph::from_triples(triples.to_vec());
    let serializer = Serializer::new(format);

    serializer
        .serialize_graph(&graph)
        .map_err(|e| GspError::Internal(format!("Serialization error: {}", e)))
}

/// Parse RDF data from bytes in specified format
pub fn parse_triples(
    data: &[u8],
    format: RdfFormat,
) -> Result<Vec<oxirs_core::model::Triple>, GspError> {
    use oxirs_core::parser::Parser;

    let parser = Parser::new(format);
    let triples = parser
        .parse_bytes_to_quads(data)
        .map_err(|e| GspError::ParseError(format!("Parse error: {}", e)))?
        .into_iter()
        .map(|quad| quad.to_triple())
        .collect();

    Ok(triples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negotiate_format_simple() {
        let result = negotiate_format(Some("text/turtle"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RdfFormat::Turtle);
    }

    #[test]
    fn test_negotiate_format_with_quality() {
        let accept = "application/rdf+xml;q=0.5, text/turtle;q=1.0, application/n-triples;q=0.8";
        let result = negotiate_format(Some(accept));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RdfFormat::Turtle);
    }

    #[test]
    fn test_negotiate_format_wildcard() {
        let result = negotiate_format(Some("*/*"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RdfFormat::Turtle);
    }

    #[test]
    fn test_negotiate_format_none() {
        let result = negotiate_format(None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RdfFormat::Turtle);
    }

    #[test]
    fn test_parse_content_type() {
        let result = parse_content_type(Some("text/turtle"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RdfFormat::Turtle);

        let result = parse_content_type(Some("application/n-triples; charset=utf-8"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), RdfFormat::NTriples);
    }

    #[test]
    fn test_parse_content_type_no_header() {
        let result = parse_content_type(None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GspError::BadRequest(_)));
    }

    #[test]
    fn test_parse_accept_header() {
        let types = parse_accept_header("text/turtle;q=0.9, application/rdf+xml;q=0.5");
        assert_eq!(types.len(), 2);
        assert_eq!(types[0].0, "text/turtle");
        assert_eq!(types[0].1, 0.9);
        assert_eq!(types[1].0, "application/rdf+xml");
        assert_eq!(types[1].1, 0.5);
    }
}
