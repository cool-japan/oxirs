//! HTTP content negotiation for RDF format selection.
//!
//! Parses `Accept` headers, selects the best RDF serialization format based on
//! quality values, and generates appropriate `Content-Type` response headers.

use std::collections::HashMap;
use std::fmt;

/// Supported RDF serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RdfFormat {
    Turtle,
    NTriples,
    JsonLd,
    RdfXml,
    NQuads,
    TriG,
}

impl RdfFormat {
    /// Primary MIME type string.
    pub fn media_type(&self) -> &'static str {
        match self {
            RdfFormat::Turtle => "text/turtle",
            RdfFormat::NTriples => "application/n-triples",
            RdfFormat::JsonLd => "application/ld+json",
            RdfFormat::RdfXml => "application/rdf+xml",
            RdfFormat::NQuads => "application/n-quads",
            RdfFormat::TriG => "application/trig",
        }
    }

    /// Common file extension for the format.
    pub fn extension(&self) -> &'static str {
        match self {
            RdfFormat::Turtle => "ttl",
            RdfFormat::NTriples => "nt",
            RdfFormat::JsonLd => "jsonld",
            RdfFormat::RdfXml => "rdf",
            RdfFormat::NQuads => "nq",
            RdfFormat::TriG => "trig",
        }
    }

    /// All supported formats in default preference order.
    pub fn all() -> &'static [RdfFormat] {
        &[
            RdfFormat::Turtle,
            RdfFormat::NTriples,
            RdfFormat::JsonLd,
            RdfFormat::RdfXml,
            RdfFormat::NQuads,
            RdfFormat::TriG,
        ]
    }
}

impl fmt::Display for RdfFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.media_type())
    }
}

/// Errors during content negotiation.
#[derive(Debug, Clone, PartialEq)]
pub enum NegotiationError {
    /// No acceptable format found.
    NotAcceptable(String),
    /// Malformed Accept header.
    MalformedHeader(String),
    /// Unknown file extension.
    UnknownExtension(String),
}

impl fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NegotiationError::NotAcceptable(msg) => write!(f, "Not Acceptable: {msg}"),
            NegotiationError::MalformedHeader(msg) => write!(f, "Malformed header: {msg}"),
            NegotiationError::UnknownExtension(ext) => write!(f, "Unknown extension: {ext}"),
        }
    }
}

impl std::error::Error for NegotiationError {}

/// A single media range from an `Accept` header.
///
/// E.g., `text/turtle;q=0.9;charset=utf-8`
#[derive(Debug, Clone, PartialEq)]
pub struct MediaRange {
    /// Main type (e.g., `"text"`, `"application"`, `"*"`).
    pub main_type: String,
    /// Sub-type (e.g., `"turtle"`, `"ld+json"`, `"*"`).
    pub sub_type: String,
    /// Quality value in `[0.0, 1.0]`.
    pub quality: f64,
    /// Additional parameters (e.g., `charset=utf-8`).
    pub params: HashMap<String, String>,
}

impl MediaRange {
    /// Does this range match a concrete media type string?
    pub fn matches(&self, media_type: &str) -> bool {
        if self.main_type == "*" && self.sub_type == "*" {
            return true;
        }
        let parts: Vec<&str> = media_type.splitn(2, '/').collect();
        if parts.len() != 2 {
            return false;
        }
        let mt = parts[0];
        let st = parts[1];
        if self.main_type == "*" {
            return self.sub_type == st || self.sub_type == "*";
        }
        if self.main_type != mt {
            return false;
        }
        self.sub_type == "*" || self.sub_type == st
    }

    /// Specificity for tie-breaking: 3 = exact, 2 = sub-wildcard, 1 = type-wildcard, 0 = full wildcard.
    pub fn specificity(&self) -> u8 {
        if self.main_type == "*" && self.sub_type == "*" {
            0
        } else if self.sub_type == "*" || self.main_type == "*" {
            1
        } else {
            2
        }
    }
}

/// Parse an `Accept` header value into a list of [`MediaRange`]s sorted by
/// quality (descending), then specificity (descending).
pub fn parse_accept(header: &str) -> Result<Vec<MediaRange>, NegotiationError> {
    let mut ranges = Vec::new();
    for part in header.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let range = parse_single_media_range(part)?;
        ranges.push(range);
    }
    // Stable sort: higher quality first, then higher specificity first.
    ranges.sort_by(|a, b| {
        b.quality
            .partial_cmp(&a.quality)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.specificity().cmp(&a.specificity()))
    });
    Ok(ranges)
}

/// Parse a single media range token (e.g., `"text/turtle;q=0.9;charset=utf-8"`).
fn parse_single_media_range(input: &str) -> Result<MediaRange, NegotiationError> {
    let mut segments = input.split(';').map(str::trim);
    let type_part = segments
        .next()
        .ok_or_else(|| NegotiationError::MalformedHeader("empty media range".into()))?;

    let (main_type, sub_type) = if let Some((m, s)) = type_part.split_once('/') {
        (m.trim().to_ascii_lowercase(), s.trim().to_ascii_lowercase())
    } else {
        return Err(NegotiationError::MalformedHeader(format!(
            "missing '/' in media range: {type_part}"
        )));
    };

    let mut quality: f64 = 1.0;
    let mut params = HashMap::new();

    for seg in segments {
        if let Some((key, value)) = seg.split_once('=') {
            let key = key.trim().to_ascii_lowercase();
            let value = value.trim();
            if key == "q" {
                quality = value.parse::<f64>().map_err(|_| {
                    NegotiationError::MalformedHeader(format!("invalid quality value: {value}"))
                })?;
                quality = quality.clamp(0.0, 1.0);
            } else {
                params.insert(key, value.to_string());
            }
        }
    }

    Ok(MediaRange {
        main_type,
        sub_type,
        quality,
        params,
    })
}

/// Content negotiation configuration.
#[derive(Debug, Clone)]
pub struct NegotiationConfig {
    /// Available formats the server can produce.
    pub available_formats: Vec<RdfFormat>,
    /// Default format when no `Accept` header is present.
    pub default_format: RdfFormat,
    /// Default charset to advertise.
    pub default_charset: String,
}

impl Default for NegotiationConfig {
    fn default() -> Self {
        Self {
            available_formats: RdfFormat::all().to_vec(),
            default_format: RdfFormat::Turtle,
            default_charset: "utf-8".to_string(),
        }
    }
}

/// Content negotiator that selects the best RDF format.
pub struct ContentNegotiator {
    config: NegotiationConfig,
}

impl ContentNegotiator {
    /// Create a negotiator with default settings (all formats, Turtle default).
    pub fn new() -> Self {
        Self {
            config: NegotiationConfig::default(),
        }
    }

    /// Create a negotiator from explicit configuration.
    pub fn with_config(config: NegotiationConfig) -> Self {
        Self { config }
    }

    /// Select the best format from an `Accept` header value.
    ///
    /// Returns `Ok(format)` for the best match, or `Err` if no acceptable format exists.
    pub fn negotiate(&self, accept_header: Option<&str>) -> Result<RdfFormat, NegotiationError> {
        let header = match accept_header {
            Some(h) if !h.trim().is_empty() => h,
            _ => return Ok(self.config.default_format),
        };

        let ranges = parse_accept(header)?;
        if ranges.is_empty() {
            return Ok(self.config.default_format);
        }

        // Check each range against available formats in priority order.
        for range in &ranges {
            if range.quality <= 0.0 {
                continue;
            }
            for fmt in &self.config.available_formats {
                if range.matches(fmt.media_type()) {
                    return Ok(*fmt);
                }
            }
        }

        Err(NegotiationError::NotAcceptable(format!(
            "None of the requested formats are available: {header}"
        )))
    }

    /// Generate a `Content-Type` header value for a given format.
    pub fn content_type_header(&self, format: RdfFormat) -> String {
        format!(
            "{}; charset={}",
            format.media_type(),
            self.config.default_charset
        )
    }

    /// Generate a `Content-Type` header with a custom charset.
    pub fn content_type_with_charset(&self, format: RdfFormat, charset: &str) -> String {
        format!("{}; charset={charset}", format.media_type())
    }

    /// Map a file extension to an RDF format.
    pub fn format_from_extension(ext: &str) -> Result<RdfFormat, NegotiationError> {
        let ext_lower = ext.trim_start_matches('.').to_ascii_lowercase();
        match ext_lower.as_str() {
            "ttl" | "turtle" => Ok(RdfFormat::Turtle),
            "nt" | "ntriples" => Ok(RdfFormat::NTriples),
            "jsonld" | "json-ld" => Ok(RdfFormat::JsonLd),
            "rdf" | "xml" | "rdfxml" => Ok(RdfFormat::RdfXml),
            "nq" | "nquads" => Ok(RdfFormat::NQuads),
            "trig" => Ok(RdfFormat::TriG),
            other => Err(NegotiationError::UnknownExtension(other.to_string())),
        }
    }

    /// Map an RDF format to a file extension (without the leading dot).
    pub fn extension_for_format(format: RdfFormat) -> &'static str {
        format.extension()
    }

    /// Return the default format.
    pub fn default_format(&self) -> RdfFormat {
        self.config.default_format
    }

    /// Return the list of available formats.
    pub fn available_formats(&self) -> &[RdfFormat] {
        &self.config.available_formats
    }
}

impl Default for ContentNegotiator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Accept header parsing --

    #[test]
    fn test_parse_simple_accept() {
        let ranges = parse_accept("text/turtle").expect("parse");
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].main_type, "text");
        assert_eq!(ranges[0].sub_type, "turtle");
        assert!((ranges[0].quality - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_accept_with_quality() {
        let ranges = parse_accept("application/ld+json;q=0.8").expect("parse");
        assert_eq!(ranges.len(), 1);
        assert!((ranges[0].quality - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_parse_multiple_accept_values() {
        let ranges = parse_accept(
            "text/turtle;q=0.9, application/ld+json;q=1.0, application/n-triples;q=0.5",
        )
        .expect("parse");
        assert_eq!(ranges.len(), 3);
        // Sorted by quality descending
        assert_eq!(ranges[0].sub_type, "ld+json");
        assert_eq!(ranges[1].sub_type, "turtle");
        assert_eq!(ranges[2].sub_type, "n-triples");
    }

    #[test]
    fn test_parse_wildcard() {
        let ranges = parse_accept("*/*").expect("parse");
        assert_eq!(ranges[0].main_type, "*");
        assert_eq!(ranges[0].sub_type, "*");
    }

    #[test]
    fn test_parse_subtype_wildcard() {
        let ranges = parse_accept("text/*;q=0.5").expect("parse");
        assert_eq!(ranges[0].main_type, "text");
        assert_eq!(ranges[0].sub_type, "*");
    }

    #[test]
    fn test_parse_with_charset_param() {
        let ranges = parse_accept("text/turtle;charset=utf-8;q=0.9").expect("parse");
        assert_eq!(ranges[0].params.get("charset"), Some(&"utf-8".to_string()));
        assert!((ranges[0].quality - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_parse_empty_header() {
        let ranges = parse_accept("").expect("parse");
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_parse_malformed_no_slash() {
        let err = parse_accept("textturtle").expect_err("malformed");
        assert!(matches!(err, NegotiationError::MalformedHeader(_)));
    }

    #[test]
    fn test_parse_quality_clamped_above_one() {
        let ranges = parse_accept("text/turtle;q=1.5").expect("parse");
        assert!((ranges[0].quality - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_quality_clamped_below_zero() {
        let ranges = parse_accept("text/turtle;q=-0.5").expect("parse");
        assert!((ranges[0].quality - 0.0).abs() < f64::EPSILON);
    }

    // -- MediaRange matching --

    #[test]
    fn test_media_range_exact_match() {
        let range = MediaRange {
            main_type: "text".into(),
            sub_type: "turtle".into(),
            quality: 1.0,
            params: HashMap::new(),
        };
        assert!(range.matches("text/turtle"));
        assert!(!range.matches("application/ld+json"));
    }

    #[test]
    fn test_media_range_wildcard_match() {
        let range = MediaRange {
            main_type: "*".into(),
            sub_type: "*".into(),
            quality: 1.0,
            params: HashMap::new(),
        };
        assert!(range.matches("text/turtle"));
        assert!(range.matches("application/n-triples"));
    }

    #[test]
    fn test_media_range_subtype_wildcard() {
        let range = MediaRange {
            main_type: "application".into(),
            sub_type: "*".into(),
            quality: 1.0,
            params: HashMap::new(),
        };
        assert!(range.matches("application/ld+json"));
        assert!(range.matches("application/n-triples"));
        assert!(!range.matches("text/turtle"));
    }

    #[test]
    fn test_specificity_ordering() {
        let exact = MediaRange {
            main_type: "text".into(),
            sub_type: "turtle".into(),
            quality: 1.0,
            params: HashMap::new(),
        };
        let sub_wild = MediaRange {
            main_type: "text".into(),
            sub_type: "*".into(),
            quality: 1.0,
            params: HashMap::new(),
        };
        let full_wild = MediaRange {
            main_type: "*".into(),
            sub_type: "*".into(),
            quality: 1.0,
            params: HashMap::new(),
        };
        assert!(exact.specificity() > sub_wild.specificity());
        assert!(sub_wild.specificity() > full_wild.specificity());
    }

    // -- Format negotiation --

    #[test]
    fn test_negotiate_turtle_preferred() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(Some("text/turtle")).expect("neg");
        assert_eq!(fmt, RdfFormat::Turtle);
    }

    #[test]
    fn test_negotiate_jsonld() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(Some("application/ld+json")).expect("neg");
        assert_eq!(fmt, RdfFormat::JsonLd);
    }

    #[test]
    fn test_negotiate_ntriples() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(Some("application/n-triples")).expect("neg");
        assert_eq!(fmt, RdfFormat::NTriples);
    }

    #[test]
    fn test_negotiate_rdfxml() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(Some("application/rdf+xml")).expect("neg");
        assert_eq!(fmt, RdfFormat::RdfXml);
    }

    #[test]
    fn test_negotiate_nquads() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(Some("application/n-quads")).expect("neg");
        assert_eq!(fmt, RdfFormat::NQuads);
    }

    #[test]
    fn test_negotiate_trig() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(Some("application/trig")).expect("neg");
        assert_eq!(fmt, RdfFormat::TriG);
    }

    #[test]
    fn test_negotiate_wildcard_returns_default() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(Some("*/*")).expect("neg");
        // Wildcard matches first available, which is Turtle by default
        assert_eq!(fmt, RdfFormat::Turtle);
    }

    #[test]
    fn test_negotiate_no_header_returns_default() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(None).expect("neg");
        assert_eq!(fmt, RdfFormat::Turtle);
    }

    #[test]
    fn test_negotiate_empty_header_returns_default() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(Some("")).expect("neg");
        assert_eq!(fmt, RdfFormat::Turtle);
    }

    #[test]
    fn test_negotiate_not_acceptable() {
        let neg = ContentNegotiator::new();
        let err = neg
            .negotiate(Some("image/png"))
            .expect_err("not acceptable");
        assert!(matches!(err, NegotiationError::NotAcceptable(_)));
    }

    #[test]
    fn test_negotiate_quality_zero_excluded() {
        let neg = ContentNegotiator::new();
        // text/turtle excluded, application/ld+json available
        let fmt = neg
            .negotiate(Some("text/turtle;q=0, application/ld+json;q=1.0"))
            .expect("neg");
        assert_eq!(fmt, RdfFormat::JsonLd);
    }

    #[test]
    fn test_negotiate_highest_quality_wins() {
        let neg = ContentNegotiator::new();
        let fmt = neg
            .negotiate(Some(
                "application/n-triples;q=0.3, text/turtle;q=0.8, application/ld+json;q=0.9",
            ))
            .expect("neg");
        assert_eq!(fmt, RdfFormat::JsonLd);
    }

    #[test]
    fn test_negotiate_custom_default() {
        let cfg = NegotiationConfig {
            default_format: RdfFormat::JsonLd,
            ..Default::default()
        };
        let neg = ContentNegotiator::with_config(cfg);
        let fmt = neg.negotiate(None).expect("neg");
        assert_eq!(fmt, RdfFormat::JsonLd);
    }

    #[test]
    fn test_negotiate_limited_formats() {
        let cfg = NegotiationConfig {
            available_formats: vec![RdfFormat::Turtle, RdfFormat::NTriples],
            ..Default::default()
        };
        let neg = ContentNegotiator::with_config(cfg);
        let err = neg
            .negotiate(Some("application/ld+json"))
            .expect_err("limited");
        assert!(matches!(err, NegotiationError::NotAcceptable(_)));
    }

    // -- Content-Type header generation --

    #[test]
    fn test_content_type_header_turtle() {
        let neg = ContentNegotiator::new();
        let ct = neg.content_type_header(RdfFormat::Turtle);
        assert_eq!(ct, "text/turtle; charset=utf-8");
    }

    #[test]
    fn test_content_type_header_jsonld() {
        let neg = ContentNegotiator::new();
        let ct = neg.content_type_header(RdfFormat::JsonLd);
        assert_eq!(ct, "application/ld+json; charset=utf-8");
    }

    #[test]
    fn test_content_type_custom_charset() {
        let neg = ContentNegotiator::new();
        let ct = neg.content_type_with_charset(RdfFormat::RdfXml, "iso-8859-1");
        assert_eq!(ct, "application/rdf+xml; charset=iso-8859-1");
    }

    // -- File extension mapping --

    #[test]
    fn test_extension_to_format_ttl() {
        let fmt = ContentNegotiator::format_from_extension("ttl").expect("ttl");
        assert_eq!(fmt, RdfFormat::Turtle);
    }

    #[test]
    fn test_extension_to_format_with_dot() {
        let fmt = ContentNegotiator::format_from_extension(".jsonld").expect("jsonld");
        assert_eq!(fmt, RdfFormat::JsonLd);
    }

    #[test]
    fn test_extension_to_format_nt() {
        let fmt = ContentNegotiator::format_from_extension("nt").expect("nt");
        assert_eq!(fmt, RdfFormat::NTriples);
    }

    #[test]
    fn test_extension_to_format_nq() {
        let fmt = ContentNegotiator::format_from_extension("nq").expect("nq");
        assert_eq!(fmt, RdfFormat::NQuads);
    }

    #[test]
    fn test_extension_to_format_trig() {
        let fmt = ContentNegotiator::format_from_extension("trig").expect("trig");
        assert_eq!(fmt, RdfFormat::TriG);
    }

    #[test]
    fn test_extension_to_format_rdf() {
        let fmt = ContentNegotiator::format_from_extension("rdf").expect("rdf");
        assert_eq!(fmt, RdfFormat::RdfXml);
    }

    #[test]
    fn test_extension_to_format_unknown() {
        let err = ContentNegotiator::format_from_extension("csv").expect_err("unknown");
        assert!(matches!(err, NegotiationError::UnknownExtension(_)));
    }

    #[test]
    fn test_format_to_extension() {
        assert_eq!(
            ContentNegotiator::extension_for_format(RdfFormat::Turtle),
            "ttl"
        );
        assert_eq!(
            ContentNegotiator::extension_for_format(RdfFormat::NQuads),
            "nq"
        );
    }

    // -- Misc --

    #[test]
    fn test_rdf_format_display() {
        assert_eq!(format!("{}", RdfFormat::Turtle), "text/turtle");
        assert_eq!(format!("{}", RdfFormat::JsonLd), "application/ld+json");
    }

    #[test]
    fn test_all_formats_count() {
        assert_eq!(RdfFormat::all().len(), 6);
    }

    #[test]
    fn test_default_negotiator() {
        let neg = ContentNegotiator::default();
        assert_eq!(neg.default_format(), RdfFormat::Turtle);
        assert_eq!(neg.available_formats().len(), 6);
    }

    #[test]
    fn test_negotiation_error_display() {
        let err = NegotiationError::NotAcceptable("test".into());
        assert!(format!("{err}").contains("Not Acceptable"));
        let err2 = NegotiationError::MalformedHeader("bad".into());
        assert!(format!("{err2}").contains("Malformed"));
        let err3 = NegotiationError::UnknownExtension("xyz".into());
        assert!(format!("{err3}").contains("Unknown extension"));
    }

    #[test]
    fn test_media_range_no_slash_input() {
        // Already tested via parse_accept, but let's directly test specificity = 0
        let range = MediaRange {
            main_type: "*".into(),
            sub_type: "*".into(),
            quality: 0.5,
            params: HashMap::new(),
        };
        assert_eq!(range.specificity(), 0);
    }

    #[test]
    fn test_negotiate_subtype_wildcard_selects_first_match() {
        let neg = ContentNegotiator::new();
        let fmt = neg.negotiate(Some("application/*")).expect("neg");
        // First available application/* is N-Triples (second in default list)
        assert_eq!(fmt, RdfFormat::NTriples);
    }
}
