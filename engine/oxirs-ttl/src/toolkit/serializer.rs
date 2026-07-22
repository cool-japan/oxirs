//! Generic serializer framework for writing RDF elements to streams
//!
//! This module provides the serialization infrastructure for converting
//! RDF triples and quads back to text formats.

use crate::error::TurtleResult;
// use oxirs_core::model::{Quad, Triple};
use std::io::Write;

/// A generic serializer trait for RDF formats
pub trait Serializer<Input> {
    /// Serialize to a writer
    fn serialize<W: Write>(&self, input: &[Input], writer: W) -> TurtleResult<()>;

    /// Serialize a single item
    fn serialize_item<W: Write>(&self, input: &Input, writer: W) -> TurtleResult<()>;
}

/// Async serializer trait for Tokio integration
#[cfg(feature = "async-tokio")]
pub trait AsyncSerializer<Input> {
    /// Serialize to an async writer
    fn serialize_async<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        input: &[Input],
        writer: W,
    ) -> impl std::future::Future<Output = TurtleResult<()>> + Send;

    /// Serialize a single item async
    fn serialize_item_async<W: tokio::io::AsyncWrite + Unpin>(
        &self,
        input: &Input,
        writer: W,
    ) -> impl std::future::Future<Output = TurtleResult<()>> + Send;
}

/// Configuration for serialization
#[derive(Debug, Clone)]
pub struct SerializationConfig {
    /// Whether to use pretty printing (with indentation and spacing)
    pub pretty: bool,
    /// Base IRI for relative IRI generation
    pub base_iri: Option<String>,
    /// Prefix declarations to use
    pub prefixes: std::collections::HashMap<String, String>,
    /// Whether to use prefix abbreviations
    pub use_prefixes: bool,
    /// Maximum line length for formatting
    pub max_line_length: Option<usize>,
    /// Indentation string (typically spaces or tabs)
    pub indent: String,
    /// Whether to normalize IRIs per RFC 3987 during serialization
    ///
    /// When enabled, IRIs are normalized to canonical form for consistent output:
    /// - Case normalization (scheme and host to lowercase)
    /// - Percent-encoding normalization (decode unreserved characters)
    /// - Path normalization (remove dot segments)
    /// - Default port removal (http:80, https:443, etc.)
    ///
    /// This helps ensure consistent IRI representation across different systems.
    pub normalize_iris: bool,
}

impl Default for SerializationConfig {
    fn default() -> Self {
        Self {
            pretty: true,
            base_iri: None,
            prefixes: std::collections::HashMap::new(),
            use_prefixes: true,
            max_line_length: Some(80),
            indent: "  ".to_string(),
            normalize_iris: false, // Disabled by default for backward compatibility
        }
    }
}

impl SerializationConfig {
    /// Create a new serialization config
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable pretty printing
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Set the base IRI
    pub fn with_base_iri(mut self, base_iri: String) -> Self {
        self.base_iri = Some(base_iri);
        self
    }

    /// Add a prefix declaration
    pub fn with_prefix(mut self, prefix: String, iri: String) -> Self {
        self.prefixes.insert(prefix, iri);
        self
    }

    /// Set whether to use prefix abbreviations
    pub fn with_use_prefixes(mut self, use_prefixes: bool) -> Self {
        self.use_prefixes = use_prefixes;
        self
    }

    /// Set the maximum line length
    pub fn with_max_line_length(mut self, max_length: Option<usize>) -> Self {
        self.max_line_length = max_length;
        self
    }

    /// Set the indentation string
    pub fn with_indent(mut self, indent: String) -> Self {
        self.indent = indent;
        self
    }

    /// Enable or disable IRI normalization
    ///
    /// When enabled, all IRIs in serialized output are normalized per RFC 3987
    /// for consistent canonical representation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_ttl::toolkit::SerializationConfig;
    ///
    /// let config = SerializationConfig::new()
    ///     .with_normalize_iris(true);
    ///
    /// assert!(config.normalize_iris);
    /// ```
    pub fn with_normalize_iris(mut self, normalize: bool) -> Self {
        self.normalize_iris = normalize;
        self
    }
}

// ─── Turtle PN_LOCAL grammar helpers ─────────────────────────────────────────
//
// Per the Turtle 1.1 / SPARQL grammar:
//   PN_CHARS_BASE ::= [A-Z] | [a-z] | [#x00C0-#x00D6] | [#x00D8-#x00F6]
//                   | [#x00F8-#x02FF] | [#x0370-#x037D] | [#x037F-#x1FFF]
//                   | [#x200C-#x200D] | [#x2070-#x218F] | [#x2C00-#x2FEF]
//                   | [#x3001-#xD7FF] | [#xF900-#xFDCF] | [#xFDF0-#xFFFD]
//                   | [#x10000-#xEFFFF]
//   PN_CHARS_U    ::= PN_CHARS_BASE | '_'
//   PN_CHARS      ::= PN_CHARS_U | '-' | [0-9] | #x00B7
//                   | [#x0300-#x036F] | [#x203F-#x2040]
//   PN_LOCAL_ESC  ::= '\' ('_' | '~' | '.' | '-' | '!' | '$' | '&' | "'"
//                   | '(' | ')' | '*' | '+' | ',' | ';' | '=' | '/' | '?'
//                   | '#' | '@' | '%')
//   PLX           ::= PERCENT | PN_LOCAL_ESC
//   PN_LOCAL      ::= (PN_CHARS_U | ':' | [0-9] | PLX)
//                     ((PN_CHARS | '.' | ':' | PLX)* (PN_CHARS | ':' | PLX))?

fn is_pn_chars_base(c: char) -> bool {
    matches!(c,
        'A'..='Z' | 'a'..='z' |
        '\u{00C0}'..='\u{00D6}' | '\u{00D8}'..='\u{00F6}' | '\u{00F8}'..='\u{02FF}' |
        '\u{0370}'..='\u{037D}' | '\u{037F}'..='\u{1FFF}' |
        '\u{200C}'..='\u{200D}' | '\u{2070}'..='\u{218F}' |
        '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}' |
        '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}' |
        '\u{10000}'..='\u{EFFFF}'
    )
}

fn is_pn_chars_u(c: char) -> bool {
    is_pn_chars_base(c) || c == '_'
}

fn is_pn_chars(c: char) -> bool {
    is_pn_chars_u(c)
        || c == '-'
        || c.is_ascii_digit()
        || c == '\u{00B7}'
        || ('\u{0300}'..='\u{036F}').contains(&c)
        || ('\u{203F}'..='\u{2040}').contains(&c)
}

/// Characters that `PN_LOCAL_ESC` permits escaping with a leading backslash.
fn is_pn_local_esc_char(c: char) -> bool {
    matches!(
        c,
        '_' | '~'
            | '.'
            | '-'
            | '!'
            | '$'
            | '&'
            | '\''
            | '('
            | ')'
            | '*'
            | '+'
            | ','
            | ';'
            | '='
            | '/'
            | '?'
            | '#'
            | '@'
            | '%'
    )
}

/// Attempt to render `local` — the portion of an IRI following a matched
/// namespace prefix — as a syntactically legal Turtle `PN_LOCAL`, escaping
/// any `PN_LOCAL_ESC`-eligible character with a backslash where required by
/// its position. Returns `None` when `local` contains a character (or an
/// invalid `%` escape) that cannot legally appear in, or be escaped into, a
/// `PN_LOCAL` — the caller must then fall back to the full `<iri>` form
/// rather than emit a broken prefixed name.
fn escape_pn_local(local: &str) -> Option<String> {
    if local.is_empty() {
        // `prefix:` with an empty local part is legal.
        return Some(String::new());
    }

    let chars: Vec<char> = local.chars().collect();
    let n = chars.len();
    let mut result = String::with_capacity(local.len());
    let mut i = 0;

    while i < n {
        let c = chars[i];
        let is_first = i == 0;

        if c == '%' {
            // Only a well-formed PERCENT triplet ('%' HEX HEX) may pass
            // through unescaped; anything else cannot be represented.
            if i + 2 < n && chars[i + 1].is_ascii_hexdigit() && chars[i + 2].is_ascii_hexdigit() {
                result.push('%');
                result.push(chars[i + 1]);
                result.push(chars[i + 2]);
                i += 3;
                continue;
            }
            return None;
        }

        let allowed_unescaped = if is_first {
            is_pn_chars_u(c) || c.is_ascii_digit() || c == ':'
        } else {
            is_pn_chars(c) || c == ':' || c == '.'
        };

        if allowed_unescaped {
            result.push(c);
            i += 1;
            continue;
        }

        if is_pn_local_esc_char(c) {
            result.push('\\');
            result.push(c);
            i += 1;
            continue;
        }

        // No legal unescaped or escaped representation exists for this
        // character (e.g. '<', '>', '"', whitespace, control characters).
        return None;
    }

    // The final character of a multi-character PN_LOCAL must not be an
    // unescaped '.': only PN_CHARS | ':' | PLX are permitted there. If the
    // loop above emitted a raw trailing '.', escape it now.
    if result.ends_with('.') && !result.ends_with("\\.") {
        result.pop();
        result.push('\\');
        result.push('.');
    }

    Some(result)
}

/// Helper for writing formatted output
pub struct FormattedWriter<W: Write> {
    writer: W,
    config: SerializationConfig,
    current_line_length: usize,
    indent_level: usize,
}

impl<W: Write> FormattedWriter<W> {
    /// Create a new formatted writer
    pub fn new(writer: W, config: SerializationConfig) -> Self {
        Self {
            writer,
            config,
            current_line_length: 0,
            indent_level: 0,
        }
    }

    /// Write a string, handling line breaks and indentation
    pub fn write_str(&mut self, s: &str) -> std::io::Result<()> {
        if self.config.pretty {
            // Check if we need to break the line
            if let Some(max_len) = self.config.max_line_length {
                if self.current_line_length + s.len() > max_len && self.current_line_length > 0 {
                    self.write_newline()?;
                }
            }
        }

        self.writer.write_all(s.as_bytes())?;
        self.current_line_length += s.len();
        Ok(())
    }

    /// Write a newline and appropriate indentation
    pub fn write_newline(&mut self) -> std::io::Result<()> {
        self.writer.write_all(b"\n")?;
        self.current_line_length = 0;

        if self.config.pretty {
            for _ in 0..self.indent_level {
                self.writer.write_all(self.config.indent.as_bytes())?;
                self.current_line_length += self.config.indent.len();
            }
        }
        Ok(())
    }

    /// Increase indentation level
    pub fn increase_indent(&mut self) {
        self.indent_level += 1;
    }

    /// Decrease indentation level
    pub fn decrease_indent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level -= 1;
        }
    }

    /// Write a space if pretty printing is enabled
    pub fn write_space(&mut self) -> std::io::Result<()> {
        if self.config.pretty {
            self.write_str(" ")
        } else {
            Ok(())
        }
    }

    /// Abbreviate an IRI using prefixes if possible
    ///
    /// An IRI is only compacted to `prefix:local` when the portion of the
    /// IRI after the matched namespace can be legally represented as a
    /// Turtle `PN_LOCAL` (escaping any `PN_LOCAL_ESC`-eligible character as
    /// needed). If no registered prefix yields a legal `PN_LOCAL` — e.g.
    /// because the local part contains a character such as `<`, `>`, `"`,
    /// whitespace, or a control character that cannot appear in, or be
    /// escaped into, a prefixed name — the full `<iri>` form is emitted
    /// instead so the output always remains syntactically valid Turtle that
    /// round-trips through a conformant parser.
    pub fn abbreviate_iri(&self, iri: &str) -> String {
        if !self.config.use_prefixes {
            return format!("<{iri}>");
        }

        // Consider every registered namespace prefix that matches, preferring
        // the longest (most specific) match first, and only accept a match
        // whose local part can be legally escaped into a PN_LOCAL.
        let mut candidates: Vec<(&String, &String)> = self
            .config
            .prefixes
            .iter()
            .filter(|(_, prefix_iri)| {
                !prefix_iri.is_empty() && iri.starts_with(prefix_iri.as_str())
            })
            .collect();
        candidates.sort_by_key(|(_, prefix_iri)| std::cmp::Reverse(prefix_iri.len()));

        for (prefix, prefix_iri) in candidates {
            let local = &iri[prefix_iri.len()..];
            if let Some(escaped_local) = escape_pn_local(local) {
                return format!("{prefix}:{escaped_local}");
            }
        }

        // Try relative IRI if base is set
        if let Some(ref base) = self.config.base_iri {
            if iri.starts_with(base) {
                let relative = &iri[base.len()..];
                return format!("<{relative}>");
            }
        }

        format!("<{iri}>")
    }

    /// Escape a string literal
    pub fn escape_string(&self, s: &str) -> String {
        let mut result = String::with_capacity(s.len() + 2);
        result.push('"');

        for ch in s.chars() {
            match ch {
                '"' => result.push_str("\\\""),
                '\\' => result.push_str("\\\\"),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\t' => result.push_str("\\t"),
                c if c.is_control() => {
                    result.push_str(&format!("\\u{:04X}", c as u32));
                }
                c => result.push(c),
            }
        }

        result.push('"');
        result
    }

    /// Get the underlying writer
    pub fn into_inner(self) -> W {
        self.writer
    }
}

impl<W: Write> Write for FormattedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let s = std::str::from_utf8(buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.write_str(s)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use std::io::Cursor;

    fn writer_with_prefix(prefix: &str, ns: &str) -> FormattedWriter<Cursor<Vec<u8>>> {
        let config = SerializationConfig::new().with_prefix(prefix.to_string(), ns.to_string());
        FormattedWriter::new(Cursor::new(Vec::new()), config)
    }

    #[test]
    fn regression_abbreviate_iri_escapes_parentheses_in_local_part() {
        let w = writer_with_prefix("ex", "http://example.org/");
        let out = w.abbreviate_iri("http://example.org/page(disambiguation)");
        assert_eq!(out, "ex:page\\(disambiguation\\)");
    }

    #[test]
    fn regression_abbreviate_iri_falls_back_to_full_iri_for_illegal_local() {
        // '<' cannot legally appear in, or be escaped into, a PN_LOCAL.
        let w = writer_with_prefix("ex", "http://example.org/");
        let out = w.abbreviate_iri("http://example.org/a<b");
        assert_eq!(out, "<http://example.org/a<b>");
    }

    #[test]
    fn regression_abbreviate_iri_escapes_extra_slash_in_local_part() {
        let w = writer_with_prefix("ex", "http://example.org/");
        let out = w.abbreviate_iri("http://example.org/a/b");
        assert_eq!(out, "ex:a\\/b");
    }

    #[test]
    fn regression_abbreviate_iri_escapes_trailing_dot() {
        let w = writer_with_prefix("ex", "http://example.org/");
        let out = w.abbreviate_iri("http://example.org/v1.0.");
        assert_eq!(out, "ex:v1.0\\.");
    }

    #[test]
    fn regression_abbreviate_iri_plain_local_unchanged() {
        let w = writer_with_prefix("ex", "http://example.org/");
        let out = w.abbreviate_iri("http://example.org/alice");
        assert_eq!(out, "ex:alice");
    }

    #[test]
    fn regression_abbreviate_iri_escapes_every_reserved_char() {
        let w = writer_with_prefix("ex", "http://example.org/");
        let out = w.abbreviate_iri("http://example.org/page(disambiguation),v2");
        assert_eq!(out, "ex:page\\(disambiguation\\)\\,v2");
    }
}
