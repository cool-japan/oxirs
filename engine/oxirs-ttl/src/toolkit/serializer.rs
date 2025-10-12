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
    pub fn abbreviate_iri(&self, iri: &str) -> String {
        if !self.config.use_prefixes {
            return format!("<{iri}>");
        }

        // Try to find a matching prefix
        for (prefix, prefix_iri) in &self.config.prefixes {
            if iri.starts_with(prefix_iri) {
                let local = &iri[prefix_iri.len()..];
                return format!("{prefix}:{local}");
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
