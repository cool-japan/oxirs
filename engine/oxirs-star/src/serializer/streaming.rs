//! Streaming serialization for memory-efficient processing

use super::config::{SerializationContext, StreamingConfig};
use super::parallel::ChunkedIterator;
use super::star_serializer::StarSerializer;
use crate::model::{StarTerm, StarTriple};
use crate::parser::StarFormat;
use crate::{StarError, StarResult};
use std::io::Write;
use std::sync::{Arc, Mutex};

/// Streaming serializer for memory-efficient processing of large graphs
pub struct StreamingSerializer<W: Write> {
    writer: Arc<Mutex<W>>,
    config: StreamingConfig,
    context: SerializationContext,
    buffer: Vec<u8>,
    written_bytes: usize,
}

impl<W: Write> StreamingSerializer<W> {
    /// Create a new streaming serializer
    pub fn new(writer: W, config: StreamingConfig) -> Self {
        Self {
            writer: Arc::new(Mutex::new(writer)),
            buffer: Vec::with_capacity(config.buffer_capacity),
            config,
            context: SerializationContext::new(),
            written_bytes: 0,
        }
    }

    /// Write data to the output stream with buffering
    fn write_buffered(&mut self, data: &[u8]) -> StarResult<()> {
        if self.config.enable_buffering {
            self.buffer.extend_from_slice(data);

            // Flush if buffer is full or memory threshold reached
            if self.buffer.len() >= self.config.buffer_capacity
                || self.written_bytes >= self.config.memory_threshold
            {
                self.flush_buffer()?;
            }
        } else {
            // Direct write without buffering
            let mut writer = self
                .writer
                .lock()
                .map_err(|e| StarError::serialization_error(format!("Lock error: {e}")))?;
            writer
                .write_all(data)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
            self.written_bytes += data.len();
        }
        Ok(())
    }

    /// Flush the internal buffer to the writer
    fn flush_buffer(&mut self) -> StarResult<()> {
        if !self.buffer.is_empty() {
            let data = if self.config.compress_chunks {
                self.compress_chunk(&self.buffer)?
            } else {
                self.buffer.clone()
            };

            let mut writer = self
                .writer
                .lock()
                .map_err(|e| StarError::serialization_error(format!("Lock error: {e}")))?;
            writer
                .write_all(&data)
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
            writer
                .flush()
                .map_err(|e| StarError::serialization_error(e.to_string()))?;

            self.written_bytes += data.len();
            self.buffer.clear();
        }
        Ok(())
    }

    /// Compress a chunk of data (placeholder implementation)
    fn compress_chunk(&self, data: &[u8]) -> StarResult<Vec<u8>> {
        // For now, return data as-is
        // In a full implementation, this would use actual compression libraries
        Ok(data.to_vec())
    }

    /// Serialize triples in streaming fashion
    pub fn serialize_triples_streaming<I>(
        &mut self,
        triples: I,
        format: StarFormat,
    ) -> StarResult<()>
    where
        I: Iterator<Item = StarTriple>,
    {
        for chunk in ChunkedIterator::new(triples, self.config.chunk_size) {
            self.serialize_chunk(&chunk, format)?;
        }
        self.flush_buffer()?;
        Ok(())
    }

    /// Serialize a chunk of triples
    fn serialize_chunk(&mut self, chunk: &[StarTriple], format: StarFormat) -> StarResult<()> {
        for triple in chunk {
            let line = match format {
                StarFormat::NTriplesStar => {
                    let subject = Self::format_term_ntriples(&triple.subject)?;
                    let predicate = Self::format_term_ntriples(&triple.predicate)?;
                    let object = Self::format_term_ntriples(&triple.object)?;
                    format!("{subject} {predicate} {object} .\n")
                }
                StarFormat::TurtleStar => {
                    let subject = self.format_term_turtle(&triple.subject)?;
                    let predicate = self.format_term_turtle(&triple.predicate)?;
                    let object = self.format_term_turtle(&triple.object)?;
                    format!("{subject} {predicate} {object} .\n")
                }
                StarFormat::TrigStar => {
                    // TriG-star format with default graph
                    let subject = Self::format_term_ntriples(&triple.subject)?;
                    let predicate = Self::format_term_ntriples(&triple.predicate)?;
                    let object = Self::format_term_ntriples(&triple.object)?;
                    format!("{subject} {predicate} {object} .\n")
                }
                StarFormat::NQuadsStar => {
                    // N-Quads-star format with default graph
                    let subject = Self::format_term_ntriples(&triple.subject)?;
                    let predicate = Self::format_term_ntriples(&triple.predicate)?;
                    let object = Self::format_term_ntriples(&triple.object)?;
                    format!("{subject} {predicate} {object} <> .\n") // <> represents default graph
                }
                StarFormat::JsonLdStar => {
                    // JSON-LD-star format - streaming not fully supported yet
                    // For now, use a simple N-Triples-like representation
                    let subject = Self::format_term_ntriples(&triple.subject)?;
                    let predicate = Self::format_term_ntriples(&triple.predicate)?;
                    let object = Self::format_term_ntriples(&triple.object)?;
                    format!("{subject} {predicate} {object} .\n")
                }
            };
            self.write_buffered(line.as_bytes())?;
        }
        Ok(())
    }

    /// Format term for N-Triples output
    fn format_term_ntriples(term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(format!("<{}>", node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", StarSerializer::escape_literal(&literal.value));
                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{lang}"));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^<{}>", datatype.iri));
                }
                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = Self::format_term_ntriples(&triple.subject)?;
                let predicate = Self::format_term_ntriples(&triple.predicate)?;
                let object = Self::format_term_ntriples(&triple.object)?;
                Ok(format!("<< {subject} {predicate} {object} >>"))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Format term for Turtle output
    fn format_term_turtle(&self, term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(self.context.compress_iri(&node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", StarSerializer::escape_literal(&literal.value));
                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{lang}"));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^{}", self.context.compress_iri(&datatype.iri)));
                }
                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = self.format_term_turtle(&triple.subject)?;
                let predicate = self.format_term_turtle(&triple.predicate)?;
                let object = self.format_term_turtle(&triple.object)?;
                Ok(format!("<< {subject} {predicate} {object} >>"))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }
}
