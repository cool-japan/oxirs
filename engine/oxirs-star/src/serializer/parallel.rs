//! Parallel serialization for multi-threaded processing

use super::config::SerializationOptions;
use super::star_serializer::StarSerializer;
use crate::model::{StarGraph, StarTerm, StarTriple};
use crate::parser::StarFormat;
use crate::{StarError, StarResult};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;

/// Parallel serializer for multi-threaded processing
pub struct ParallelSerializer {
    #[allow(dead_code)]
    num_threads: usize,
    batch_size: usize,
}

impl ParallelSerializer {
    /// Create a new parallel serializer
    pub fn new(num_threads: usize, batch_size: usize) -> Self {
        Self {
            num_threads,
            batch_size,
        }
    }

    /// Static method for formatting term in N-Triples
    fn format_term_ntriples(term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(format!("<{}>", node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!(
                    "\"{}\"",
                    StarSerializer::escape_literal_static(&literal.value)
                );
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

    /// Static method for formatting term in Turtle
    fn format_term_turtle(term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(format!("<{}>", node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!(
                    "\"{}\"",
                    StarSerializer::escape_literal_static(&literal.value)
                );
                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{lang}"));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^<{}>", datatype.iri));
                }
                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = Self::format_term_turtle(&triple.subject)?;
                let predicate = Self::format_term_turtle(&triple.predicate)?;
                let object = Self::format_term_turtle(&triple.object)?;
                Ok(format!("<< {subject} {predicate} {object} >>"))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Serialize graph using multiple threads
    pub fn serialize_parallel<W: Write + Send + 'static>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
        _options: &SerializationOptions,
    ) -> StarResult<()> {
        let writer = Arc::new(Mutex::new(writer));
        let triples: Vec<_> = graph.triples().iter().collect();

        // Split into batches for parallel processing
        let batches: Vec<_> = triples.chunks(self.batch_size).collect();
        let mut handles = Vec::new();

        for batch in batches {
            let batch: Vec<StarTriple> = batch.iter().map(|t| (*t).clone()).collect();
            let writer_clone = Arc::clone(&writer);
            let format_clone = format;

            let handle =
                thread::spawn(move || Self::process_batch(batch, writer_clone, format_clone));
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().map_err(|e| {
                StarError::serialization_error(format!("Thread join error: {e:?}"))
            })??;
        }

        Ok(())
    }

    /// Process a batch of triples in a worker thread
    fn process_batch<W: Write>(
        batch: Vec<StarTriple>,
        writer: Arc<Mutex<W>>,
        format: StarFormat,
    ) -> StarResult<()> {
        let mut output = Vec::new();

        for triple in batch {
            let line = match format {
                StarFormat::NTriplesStar => {
                    format!(
                        "{} {} {} .\n",
                        Self::format_term_ntriples(&triple.subject)?,
                        Self::format_term_ntriples(&triple.predicate)?,
                        Self::format_term_ntriples(&triple.object)?
                    )
                }
                StarFormat::TurtleStar => {
                    format!(
                        "{} {} {} .\n",
                        Self::format_term_turtle(&triple.subject)?,
                        Self::format_term_turtle(&triple.predicate)?,
                        Self::format_term_turtle(&triple.object)?
                    )
                }
                StarFormat::TrigStar => {
                    // TriG-star format with default graph
                    format!(
                        "{} {} {} .\n",
                        Self::format_term_ntriples(&triple.subject)?,
                        Self::format_term_ntriples(&triple.predicate)?,
                        Self::format_term_ntriples(&triple.object)?
                    )
                }
                StarFormat::NQuadsStar => {
                    // N-Quads-star format with default graph
                    format!(
                        "{} {} {} <> .\n",
                        Self::format_term_ntriples(&triple.subject)?,
                        Self::format_term_ntriples(&triple.predicate)?,
                        Self::format_term_ntriples(&triple.object)?
                    )
                }
                StarFormat::JsonLdStar => {
                    // JSON-LD-star format - streaming as individual objects
                    let subject_value =
                        StarSerializer::term_to_jsonld_value_static(&triple.subject)?;
                    let predicate_str =
                        StarSerializer::term_to_jsonld_predicate_static(&triple.predicate)?;
                    let object_value = StarSerializer::term_to_jsonld_value_static(&triple.object)?;

                    let mut triple_obj = serde_json::Map::new();
                    triple_obj.insert("@id".to_string(), subject_value);
                    triple_obj.insert(predicate_str, object_value);

                    let json_str = serde_json::to_string(&serde_json::Value::Object(triple_obj))
                        .map_err(|e| {
                            StarError::serialization_error(format!("JSON serialization error: {e}"))
                        })?;
                    format!("{json_str}\n")
                }
            };
            output.extend_from_slice(line.as_bytes());
        }

        let mut writer = writer
            .lock()
            .map_err(|e| StarError::serialization_error(format!("Lock error: {e}")))?;
        writer
            .write_all(&output)
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        Ok(())
    }
}

/// Chunked iterator for processing large collections in batches
pub(crate) struct ChunkedIterator<I> {
    inner: I,
    chunk_size: usize,
}

impl<I> ChunkedIterator<I> {
    pub(crate) fn new(inner: I, chunk_size: usize) -> Self {
        Self { inner, chunk_size }
    }
}

impl<I, T> Iterator for ChunkedIterator<I>
where
    I: Iterator<Item = T>,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = Vec::with_capacity(self.chunk_size);
        for _ in 0..self.chunk_size {
            match self.inner.next() {
                Some(item) => chunk.push(item),
                None => break,
            }
        }
        if chunk.is_empty() {
            None
        } else {
            Some(chunk)
        }
    }
}
