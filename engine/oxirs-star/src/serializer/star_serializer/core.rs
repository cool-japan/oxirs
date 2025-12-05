//! Core RDF-star serialization methods

use super::super::config::*;
use super::super::parallel::ParallelSerializer;
use super::super::streaming::StreamingSerializer;
use super::StarSerializer;
use crate::model::{StarGraph, StarQuad, StarTerm, StarTriple};
use crate::parser::StarFormat;
use crate::{StarConfig, StarError, StarResult};
use flate2::write::GzEncoder;
use flate2::Compression;
use lz4_flex::frame::FrameEncoder;
use std::io::{BufWriter, Write};
use tracing::{debug, span, Level};
use zstd::stream::write::Encoder as ZstdEncoder;

impl StarSerializer {
    /// Create a new serializer with default configuration
    pub fn new() -> Self {
        Self {
            config: StarConfig::default(),
            simd_escaper: super::super::simd_escape::SimdEscaper::new(),
        }
    }

    /// Create a new serializer with custom configuration
    pub fn with_config(config: StarConfig) -> Self {
        Self {
            config,
            simd_escaper: super::super::simd_escape::SimdEscaper::new(),
        }
    }

    /// Serialize a StarGraph to a writer in the specified format
    pub fn serialize<W: Write>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
    ) -> StarResult<()> {
        let span = span!(Level::INFO, "serialize_rdf_star", format = ?format);
        let _enter = span.enter();

        match format {
            StarFormat::TurtleStar => self.serialize_turtle_star(graph, writer),
            StarFormat::NTriplesStar => self.serialize_ntriples_star(graph, writer),
            StarFormat::TrigStar => self.serialize_trig_star(graph, writer),
            StarFormat::NQuadsStar => self.serialize_nquads_star(graph, writer),
            StarFormat::JsonLdStar => self.serialize_jsonld_star(graph, writer),
        }
    }

    /// Serialize to string in the specified format
    pub fn serialize_to_string(&self, graph: &StarGraph, format: StarFormat) -> StarResult<String> {
        let mut buffer = Vec::new();
        self.serialize(graph, &mut buffer, format)?;
        String::from_utf8(buffer).map_err(|e| StarError::serialization_error(e.to_string()))
    }

    /// Serialize with advanced options (streaming, compression, parallel processing)
    pub fn serialize_with_options<W: Write + Send + 'static>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
        options: &SerializationOptions,
    ) -> StarResult<()> {
        let span = span!(Level::INFO, "serialize_with_options", format = ?format, streaming = options.streaming, parallel = options.parallel);
        let _enter = span.enter();

        // Choose serialization strategy based on options and graph size
        let triple_count = graph.total_len();

        if options.parallel && triple_count > options.batch_size {
            debug!("Using parallel serialization for {} triples", triple_count);
            let parallel_serializer =
                ParallelSerializer::new(options.max_threads, options.batch_size);
            parallel_serializer.serialize_parallel(graph, writer, format, options)
        } else if options.streaming && triple_count > 50000 {
            debug!("Using streaming serialization for {} triples", triple_count);
            let streaming_config = StreamingConfig {
                chunk_size: options.batch_size,
                buffer_capacity: options.buffer_size,
                enable_buffering: true,
                memory_threshold: options.buffer_size * 64, // 64x buffer size
                compress_chunks: options.compression != CompressionType::None,
            };
            let mut streaming_serializer = StreamingSerializer::new(writer, streaming_config);
            streaming_serializer
                .serialize_triples_streaming(graph.triples().iter().cloned(), format)
        } else {
            debug!("Using standard serialization for {} triples", triple_count);
            // Apply compression wrapper if requested
            if options.compression != CompressionType::None {
                let compressed_writer =
                    self.create_compressed_writer(writer, options.compression)?;
                self.serialize(graph, compressed_writer, format)
            } else {
                self.serialize(graph, writer, format)
            }
        }
    }

    /// Create a compressed writer based on compression type
    fn create_compressed_writer<W: Write + 'static>(
        &self,
        writer: W,
        compression: CompressionType,
    ) -> StarResult<Box<dyn Write>> {
        match compression {
            CompressionType::None => Ok(Box::new(writer)),
            CompressionType::Gzip => {
                debug!("Creating Gzip compressed writer");
                let encoder = GzEncoder::new(writer, Compression::default());
                Ok(Box::new(encoder))
            }
            CompressionType::Zstd => {
                debug!("Creating Zstd compressed writer");
                let encoder = ZstdEncoder::new(writer, 3).map_err(|e| {
                    StarError::serialization_error(format!("Zstd encoder error: {e}"))
                })?;
                Ok(Box::new(encoder))
            }
            CompressionType::Lz4 => {
                debug!("Creating LZ4 compressed writer");
                let encoder = FrameEncoder::new(writer);
                Ok(Box::new(encoder))
            }
        }
    }

    /// Serialize large dataset using streaming approach
    pub fn serialize_streaming<W: Write>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
        chunk_size: usize,
    ) -> StarResult<()> {
        let span =
            span!(Level::DEBUG, "serialize_streaming", format = ?format, chunk_size = chunk_size);
        let _enter = span.enter();

        let config = StreamingConfig {
            chunk_size,
            ..Default::default()
        };
        let mut streaming_serializer = StreamingSerializer::new(writer, config);
        streaming_serializer
            .serialize_triples_streaming(graph.triples().iter().cloned(), format)?;

        debug!(
            "Streamed {} triples in format {:?}",
            graph.total_len(),
            format
        );
        Ok(())
    }

    /// Serialize using parallel processing for large graphs
    pub fn serialize_parallel<W: Write + Send + 'static>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
        num_threads: usize,
        batch_size: usize,
    ) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_parallel", format = ?format, num_threads = num_threads, batch_size = batch_size);
        let _enter = span.enter();

        let parallel_serializer = ParallelSerializer::new(num_threads, batch_size);
        let options = SerializationOptions::default();
        parallel_serializer.serialize_parallel(graph, writer, format, &options)?;

        debug!(
            "Parallel serialization completed for {} triples using {} threads",
            graph.total_len(),
            num_threads
        );
        Ok(())
    }

    /// Auto-detect optimal serialization strategy based on graph characteristics
    pub fn serialize_optimized<W: Write + Send + 'static>(
        &self,
        graph: &StarGraph,
        writer: W,
        format: StarFormat,
    ) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_optimized", format = ?format);
        let _enter = span.enter();

        let triple_count = graph.total_len();
        let quoted_count = graph.count_quoted_triples();
        let complexity_score = quoted_count as f64 / triple_count.max(1) as f64;

        debug!(
            "Graph analysis: {} triples, {} quoted (complexity: {:.2})",
            triple_count, quoted_count, complexity_score
        );

        let mut options = SerializationOptions::default();

        // Configure based on graph characteristics
        if triple_count > 1_000_000 {
            // Very large graph - use streaming
            options.streaming = true;
            options.compression = CompressionType::Zstd; // High performance compression
            options.buffer_size = 4 * 1024 * 1024; // 4MB buffer
            debug!("Selected streaming strategy for very large graph");
        } else if triple_count > 100_000 && complexity_score < 0.1 {
            // Large simple graph - use parallel processing
            options.parallel = true;
            options.max_threads = std::cmp::min(8, 4); // Use 4 as default thread count
            options.batch_size = 25000;
            debug!("Selected parallel strategy for large simple graph");
        } else if complexity_score > 0.3 {
            // Complex graph with many quoted triples - use smaller batches
            options.batch_size = 5000;
            options.buffer_size = 512 * 1024; // 512KB buffer
            debug!("Selected conservative strategy for complex graph");
        }
        // else: use default strategy for smaller/simpler graphs

        self.serialize_with_options(graph, writer, format, &options)
    }

    /// Get memory usage estimation for serialization
    pub fn estimate_memory_usage(
        &self,
        graph: &StarGraph,
        format: StarFormat,
        options: &SerializationOptions,
    ) -> usize {
        let base_memory = self.estimate_size(graph, format);

        let memory_multiplier = if options.parallel {
            // Parallel processing uses more memory for batching
            2.5
        } else if options.streaming {
            // Streaming uses less memory
            0.5
        } else {
            1.0
        };

        let buffer_overhead = if options.streaming {
            options.buffer_size * 2 // Double buffering
        } else {
            options.batch_size * 100 // Rough estimate of batch memory
        };

        ((base_memory as f64 * memory_multiplier) as usize) + buffer_overhead
    }

    /// Serialize to Turtle-star format
    pub fn serialize_turtle_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_turtle_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let mut context = SerializationContext::new();

        // Add common prefixes
        self.add_common_prefixes(&mut context);

        // Write prefixes
        self.write_turtle_prefixes(&mut buf_writer, &context)?;

        // Write triples
        for triple in graph.triples() {
            self.write_turtle_triple(&mut buf_writer, triple, &context)?;
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        debug!("Serialized {} triples in Turtle-star format", graph.len());
        Ok(())
    }

    /// Write Turtle-star prefixes
    fn write_turtle_prefixes<W: Write>(
        &self,
        writer: &mut W,
        context: &SerializationContext,
    ) -> StarResult<()> {
        for (prefix, namespace) in &context.prefixes {
            writeln!(writer, "@prefix {prefix}: <{namespace}> .")
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        if !context.prefixes.is_empty() {
            writeln!(writer).map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        Ok(())
    }

    /// Write a single Turtle-star triple
    fn write_turtle_triple<W: Write>(
        &self,
        writer: &mut W,
        triple: &StarTriple,
        context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = self.format_term(&triple.subject, context)?;
        let predicate_str = self.format_term(&triple.predicate, context)?;
        let object_str = self.format_term(&triple.object, context)?;

        if context.pretty_print {
            writeln!(
                writer,
                "{}{} {} {} .",
                context.current_indent(),
                subject_str,
                predicate_str,
                object_str
            )
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        } else {
            writeln!(writer, "{subject_str} {predicate_str} {object_str} .")
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        Ok(())
    }

    /// Serialize to N-Triples-star format
    pub fn serialize_ntriples_star<W: Write>(
        &self,
        graph: &StarGraph,
        writer: W,
    ) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_ntriples_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let context = SerializationContext::new(); // N-Triples doesn't use prefixes

        for triple in graph.triples() {
            self.write_ntriples_triple(&mut buf_writer, triple, &context)?;
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        debug!(
            "Serialized {} triples in N-Triples-star format",
            graph.len()
        );
        Ok(())
    }

    /// Write a single N-Triples-star triple
    fn write_ntriples_triple<W: Write>(
        &self,
        writer: &mut W,
        triple: &StarTriple,
        _context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = Self::format_term_ntriples(&triple.subject)?;
        let predicate_str = Self::format_term_ntriples(&triple.predicate)?;
        let object_str = Self::format_term_ntriples(&triple.object)?;

        writeln!(writer, "{subject_str} {predicate_str} {object_str} .")
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        Ok(())
    }

    /// Serialize to TriG-star format (with named graphs)
    pub fn serialize_trig_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_trig_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let mut context = SerializationContext::new();
        context.pretty_print = true;

        // Add common prefixes
        self.add_common_prefixes(&mut context);

        // Write prefixes first
        self.write_turtle_prefixes(&mut buf_writer, &context)?;

        // Serialize default graph if it has triples
        if !graph.triples().is_empty() {
            writeln!(buf_writer, "{{")
                .map_err(|e| StarError::serialization_error(e.to_string()))?;

            context.increase_indent();
            for triple in graph.triples() {
                self.write_turtle_triple(&mut buf_writer, triple, &context)?;
            }
            context.decrease_indent();

            writeln!(buf_writer, "}}")
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
            writeln!(buf_writer).map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        // Serialize named graphs
        for graph_name in graph.named_graph_names() {
            if let Some(named_triples) = graph.named_graph_triples(graph_name) {
                if !named_triples.is_empty() {
                    // Write graph declaration
                    let graph_term = self.parse_graph_name(graph_name, &context)?;
                    writeln!(buf_writer, "{graph_term} {{")
                        .map_err(|e| StarError::serialization_error(e.to_string()))?;

                    context.increase_indent();
                    for triple in named_triples {
                        self.write_turtle_triple(&mut buf_writer, triple, &context)?;
                    }
                    context.decrease_indent();

                    writeln!(buf_writer, "}}")
                        .map_err(|e| StarError::serialization_error(e.to_string()))?;
                    writeln!(buf_writer)
                        .map_err(|e| StarError::serialization_error(e.to_string()))?;
                }
            }
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        debug!(
            "Serialized {} quads ({} total triples) in TriG-star format",
            graph.quad_len(),
            graph.total_len()
        );
        Ok(())
    }

    /// Serialize to N-Quads-star format
    pub fn serialize_nquads_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_nquads_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let context = SerializationContext::new(); // N-Quads doesn't use prefixes

        // Serialize all quads from the graph (including both default and named graphs)
        for quad in graph.quads() {
            self.write_nquads_quad_complete(&mut buf_writer, quad, &context)?;
        }

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        debug!(
            "Serialized {} quads ({} total triples) in N-Quads-star format",
            graph.quad_len(),
            graph.total_len()
        );
        Ok(())
    }

    /// Write a single N-Quads-star quad with proper graph context
    fn write_nquads_quad_complete<W: Write>(
        &self,
        writer: &mut W,
        quad: &StarQuad,
        _context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = Self::format_term_ntriples(&quad.subject)?;
        let predicate_str = Self::format_term_ntriples(&quad.predicate)?;
        let object_str = Self::format_term_ntriples(&quad.object)?;

        if let Some(ref graph_term) = quad.graph {
            // Named graph quad
            let graph_str = Self::format_term_ntriples(graph_term)?;
            writeln!(
                writer,
                "{subject_str} {predicate_str} {object_str} {graph_str} ."
            )
            .map_err(|e| StarError::serialization_error(e.to_string()))?;
        } else {
            // Default graph quad (triple)
            writeln!(writer, "{subject_str} {predicate_str} {object_str} .")
                .map_err(|e| StarError::serialization_error(e.to_string()))?;
        }

        Ok(())
    }

    /// Write a single N-Quads-star quad (triple + optional graph) - legacy method
    #[allow(dead_code)]
    fn write_nquads_quad<W: Write>(
        &self,
        writer: &mut W,
        triple: &StarTriple,
        _context: &SerializationContext,
    ) -> StarResult<()> {
        let subject_str = Self::format_term_ntriples(&triple.subject)?;
        let predicate_str = Self::format_term_ntriples(&triple.predicate)?;
        let object_str = Self::format_term_ntriples(&triple.object)?;

        // Default graph (no graph component)
        writeln!(writer, "{subject_str} {predicate_str} {object_str} .")
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        Ok(())
    }

    /// Serialize to JSON-LD-star format
    pub fn serialize_jsonld_star<W: Write>(&self, graph: &StarGraph, writer: W) -> StarResult<()> {
        let span = span!(Level::DEBUG, "serialize_jsonld_star");
        let _enter = span.enter();

        let mut buf_writer = BufWriter::new(writer);
        let context = SerializationContext::new();

        // Create JSON-LD context
        let mut jsonld_document = serde_json::Map::new();

        // Add JSON-LD context
        let mut context_obj = serde_json::Map::new();
        context_obj.insert(
            "@vocab".to_string(),
            serde_json::Value::String("http://example.org/".to_string()),
        );

        // Add common prefixes
        context_obj.insert(
            "rdf".to_string(),
            serde_json::Value::String("http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string()),
        );
        context_obj.insert(
            "rdfs".to_string(),
            serde_json::Value::String("http://www.w3.org/2000/01/rdf-schema#".to_string()),
        );
        context_obj.insert(
            "xsd".to_string(),
            serde_json::Value::String("http://www.w3.org/2001/XMLSchema#".to_string()),
        );

        jsonld_document.insert(
            "@context".to_string(),
            serde_json::Value::Object(context_obj),
        );

        // Group quads by subject
        let mut subjects = std::collections::HashMap::new();

        // Process all quads
        for quad in graph.quads() {
            self.add_quad_to_jsonld(&mut subjects, quad)?;
        }

        // Convert subjects to JSON-LD array
        let mut graph_array = Vec::new();
        for (subject_str, properties) in subjects {
            let mut subject_obj = serde_json::Map::new();

            // Add @id for non-blank nodes
            if !subject_str.starts_with("_:") {
                subject_obj.insert("@id".to_string(), serde_json::Value::String(subject_str));
            }

            // Add properties
            for (predicate, values) in properties {
                subject_obj.insert(predicate, serde_json::Value::Array(values));
            }

            graph_array.push(serde_json::Value::Object(subject_obj));
        }

        jsonld_document.insert("@graph".to_string(), serde_json::Value::Array(graph_array));

        // Write JSON with pretty printing
        let json_output = if context.pretty_print {
            serde_json::to_string_pretty(&jsonld_document)
        } else {
            serde_json::to_string(&jsonld_document)
        }
        .map_err(|e| StarError::serialization_error(format!("JSON serialization error: {e}")))?;

        buf_writer
            .write_all(json_output.as_bytes())
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        buf_writer
            .flush()
            .map_err(|e| StarError::serialization_error(e.to_string()))?;

        debug!(
            "Serialized {} quads in JSON-LD-star format",
            graph.quad_len()
        );
        Ok(())
    }

    /// Add a quad to the JSON-LD structure
    fn add_quad_to_jsonld(
        &self,
        subjects: &mut std::collections::HashMap<
            String,
            std::collections::HashMap<String, Vec<serde_json::Value>>,
        >,
        quad: &StarQuad,
    ) -> StarResult<()> {
        let subject_str = self.term_to_jsonld_id(&quad.subject)?;
        let predicate_str = self.term_to_jsonld_predicate(&quad.predicate)?;
        let object_value = self.term_to_jsonld_value(&quad.object)?;

        let subject_props = subjects.entry(subject_str).or_default();
        let prop_values = subject_props.entry(predicate_str).or_default();
        prop_values.push(object_value);

        Ok(())
    }

    /// Convert a StarTerm to JSON-LD @id format
    pub fn term_to_jsonld_id(&self, term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(node.iri.clone()),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::QuotedTriple(triple) => {
                // For quoted triples as subjects, create a special identifier
                Ok(format!("_:qt_{}", self.hash_triple(triple)))
            }
            _ => Err(StarError::serialization_error(
                "Invalid subject term".to_string(),
            )),
        }
    }

    /// Convert a StarTerm to JSON-LD predicate format
    pub fn term_to_jsonld_predicate(&self, term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(node.iri.clone()),
            _ => Err(StarError::serialization_error(
                "Invalid predicate term".to_string(),
            )),
        }
    }

    /// Convert a StarTerm to JSON-LD value format
    #[allow(clippy::only_used_in_recursion)]
    pub fn term_to_jsonld_value(&self, term: &StarTerm) -> StarResult<serde_json::Value> {
        match term {
            StarTerm::NamedNode(node) => {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@id".to_string(),
                    serde_json::Value::String(node.iri.clone()),
                );
                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::BlankNode(node) => {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@id".to_string(),
                    serde_json::Value::String(format!("_:{}", node.id)),
                );
                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::Literal(literal) => {
                let mut obj = serde_json::Map::new();

                // Add the value
                if let Ok(num) = literal.value.parse::<f64>() {
                    // Numeric literal
                    obj.insert(
                        "@value".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(num)
                                .unwrap_or(serde_json::Number::from(0)),
                        ),
                    );
                } else if literal.value == "true" || literal.value == "false" {
                    // Boolean literal
                    obj.insert(
                        "@value".to_string(),
                        serde_json::Value::Bool(literal.value == "true"),
                    );
                } else {
                    // String literal
                    obj.insert(
                        "@value".to_string(),
                        serde_json::Value::String(literal.value.clone()),
                    );
                }

                // Add datatype if present
                if let Some(ref datatype) = literal.datatype {
                    obj.insert(
                        "@type".to_string(),
                        serde_json::Value::String(datatype.iri.clone()),
                    );
                }

                // Add language if present
                if let Some(ref lang) = literal.language {
                    obj.insert(
                        "@language".to_string(),
                        serde_json::Value::String(lang.clone()),
                    );
                }

                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::QuotedTriple(triple) => {
                // RDF-star extension: represent quoted triple as annotation
                let mut annotation_obj = serde_json::Map::new();
                annotation_obj.insert(
                    "subject".to_string(),
                    self.term_to_jsonld_value(&triple.subject)?,
                );
                annotation_obj.insert(
                    "predicate".to_string(),
                    self.term_to_jsonld_value(&triple.predicate)?,
                );
                annotation_obj.insert(
                    "object".to_string(),
                    self.term_to_jsonld_value(&triple.object)?,
                );

                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@annotation".to_string(),
                    serde_json::Value::Object(annotation_obj),
                );
                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::Variable(var) => {
                // Variables are typically used in SPARQL queries, not in serialized data
                // For JSON-LD, we'll represent them as special objects
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@variable".to_string(),
                    serde_json::Value::String(var.name.clone()),
                );
                Ok(serde_json::Value::Object(obj))
            }
        }
    }

    /// Generate a hash for a triple (simple implementation)
    fn hash_triple(&self, triple: &StarTriple) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{triple:?}").hash(&mut hasher);
        hasher.finish()
    }

    /// Format a StarTerm for Turtle-star (with prefix compression)
    #[allow(clippy::only_used_in_recursion)]
    fn format_term(&self, term: &StarTerm, context: &SerializationContext) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(context.compress_iri(&node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", Self::escape_literal_static(&literal.value));

                if let Some(ref lang) = literal.language {
                    result.push_str(&format!("@{lang}"));
                } else if let Some(ref datatype) = literal.datatype {
                    result.push_str(&format!("^^{}", context.compress_iri(&datatype.iri)));
                }

                Ok(result)
            }
            StarTerm::QuotedTriple(triple) => {
                let subject = self.format_term(&triple.subject, context)?;
                let predicate = self.format_term(&triple.predicate, context)?;
                let object = self.format_term(&triple.object, context)?;
                Ok(format!("<< {subject} {predicate} {object} >>"))
            }
            StarTerm::Variable(var) => Ok(format!("?{}", var.name)),
        }
    }

    /// Format a StarTerm for N-Triples-star (full IRIs, no prefixes)
    fn format_term_ntriples(term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(format!("<{}>", node.iri)),
            StarTerm::BlankNode(node) => Ok(format!("_:{}", node.id)),
            StarTerm::Literal(literal) => {
                let mut result = format!("\"{}\"", Self::escape_literal_static(&literal.value));

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

    /// Escape special characters in literals (SIMD-accelerated)
    pub fn escape_literal(&self, value: &str) -> String {
        self.simd_escaper.escape_literal(value)
    }

    /// Static escape function for backward compatibility
    pub fn escape_literal_static(value: &str) -> String {
        // Use a temporary escaper for static calls
        let escaper = super::super::simd_escape::SimdEscaper::new();
        escaper.escape_literal(value)
    }

    /// Add common namespace prefixes
    fn add_common_prefixes(&self, context: &mut SerializationContext) {
        context.add_prefix("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
        context.add_prefix("rdfs", "http://www.w3.org/2000/01/rdf-schema#");
        context.add_prefix("owl", "http://www.w3.org/2002/07/owl#");
        context.add_prefix("xsd", "http://www.w3.org/2001/XMLSchema#");
        context.add_prefix("foaf", "http://xmlns.com/foaf/0.1/");
        context.add_prefix("dc", "http://purl.org/dc/terms/");
    }

    /// Parse a graph name string back to a term for TriG serialization
    fn parse_graph_name(
        &self,
        graph_name: &str,
        context: &SerializationContext,
    ) -> StarResult<String> {
        if graph_name.starts_with("_:") {
            // Blank node graph name
            Ok(graph_name.to_string())
        } else {
            // Named node graph name - compress with prefixes if possible
            Ok(context.compress_iri(graph_name))
        }
    }

    /// Static method for converting StarTerm to JSON-LD predicate format
    pub fn term_to_jsonld_predicate_static(term: &StarTerm) -> StarResult<String> {
        match term {
            StarTerm::NamedNode(node) => Ok(node.iri.clone()),
            _ => Err(StarError::serialization_error(
                "Only named nodes can be used as predicates in JSON-LD".to_string(),
            )),
        }
    }

    /// Static method for converting StarTerm to JSON-LD value format  
    pub fn term_to_jsonld_value_static(term: &StarTerm) -> StarResult<serde_json::Value> {
        match term {
            StarTerm::NamedNode(node) => {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@id".to_string(),
                    serde_json::Value::String(node.iri.clone()),
                );
                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::BlankNode(node) => {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@id".to_string(),
                    serde_json::Value::String(format!("_:{}", node.id)),
                );
                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::Literal(literal) => {
                if let Some(ref lang) = literal.language {
                    let mut obj = serde_json::Map::new();
                    obj.insert(
                        "@value".to_string(),
                        serde_json::Value::String(literal.value.clone()),
                    );
                    obj.insert(
                        "@language".to_string(),
                        serde_json::Value::String(lang.clone()),
                    );
                    Ok(serde_json::Value::Object(obj))
                } else if let Some(ref datatype) = literal.datatype {
                    let mut obj = serde_json::Map::new();
                    obj.insert(
                        "@value".to_string(),
                        serde_json::Value::String(literal.value.clone()),
                    );
                    obj.insert(
                        "@type".to_string(),
                        serde_json::Value::String(datatype.iri.clone()),
                    );
                    Ok(serde_json::Value::Object(obj))
                } else {
                    // Plain literal defaults to string
                    Ok(serde_json::Value::String(literal.value.clone()))
                }
            }
            StarTerm::QuotedTriple(triple) => {
                // JSON-LD-star representation of quoted triples as annotations
                let mut annotation_obj = serde_json::Map::new();
                annotation_obj.insert(
                    "subject".to_string(),
                    Self::term_to_jsonld_value_static(&triple.subject)?,
                );
                annotation_obj.insert(
                    "predicate".to_string(),
                    Self::term_to_jsonld_value_static(&triple.predicate)?,
                );
                annotation_obj.insert(
                    "object".to_string(),
                    Self::term_to_jsonld_value_static(&triple.object)?,
                );

                let mut obj = serde_json::Map::new();
                obj.insert(
                    "@annotation".to_string(),
                    serde_json::Value::Object(annotation_obj),
                );
                Ok(serde_json::Value::Object(obj))
            }
            StarTerm::Variable(var) => Err(StarError::serialization_error(format!(
                "Variables cannot be serialized to JSON-LD: ?{}",
                var.name
            ))),
        }
    }
}
